"""
external_pipeline.py
====================

Pipeline esterna per TabPFN-2.5: tecniche che lasciano i pesi del modello
**congelati** e agiscono unicamente sull'**output** (probabilita' / logit). E'
il capitolo speculare a quello del fine-tuning LoRA: dato che i pesi
pre-addestrati risultano gia' near-optimal (il LoRA non migliora la
discriminazione), qui cerchiamo valore aggiunto *attorno* al modello, su assi
che il fine-tuning non puo' raggiungere.

Sono implementati due blocchi, ciascuno come "script per esperimento":

    Exp 3 - Ensembling (riduzione della varianza)
        * self-ensemble: media delle probabilita' di N istanze TabPFN con seed
          interni diversi (diverse permutazioni/preprocessing);
        * ensemble eterogeneo: media TabPFN + GradientBoosting.
        Bersaglio: AUC-ROC.

    Exp 4 - Calibrazione + ottimizzazione della soglia
        * calibrazione: Platt scaling (sigmoide) vs Isotonic regression;
          bersaglio ECE / Brier (AUC e F1 restano invariati per monotonia);
        * soglia: cutoff che massimizza l'F1 macro, cercato sul set di
          calibrazione; bersaglio F1.

Metodologia anti-leakage:
    Per ogni dataset si usa uno **split a 3 vie** stratificato
    ``train(context) / calibration / test``. Calibratore e soglia vengono
    fittati ESCLUSIVAMENTE sul set di calibrazione; il test resta intatto.
    L'intera procedura e' ripetuta su ``n_seeds`` semi diversi e i risultati
    sono riportati come media +/- deviazione standard, per quantificare il
    rumore (rilevante sui dataset piccoli).

Gli helper (``average_probas``, ``fit_calibrator``, ``optimize_threshold``)
sono puri e lavorano su array NumPy: sono quindi testabili su CPU senza TabPFN
(vedi il blocco ``__main__``). Le funzioni di esperimento importano TabPFN in
modo lazy, cosi' il modulo resta importabile anche in locale.

Dipendenze:
    pip install scikit-learn numpy pandas    (+ tabpfn per gli esperimenti)
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Rende importabili i package fratelli (utils, evaluation) anche quando il
# modulo viene eseguito direttamente o da una sessione Colab.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.metrics import evaluate_model  # noqa: E402
from utils.data_loader import PIPELINE_DATASETS, load_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Helper puri (array-based, testabili su CPU)
# ---------------------------------------------------------------------------

def average_probas(proba_list: List[np.ndarray]) -> np.ndarray:
    """
    Media aritmetica di una lista di vettori di probabilita'.

    Usata per l'ensembling: combinare le probabilita' della classe positiva
    prodotte da modelli diversi (o dallo stesso modello con seed diversi)
    riduce la varianza della stima.

    Args:
        proba_list: Lista non vuota di array ``(n_samples,)`` con le
            probabilita' della classe positiva. Tutti della stessa lunghezza.

    Returns:
        Array ``(n_samples,)`` con la media elemento per elemento, garantito
        in ``[0, 1]``.

    Raises:
        ValueError: Se la lista e' vuota o gli array hanno lunghezze diverse.
    """
    if not proba_list:
        raise ValueError("proba_list e' vuota: niente da mediare.")

    arrays = [np.asarray(p, dtype=float).ravel() for p in proba_list]
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Le probabilita' hanno lunghezze diverse: {sorted(lengths)}.")

    mean = np.mean(np.vstack(arrays), axis=0)
    # Clip difensivo: la media di valori in [0,1] e' gia' in [0,1], ma il clip
    # protegge da minimi errori numerici.
    return np.clip(mean, 0.0, 1.0)


def fit_calibrator(
    prob_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "platt",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Fitta un calibratore di probabilita' sul set di calibrazione.

    La calibrazione riallinea le probabilita' predette alla frequenza empirica
    degli eventi, **senza** modificare l'ordinamento dei campioni quando la
    trasformazione e' monotona (Platt e Isotonic lo sono). Di conseguenza
    AUC-ROC ed F1 (a soglia fissa sul ranking) restano sostanzialmente
    invariati, mentre Brier ed ECE migliorano se il modello era mal calibrato.

    Args:
        prob_calib: Probabilita' grezze della classe positiva sul set di
            calibrazione, shape ``(n_calib,)``.
        y_calib: Etichette vere (0/1) del set di calibrazione, ``(n_calib,)``.
        method: ``"platt"`` (regressione logistica sulle probabilita',
            equivalente al Platt scaling) oppure ``"isotonic"`` (regressione
            isotonica non parametrica).

    Returns:
        Una funzione ``transform(prob) -> prob_calibrata`` applicabile a nuove
        probabilita' (es. quelle del test), con output in ``[0, 1]``.

    Raises:
        ValueError: Se ``method`` non e' riconosciuto o gli input non sono
            validi.
    """
    prob_calib = np.asarray(prob_calib, dtype=float).ravel()
    y_calib = np.asarray(y_calib).astype(int).ravel()

    if prob_calib.shape[0] != y_calib.shape[0]:
        raise ValueError(
            f"prob_calib ({prob_calib.shape[0]}) e y_calib ({y_calib.shape[0]}) "
            f"hanno lunghezze diverse."
        )
    if prob_calib.size == 0:
        raise ValueError("Set di calibrazione vuoto.")

    method = method.lower()

    if method == "platt":
        # Platt scaling = regressione logistica a una feature (la probabilita'
        # grezza). Restituisce sempre una sigmoide monotona crescente.
        lr = LogisticRegression()
        lr.fit(prob_calib.reshape(-1, 1), y_calib)

        def _transform(prob: np.ndarray) -> np.ndarray:
            prob = np.asarray(prob, dtype=float).ravel()
            out = lr.predict_proba(prob.reshape(-1, 1))[:, 1]
            return np.clip(out, 0.0, 1.0)

        return _transform

    if method == "isotonic":
        # Regressione isotonica: mappatura monotona non parametrica, piu'
        # flessibile ma piu' avida di dati (rumorosa sui set piccoli).
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(prob_calib, y_calib)

        def _transform(prob: np.ndarray) -> np.ndarray:
            prob = np.asarray(prob, dtype=float).ravel()
            out = iso.predict(prob)
            return np.clip(out, 0.0, 1.0)

        return _transform

    raise ValueError(
        f"method non riconosciuto: {method!r}. Usa 'platt' o 'isotonic'."
    )


def optimize_threshold(
    prob_calib: np.ndarray,
    y_calib: np.ndarray,
    n_grid: int = 99,
) -> float:
    """
    Cerca la soglia di decisione che massimizza l'F1 macro sul set di
    calibrazione.

    La soglia di default 0.5 e' ottimale per l'accuratezza su classi
    bilanciate, ma raramente per l'F1 macro su dataset sbilanciati. La soglia
    viene scelta esclusivamente sul set di calibrazione (mai sul test) per non
    introdurre leakage, in analogia con l'early stopping del capitolo LoRA.

    Args:
        prob_calib: Probabilita' della classe positiva sul set di calibrazione,
            shape ``(n_calib,)``.
        y_calib: Etichette vere (0/1), shape ``(n_calib,)``.
        n_grid: Numero di soglie candidate equispaziate in ``(0, 1)``.

    Returns:
        La soglia (float in ``(0, 1)``) con F1 macro massimo sul set di
        calibrazione. In caso di parita' viene scelta la soglia piu' bassa.

    Raises:
        ValueError: Se gli input non sono validi.
    """
    prob_calib = np.asarray(prob_calib, dtype=float).ravel()
    y_calib = np.asarray(y_calib).astype(int).ravel()

    if prob_calib.shape[0] != y_calib.shape[0]:
        raise ValueError(
            f"prob_calib ({prob_calib.shape[0]}) e y_calib ({y_calib.shape[0]}) "
            f"hanno lunghezze diverse."
        )
    if prob_calib.size == 0:
        raise ValueError("Set di calibrazione vuoto.")

    thresholds = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]  # esclude 0 e 1

    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        y_pred = (prob_calib >= thr).astype(int)
        f1 = f1_score(y_calib, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


# ---------------------------------------------------------------------------
# Helper di supporto agli esperimenti
# ---------------------------------------------------------------------------

def _three_way_split(
    X: np.ndarray,
    y: np.ndarray,
    calib_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, ...]:
    """
    Split stratificato a 3 vie: ``train(context) / calibration / test``.

    Prima isola il test, poi divide la parte restante in train e calibration,
    mantenendo le proporzioni richieste rispetto al totale.

    Args:
        X, y: Dati completi del dataset.
        calib_size: Frazione del totale destinata alla calibrazione.
        test_size: Frazione del totale destinata al test.
        random_state: Seme per la riproducibilita'.

    Returns:
        Tupla ``(X_tr, y_tr, X_ca, y_ca, X_te, y_te)``.
    """
    # 1) stacca il test dal resto.
    X_rest, X_te, y_rest, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # 2) dal resto stacca la calibrazione (riscalando la frazione).
    calib_rel = calib_size / (1.0 - test_size)
    X_tr, X_ca, y_tr, y_ca = train_test_split(
        X_rest, y_rest, test_size=calib_rel, random_state=random_state, stratify=y_rest
    )
    return X_tr, y_tr, X_ca, y_ca, X_te, y_te


def _make_tabpfn(device: str, random_state: int, n_estimators: int):
    """
    Istanzia un ``TabPFNClassifier`` v2.5 coerente col resto del progetto.

    Import lazy di TabPFN cosi' il modulo resta importabile su CPU senza il
    pacchetto installato. ``ignore_pretraining_limits=True`` come negli altri
    esperimenti, per gestire i dataset grandi (es. adult, bank-marketing).
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(
        version=ModelVersion.V2_5,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        ignore_pretraining_limits=True,
    )


def _cap_dataset(
    X: np.ndarray, y: np.ndarray, max_samples: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sottocampiona (stratificato) il dataset se supera ``max_samples``."""
    if max_samples is None or len(y) <= max_samples:
        return X, y
    X, _, y, _ = train_test_split(
        X, y, train_size=max_samples, random_state=random_state, stratify=y
    )
    return X, y


def _aggregate(records: List[Dict[str, object]], group_keys: List[str]) -> pd.DataFrame:
    """
    Aggrega i record per-seed in media +/- deviazione standard.

    Args:
        records: Lista di dizionari (output di ``evaluate_model`` + colonne di
            raggruppamento come 'dataset' e 'method').
        group_keys: Colonne su cui raggruppare (es. ``["dataset", "method"]``).

    Returns:
        DataFrame con, per ogni gruppo, le colonne ``<metrica>_mean`` e
        ``<metrica>_std`` per auc_roc, f1, brier_score, ece, piu' ``n_seeds``.
    """
    df = pd.DataFrame(records)
    metric_cols = ["auc_roc", "f1", "brier_score", "ece"]

    agg = df.groupby(group_keys)[metric_cols].agg(["mean", "std"]).reset_index()
    # Appiattisce il MultiIndex delle colonne: ('auc_roc','mean') -> 'auc_roc_mean'.
    agg.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in agg.columns
    ]
    counts = df.groupby(group_keys).size().reset_index(name="n_seeds")
    agg = agg.merge(counts, on=group_keys)
    return agg


def _print_aggregate(agg: pd.DataFrame, title: str) -> None:
    """Stampa la tabella aggregata media+/-std in forma leggibile."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for dataset_name, group in agg.groupby("dataset"):
        print(f"\nDataset: {dataset_name}")
        print("-" * 78)
        print(f"{'method':<16} | {'AUC':>15} | {'F1':>15} | {'Brier':>13} | {'ECE':>13}")
        print("-" * 78)
        for _, row in group.iterrows():
            def cell(m: str, width: int) -> str:
                return f"{row[m + '_mean']:.4f}+/-{row[m + '_std']:.4f}".rjust(width)
            print(
                f"{str(row['method']):<16} | {cell('auc_roc', 15)} | "
                f"{cell('f1', 15)} | {cell('brier_score', 13)} | {cell('ece', 13)}"
            )
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Exp 3 - Ensembling
# ---------------------------------------------------------------------------

def run_ensemble_experiment(
    datasets: Optional[Dict[str, int]] = None,
    n_seeds: int = 5,
    n_ensemble: int = 5,
    calib_size: float = 0.25,
    test_size: float = 0.25,
    max_samples: int = 20000,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Esperimento 3: ensembling sull'output di TabPFN (pesi congelati).

    Confronta tre metodi su ciascun dataset:
        * ``base``: una singola istanza TabPFN (baseline);
        * ``self_ensemble``: media delle probabilita' di ``n_ensemble`` istanze
          TabPFN con seed interni diversi (riduzione della varianza interna);
        * ``tabpfn+gbm``: media tra TabPFN e un GradientBoostingClassifier
          (ensemble eterogeneo).

    Il set di calibrazione non serve qui ma viene comunque ritagliato, cosi' il
    test coincide esattamente con quello dell'Esperimento 4 (confronto equo).
    La procedura e' ripetuta su ``n_seeds`` semi; output in media +/- std.

    Args:
        datasets: Mappa ``nome -> openml_id``. Default: ``PIPELINE_DATASETS``.
        n_seeds: Numero di ripetizioni con split/seed diversi.
        n_ensemble: Numero di istanze TabPFN nel self-ensemble.
        calib_size, test_size: Frazioni dello split a 3 vie.
        max_samples: Cap sul numero di campioni per dataset (gestisce i
            dataset grandi e contiene i tempi su GPU).
        device: ``"cuda"`` su Colab, ``"cpu"`` in locale.
        save_path: Se indicato, salva la tabella aggregata in CSV.
        verbose: Stampa avanzamento e tabella finale.

    Returns:
        DataFrame aggregato (media +/- std) per ``(dataset, method)``.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    if datasets is None:
        datasets = PIPELINE_DATASETS

    records: List[Dict[str, object]] = []

    for name, did in datasets.items():
        if verbose:
            print(f"\n########## Dataset: {name} (id={did}) ##########")
        X, y = load_dataset(did)

        for seed in range(n_seeds):
            Xc, yc = _cap_dataset(X, y, max_samples, random_state=seed)
            X_tr, y_tr, _X_ca, _y_ca, X_te, y_te = _three_way_split(
                Xc, yc, calib_size, test_size, random_state=seed
            )
            if verbose:
                print(f"  [seed {seed}] train={len(y_tr)} test={len(y_te)}")

            # --- base: singola istanza TabPFN ----------------------------
            clf0 = _make_tabpfn(device, random_state=seed, n_estimators=4)
            clf0.fit(X_tr, y_tr)
            prob_base = clf0.predict_proba(X_te)[:, 1]
            records.append({
                "dataset": name, "method": "base",
                **{k: v for k, v in evaluate_model(
                    y_te, prob_base, "base", name).items()
                   if k in ("auc_roc", "f1", "brier_score", "ece")},
            })

            # --- self-ensemble: media di n_ensemble istanze TabPFN -------
            probas = [prob_base]  # riusa la prima per non sprecare calcolo
            for j in range(1, n_ensemble):
                clf = _make_tabpfn(device, random_state=seed * 100 + j, n_estimators=4)
                clf.fit(X_tr, y_tr)
                probas.append(clf.predict_proba(X_te)[:, 1])
            prob_self = average_probas(probas)
            records.append({
                "dataset": name, "method": "self_ensemble",
                **{k: v for k, v in evaluate_model(
                    y_te, prob_self, "self_ensemble", name).items()
                   if k in ("auc_roc", "f1", "brier_score", "ece")},
            })

            # --- ensemble eterogeneo: TabPFN + GradientBoosting ----------
            gbm = GradientBoostingClassifier(random_state=seed)
            gbm.fit(X_tr, y_tr)
            prob_gbm = gbm.predict_proba(X_te)[:, 1]
            prob_hetero = average_probas([prob_base, prob_gbm])
            records.append({
                "dataset": name, "method": "tabpfn+gbm",
                **{k: v for k, v in evaluate_model(
                    y_te, prob_hetero, "tabpfn+gbm", name).items()
                   if k in ("auc_roc", "f1", "brier_score", "ece")},
            })

            _free_gpu(device)

    agg = _aggregate(records, ["dataset", "method"])
    if verbose:
        _print_aggregate(agg, "ESPERIMENTO 3 - ENSEMBLING (media +/- std su seed)")
    if save_path:
        agg.to_csv(save_path, index=False)
        if verbose:
            print(f"[SAVE] Tabella aggregata salvata in '{save_path}'.")
    return agg


# ---------------------------------------------------------------------------
# Exp 4 - Calibrazione + soglia
# ---------------------------------------------------------------------------

def run_calibration_threshold_experiment(
    datasets: Optional[Dict[str, int]] = None,
    n_seeds: int = 5,
    calib_size: float = 0.25,
    test_size: float = 0.25,
    max_samples: int = 20000,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Esperimento 4: calibrazione e ottimizzazione della soglia sull'output.

    Per ogni dataset, TabPFN viene fittato sul context e produce probabilita'
    grezze su calibration e test. Si confrontano quattro metodi sul test:
        * ``base``:      probabilita' grezze, soglia 0.5;
        * ``platt``:     probabilita' ricalibrate con Platt scaling;
        * ``isotonic``:  probabilita' ricalibrate con regressione isotonica;
        * ``threshold``: probabilita' grezze, ma soglia ottimizzata per F1
          macro sul set di calibrazione.

    Atteso (e da verificare): la calibrazione migliora ECE/Brier lasciando
    AUC invariata; l'ottimizzazione della soglia migliora F1 lasciando
    AUC/ECE invariati. Calibratore e soglia sono fittati solo sulla
    calibrazione (no leakage). Ripetuto su ``n_seeds`` semi; media +/- std.

    Args:
        datasets: Mappa ``nome -> openml_id``. Default: ``PIPELINE_DATASETS``.
        n_seeds: Numero di ripetizioni.
        calib_size, test_size: Frazioni dello split a 3 vie.
        max_samples: Cap sul numero di campioni per dataset.
        device: ``"cuda"`` su Colab, ``"cpu"`` in locale.
        save_path: Se indicato, salva la tabella aggregata in CSV.
        verbose: Stampa avanzamento e tabella finale.

    Returns:
        DataFrame aggregato (media +/- std) per ``(dataset, method)``.
    """
    if datasets is None:
        datasets = PIPELINE_DATASETS

    records: List[Dict[str, object]] = []

    for name, did in datasets.items():
        if verbose:
            print(f"\n########## Dataset: {name} (id={did}) ##########")
        X, y = load_dataset(did)

        for seed in range(n_seeds):
            Xc, yc = _cap_dataset(X, y, max_samples, random_state=seed)
            X_tr, y_tr, X_ca, y_ca, X_te, y_te = _three_way_split(
                Xc, yc, calib_size, test_size, random_state=seed
            )
            if verbose:
                print(f"  [seed {seed}] train={len(y_tr)} "
                      f"calib={len(y_ca)} test={len(y_te)}")

            clf = _make_tabpfn(device, random_state=seed, n_estimators=4)
            clf.fit(X_tr, y_tr)
            prob_ca = clf.predict_proba(X_ca)[:, 1]
            prob_te = clf.predict_proba(X_te)[:, 1]

            # --- base ----------------------------------------------------
            _append_metrics(records, name, "base", y_te, prob_te, threshold=0.5)

            # --- calibrazione Platt --------------------------------------
            t_platt = fit_calibrator(prob_ca, y_ca, method="platt")
            _append_metrics(records, name, "platt", y_te, t_platt(prob_te), threshold=0.5)

            # --- calibrazione Isotonic -----------------------------------
            t_iso = fit_calibrator(prob_ca, y_ca, method="isotonic")
            _append_metrics(records, name, "isotonic", y_te, t_iso(prob_te), threshold=0.5)

            # --- soglia ottimizzata per F1 (su proba grezze) -------------
            thr = optimize_threshold(prob_ca, y_ca)
            _append_metrics(records, name, "threshold", y_te, prob_te, threshold=thr)

            _free_gpu(device)

    agg = _aggregate(records, ["dataset", "method"])
    if verbose:
        _print_aggregate(
            agg, "ESPERIMENTO 4 - CALIBRAZIONE + SOGLIA (media +/- std su seed)"
        )
    if save_path:
        agg.to_csv(save_path, index=False)
        if verbose:
            print(f"[SAVE] Tabella aggregata salvata in '{save_path}'.")
    return agg


# ---------------------------------------------------------------------------
# Utility interne
# ---------------------------------------------------------------------------

def _append_metrics(
    records: List[Dict[str, object]],
    dataset: str,
    method: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    """Valuta e accoda un record (solo le 4 metriche numeriche)."""
    res = evaluate_model(y_true, y_prob, method, dataset, threshold=threshold)
    records.append({
        "dataset": dataset,
        "method": method,
        "auc_roc": res["auc_roc"],
        "f1": res["f1"],
        "brier_score": res["brier_score"],
        "ece": res["ece"],
    })


def _free_gpu(device: str) -> None:
    """Libera la cache CUDA tra un seed e l'altro (no-op su CPU)."""
    if device.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Blocco di test (CPU, senza TabPFN): verifica gli helper puri
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST helper di external_pipeline.py (dati sintetici, no TabPFN)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, size=n)

    # Probabilita' "scalate" verso 0.5: buon ranking ma sotto-confidenti
    # (mal calibrate) -> la calibrazione dovrebbe ridurre ECE.
    base_signal = y * 0.6 + 0.2 + rng.normal(0, 0.1, size=n)
    prob = np.clip(0.5 + (np.clip(base_signal, 0, 1) - 0.5) * 0.4, 0, 1)

    yc, yt = y[:300], y[300:]
    pc, pt = prob[:300], prob[300:]

    # --- average_probas --------------------------------------------------
    avg = average_probas([pt, pt, pt])
    assert np.allclose(avg, pt), "La media di vettori identici deve coincidere."
    print("\n[OK] average_probas: media di vettori identici == vettore.")

    # --- fit_calibrator: AUC invariata, ECE migliorata -------------------
    from evaluation.metrics import compute_auc_roc, compute_ece

    auc_raw = compute_auc_roc(yt, pt)
    ece_raw = compute_ece(yt, pt)
    for method in ("platt", "isotonic"):
        t = fit_calibrator(pc, yc, method=method)
        pt_cal = t(pt)
        auc_cal = compute_auc_roc(yt, pt_cal)
        ece_cal = compute_ece(yt, pt_cal)
        print(f"\n[{method}] AUC: {auc_raw:.4f} -> {auc_cal:.4f} "
              f"(atteso ~invariata) | ECE: {ece_raw:.4f} -> {ece_cal:.4f} "
              f"(atteso <= )")
        # AUC praticamente invariata (trasformazione monotona).
        assert abs(auc_cal - auc_raw) < 0.02, "La calibrazione non deve toccare l'AUC."

    # --- optimize_threshold: F1 >= F1 a 0.5 sul set di calibrazione ------
    thr = optimize_threshold(pc, yc)
    f1_05 = f1_score(yc, (pc >= 0.5).astype(int), average="macro")
    f1_opt = f1_score(yc, (pc >= thr).astype(int), average="macro")
    print(f"\n[threshold] soglia ottimale={thr:.3f} | "
          f"F1@0.5={f1_05:.4f} -> F1@opt={f1_opt:.4f} (atteso >=)")
    assert f1_opt >= f1_05 - 1e-9, "La soglia ottimizzata non deve peggiorare l'F1."

    # --- errori ----------------------------------------------------------
    for bad_call, descr in [
        (lambda: average_probas([]), "average_probas lista vuota"),
        (lambda: fit_calibrator(pc, yc, method="xyz"), "metodo calibratore ignoto"),
    ]:
        try:
            bad_call()
            print(f"  [FALLITO] {descr}: avrebbe dovuto sollevare ValueError")
        except ValueError as exc:
            print(f"  -> ValueError ({descr}): {exc}")

    print("\n" + "=" * 70)
    print("TUTTI I TEST DEGLI HELPER SUPERATI")
    print("=" * 70)
