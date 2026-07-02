"""
context_experiments.py
======================

Esperimenti di **context engineering** su TabPFN-2.5. Pesi congelati: si agisce
solo sul contesto in-context (l'input del modello), non sui pesi (LoRA) ne'
sull'output (pipeline esterna).

    Esperimento A - Instance Selection
        Contesto completo vs contesto ridotto ai K esempi piu' rappresentativi
        (KMeans per-classe, vedi ``select_by_clustering``). Domanda: quanta
        performance si conserva comprimendo il contesto? TabPFN paga il costo
        in-context in modo quadratico col numero di esempi, quindi un contesto
        piccolo ma ben scelto e' interessante anche in termini di efficienza.

    Esperimento B - Context Balancing
        Su dataset sbilanciati, confronta quattro varianti del contesto:
            * ``base``:            contesto originale, soglia 0.5;
            * ``smote``:           contesto ribilanciato con SMOTE, soglia 0.5;
            * ``undersample``:     contesto ridotto con undersampling, soglia 0.5;
            * ``smote_threshold``: SMOTE + soglia ottimizzata sul set di
              calibrazione (stesso metodo della pipeline esterna).
        L'ultima variante e' il ponte con la pipeline esterna: bilanciare il
        contesto e ottimizzare la soglia sono complementari o ridondanti?

Metodologia (identica al resto del progetto, per confrontabilita' diretta):
    split stratificato a 3 vie ``train(context)/calibration/test``, ``n_seeds``
    semi, media +/- deviazione standard sulle 4 metriche (AUC, F1, Brier, ECE).
    Gli helper di split/istanza-TabPFN/aggregazione/CSV sono riusati **tali e
    quali** dalla pipeline esterna, cosi' i test set coincidono e i CSV prodotti
    hanno lo stesso schema di ``pipeline_financial_*.csv``.

    Nota anti-leakage: qualsiasi ribilanciamento (SMOTE/undersampling) e
    selezione (clustering) tocca **solo** il contesto (train). Calibrazione e
    test restano sempre gli originali. La soglia e' cercata solo sul set di
    calibrazione.

Dipendenze:
    pip install scikit-learn numpy pandas imbalanced-learn   (+ tabpfn per gli
    esperimenti; import lazy, il modulo resta importabile su CPU senza TabPFN)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Rende importabili i package fratelli (utils, evaluation, pipeline) anche
# quando il modulo e' eseguito direttamente o da una sessione Colab.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.data_loader import PIPELINE_DATASETS, load_dataset  # noqa: E402
# Riuso degli helper della pipeline esterna: stesso split, stessa istanza
# TabPFN, stessa aggregazione/stampa/salvataggio -> risultati confrontabili.
from pipeline.external_pipeline import (  # noqa: E402
    _aggregate,
    _append_metrics,
    _cap_dataset,
    _free_gpu,
    _make_tabpfn,
    _print_aggregate,
    _three_way_split,
    optimize_threshold,
)
from context.context_engineering import (  # noqa: E402
    balance_context_smote,
    balance_context_undersample,
    select_by_clustering,
)


# Dataset sbilanciati per l'Esperimento B: bilanciare ha senso solo dove c'e'
# sbilanciamento da correggere. I due Polish (3.9% e 6.9% di positivi) sono gli
# unici realmente sbilanciati tra i 4 finanziari; australian (~56/44) e
# credit_g (70/30) non trarrebbero beneficio dal ribilanciamento.
IMBALANCED_DATASETS: Dict[str, int] = {
    "polish_bankruptcy_1": 42880,  # 96/4
    "polish_bankruptcy_5": 42987,  # 93/7
}


# ---------------------------------------------------------------------------
# Esperimento A - Instance Selection
# ---------------------------------------------------------------------------

def run_instance_selection_experiment(
    datasets: Optional[Dict[str, int]] = None,
    k_values: Tuple[int, ...] = (100, 200, 500),
    n_seeds: int = 5,
    calib_size: float = 0.25,
    test_size: float = 0.25,
    max_samples: int = 20000,
    n_estimators: int = 4,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Esperimento A: TabPFN con contesto completo vs contesto ridotto a K esempi.

    Per ogni dataset e seed si esegue lo split a 3 vie (il set di calibrazione
    viene ritagliato ma non serve qui: garantisce solo che il test coincida con
    quello degli altri esperimenti). Poi:
        * ``base``:   TabPFN sul contesto completo (train);
        * ``k{K}``:   TabPFN sul sottoinsieme di K esempi scelti via
          ``select_by_clustering`` (KMeans per-classe).

    Per i dataset piccoli, se ``K`` e' >= alla dimensione del contesto la
    selezione coinciderebbe col contesto pieno: quel valore di K viene
    **saltato e segnalato** nel log (es. australian_credit ha ~345 righe di
    contesto, quindi k=500 viene saltato).

    Args:
        datasets: Mappa ``nome -> openml_id``. Default: i 4 dataset finanziari
            di ``PIPELINE_DATASETS``.
        k_values: Dimensioni del contesto ridotto da testare.
        n_seeds: Numero di ripetizioni con split/seed diversi.
        calib_size, test_size: Frazioni dello split a 3 vie.
        max_samples: Cap sul numero di campioni per dataset.
        n_estimators: ``n_estimators`` interni di TabPFN.
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
            X_tr, y_tr, _X_ca, _y_ca, X_te, y_te = _three_way_split(
                Xc, yc, calib_size, test_size, random_state=seed
            )
            n_ctx = len(y_tr)
            if verbose:
                print(f"  [seed {seed}] context={n_ctx} test={len(y_te)}")

            # --- base: contesto completo --------------------------------
            clf = _make_tabpfn(device, random_state=seed, n_estimators=n_estimators)
            clf.fit(X_tr, y_tr)
            prob_te = clf.predict_proba(X_te)[:, 1]
            _append_metrics(records, name, "base", y_te, prob_te, threshold=0.5)

            # --- k{K}: contesto ridotto via clustering ------------------
            for k in k_values:
                if k >= n_ctx:
                    if verbose:
                        print(
                            f"    [seed {seed}] k={k} saltato: "
                            f"contesto ({n_ctx}) <= k, coinciderebbe col base."
                        )
                    continue

                idx = select_by_clustering(X_tr, y_tr, k=k, random_state=seed)
                if verbose:
                    print(f"    [seed {seed}] k={k}: selezionati {len(idx)} esempi")

                clf_k = _make_tabpfn(
                    device, random_state=seed, n_estimators=n_estimators
                )
                clf_k.fit(X_tr[idx], y_tr[idx])
                prob_k = clf_k.predict_proba(X_te)[:, 1]
                _append_metrics(records, name, f"k{k}", y_te, prob_k, threshold=0.5)

            _free_gpu(device)

    agg = _aggregate(records, ["dataset", "method"])
    if verbose:
        _print_aggregate(
            agg, "ESPERIMENTO A - INSTANCE SELECTION (media +/- std su seed)"
        )
    if save_path:
        agg.to_csv(save_path, index=False)
        if verbose:
            print(f"[SAVE] Tabella aggregata salvata in '{save_path}'.")
    return agg


# ---------------------------------------------------------------------------
# Esperimento B - Context Balancing
# ---------------------------------------------------------------------------

def run_context_balancing_experiment(
    datasets: Optional[Dict[str, int]] = None,
    n_seeds: int = 5,
    calib_size: float = 0.25,
    test_size: float = 0.25,
    max_samples: int = 20000,
    n_estimators: int = 4,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Esperimento B: bilanciamento del contesto su dataset sbilanciati.

    Per ogni dataset e seed (split a 3 vie), TabPFN viene fittato su quattro
    versioni del contesto e valutato sullo **stesso** test:
        * ``base``:            contesto originale, soglia 0.5;
        * ``smote``:           contesto bilanciato con SMOTE, soglia 0.5;
        * ``undersample``:     contesto bilanciato con undersampling, soglia 0.5;
        * ``smote_threshold``: contesto SMOTE, soglia ottimizzata per F1 macro
          sul set di calibrazione (stesso ``optimize_threshold`` della pipeline).

    La variante ``smote_threshold`` combina i due interventi per capire se sono
    complementari (guadagni che si sommano) o ridondanti (agiscono sulla stessa
    leva). Le probabilita' di calibrazione per la soglia sono calcolate sul set
    di calibrazione **originale** (non ribilanciato): la soglia va scelta sulla
    distribuzione reale, non su una sintetica.

    Args:
        datasets: Mappa ``nome -> openml_id``. Default: ``IMBALANCED_DATASETS``
            (i due Polish bankruptcy, gli unici realmente sbilanciati).
        n_seeds: Numero di ripetizioni.
        calib_size, test_size: Frazioni dello split a 3 vie.
        max_samples: Cap sul numero di campioni per dataset.
        n_estimators: ``n_estimators`` interni di TabPFN.
        device: ``"cuda"`` su Colab, ``"cpu"`` in locale.
        save_path: Se indicato, salva la tabella aggregata in CSV.
        verbose: Stampa avanzamento e tabella finale.

    Returns:
        DataFrame aggregato (media +/- std) per ``(dataset, method)``.
    """
    if datasets is None:
        datasets = IMBALANCED_DATASETS

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
                print(f"  [seed {seed}] context={len(y_tr)} "
                      f"calib={len(y_ca)} test={len(y_te)}")

            # --- base: contesto originale, soglia 0.5 -------------------
            clf = _make_tabpfn(device, random_state=seed, n_estimators=n_estimators)
            clf.fit(X_tr, y_tr)
            prob_te = clf.predict_proba(X_te)[:, 1]
            _append_metrics(records, name, "base", y_te, prob_te, threshold=0.5)

            # --- smote: contesto SMOTE, soglia 0.5 ----------------------
            # Fittiamo una sola volta sul contesto SMOTE e riusiamo le
            # probabilita' sia per 'smote' (0.5) sia per 'smote_threshold'.
            Xs, ys = balance_context_smote(X_tr, y_tr, random_state=seed)
            clf_s = _make_tabpfn(device, random_state=seed, n_estimators=n_estimators)
            clf_s.fit(Xs, ys)
            prob_s_ca = clf_s.predict_proba(X_ca)[:, 1]  # calib ORIGINALE
            prob_s_te = clf_s.predict_proba(X_te)[:, 1]
            _append_metrics(records, name, "smote", y_te, prob_s_te, threshold=0.5)

            # --- undersample: contesto ridotto/bilanciato, soglia 0.5 ---
            Xu, yu = balance_context_undersample(X_tr, y_tr, random_state=seed)
            clf_u = _make_tabpfn(device, random_state=seed, n_estimators=n_estimators)
            clf_u.fit(Xu, yu)
            prob_u_te = clf_u.predict_proba(X_te)[:, 1]
            _append_metrics(records, name, "undersample", y_te, prob_u_te, threshold=0.5)

            # --- smote_threshold: SMOTE + soglia ottimizzata su calib ---
            thr = optimize_threshold(prob_s_ca, y_ca)
            _append_metrics(
                records, name, "smote_threshold", y_te, prob_s_te, threshold=thr
            )

            _free_gpu(device)

    agg = _aggregate(records, ["dataset", "method"])
    if verbose:
        _print_aggregate(
            agg, "ESPERIMENTO B - CONTEXT BALANCING (media +/- std su seed)"
        )
    if save_path:
        agg.to_csv(save_path, index=False)
        if verbose:
            print(f"[SAVE] Tabella aggregata salvata in '{save_path}'.")
    return agg


# ---------------------------------------------------------------------------
# Esecuzione diretta: lancia entrambi gli esperimenti (richiede TabPFN)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Esperimenti di context engineering su TabPFN-2.5."
    )
    parser.add_argument(
        "--device", default="cuda", help="'cuda' (Colab) o 'cpu' (locale)."
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--results-dir",
        default=os.path.join(_PROJECT_ROOT, "results"),
        help="Cartella dove salvare i CSV.",
    )
    parser.add_argument(
        "--only",
        choices=["A", "B", "both"],
        default="both",
        help="Quale esperimento eseguire.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.only in ("A", "both"):
        run_instance_selection_experiment(
            n_seeds=args.n_seeds,
            device=args.device,
            save_path=os.path.join(args.results_dir, "context_instance_selection.csv"),
        )

    if args.only in ("B", "both"):
        run_context_balancing_experiment(
            n_seeds=args.n_seeds,
            device=args.device,
            save_path=os.path.join(args.results_dir, "context_balancing.csv"),
        )

    print("\n[FATTO] Esperimenti di context engineering completati.")
