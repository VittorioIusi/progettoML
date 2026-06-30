"""
data_loader.py
==============

Modulo per il caricamento e la pre-elaborazione di dataset di classificazione
binaria provenienti da OpenML, pensato per un flusso di lavoro basato su TabPFN.

Funzionalita' principali:
    * Download dei dataset da OpenML tramite ``dataset_id``.
    * Encoding delle feature categoriche (OrdinalEncoder).
    * Imputazione dei valori mancanti (SimpleImputer, strategia ``median``).
    * Encoding del target in formato binario 0/1 (LabelEncoder).
    * Caching locale dei dataset gia' scaricati in ``data/raw/``.
    * Funzioni di alto livello per ottenere split train/val (fine-tuning) e
      train/test (valutazione), oltre a utility per riepiloghi tabellari.

Dipendenze:
    pip install openml scikit-learn numpy pandas
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import openml
except ImportError as exc:  # pragma: no cover - dipendenza esterna
    raise ImportError(
        "Il pacchetto 'openml' non e' installato. "
        "Installalo con: pip install openml"
    ) from exc

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Dataset di FINE-TUNING (dominio FINANZIARIO/creditizio). Sostituiscono i
# precedenti dataset medici, che erano troppo piccoli (155-768 righe) per
# fornire abbastanza passi di gradiente. Tutti binari, di dimensione adeguata e
# sotto il limite di ~50.000 righe di TabPFN-2.5. ID verificati su OpenML.
FINETUNE_DATASETS: Dict[str, int] = {
    "bank_marketing": 1461,        # ~45.211 righe, 16 feat, 88/12
    "default_credit": 42477,       # ~30.000 righe, 23 feat, 78/22
    "polish_bankruptcy_2": 42984,  # ~10.173 righe, 64 feat, 96/4
    "polish_bankruptcy_3": 42985,  # ~10.503 righe, 64 feat, 95/5
    "polish_bankruptcy_4": 42986,  # ~9.792 righe,  64 feat, 95/5
}

# Dataset di VALUTAZIONE (stesso dominio FINANZIARIO -> valutazione IN-DOMAIN).
# Mai usati in training. Il cambio rispetto alla configurazione cross-domain
# (medico -> non medico) serve a rispondere correttamente alla domanda "il LoRA
# specializza TabPFN su un settore specifico?". ID verificati su OpenML.
#
# NOTA: gli ID dei Polish bankruptcy forniti inizialmente (40474-40478) erano
# ERRATI: puntavano a dataset 'thyroid' a 5 classi. Gli ID corretti dei
# 'polish-bankruptcy-Nyear' sono 42880/42984/42985/42986/42987.
EVALUATION_DATASETS: Dict[str, int] = {
    "polish_bankruptcy_1": 42880,  # ~7.027 righe, 64 feat, 96/4
    "polish_bankruptcy_5": 42987,  # ~5.910 righe, 64 feat, 93/7
    "australian_credit": 40981,    # ~690 righe,   14 feat, 56/44
    "credit_g": 31,                # ~1.000 righe, 20 feat, 70/30 (ora in-domain)
}

# Dataset usati per gli esperimenti della pipeline esterna (ensembling,
# calibrazione, ottimizzazione della soglia). Riprende i 4 dataset di
# valutazione del capitolo LoRA (continuita' e confronto diretto) e aggiunge
# due dataset binari di grandi dimensioni, in modo che le conclusioni su
# calibrazione/soglia poggino su campioni sufficientemente grandi da rendere
# stabili metriche come l'ECE.
PIPELINE_DATASETS: Dict[str, int] = {
    "blood_transfusion": 1464,   # ~748 righe   (piccolo)
    "credit_g": 31,              # ~1.000 righe (piccolo)
    "thyroid": 1000,             # ~3.772 righe (medio, AUC saturo)
    "adult": 1590,               # ~48.842 righe (grande)
    "bank_marketing": 1461,      # ~45.211 righe (grande)
    "magic_telescope": 1120,     # ~19.020 righe (grande)
}

# Cartella usata per il caching dei dataset gia' scaricati ed elaborati.
# Il percorso e' calcolato relativamente alla root del progetto (due livelli
# sopra questo file: utils/ -> tabpfn_project/ -> root del progetto).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")


# ---------------------------------------------------------------------------
# Funzioni di supporto interne
# ---------------------------------------------------------------------------

def _ensure_cache_dir() -> None:
    """Crea la cartella di cache (``data/raw/``) se non esiste gia'."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(dataset_id: int) -> str:
    """
    Restituisce il percorso del file di cache associato a un ``dataset_id``.

    Args:
        dataset_id: Identificativo numerico del dataset su OpenML.

    Returns:
        Percorso assoluto del file ``.pkl`` di cache.
    """
    return os.path.join(CACHE_DIR, f"dataset_{dataset_id}.pkl")


def _print_dataset_verification(name: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Stampa una riga di verifica con nome, dimensioni e distribuzione del target.

    Serve a controllare ``a vista`` che ogni dataset caricato corrisponda a
    quanto atteso (numero di righe/feature plausibile, target binario con
    proporzioni sensate). E' un presidio contro gli ID OpenML errati, problema
    gia' occorso nel progetto.

    Formato: ``Dataset <nome>: <righe> righe, <feature> feature,
    target distribution: <c0>% / <c1>%``.
    """
    n = len(y)
    if n == 0:
        print(f"[VERIFICA] Dataset {name}: 0 righe (vuoto!)")
        return
    counts = np.bincount(y.astype(int))
    dist = " / ".join(f"{c / n * 100:.1f}%" for c in counts)
    flag = "" if len(counts) == 2 else f"  [ATTENZIONE: {len(counts)} classi]"
    print(
        f"[VERIFICA] Dataset {name}: {X.shape[0]} righe, {X.shape[1]} feature, "
        f"target distribution: {dist}{flag}"
    )


# ---------------------------------------------------------------------------
# 1. load_dataset
# ---------------------------------------------------------------------------

def load_dataset(dataset_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scarica ed elabora un dataset di classificazione binaria da OpenML.

    La funzione esegue, nell'ordine:
        1. Lettura dalla cache locale (``data/raw/``) se disponibile, altrimenti
           download da OpenML.
        2. Encoding ordinale delle feature categoriche.
        3. Imputazione dei valori mancanti con strategia ``median``.
        4. Encoding del target in formato binario 0/1.

    Args:
        dataset_id: Identificativo numerico del dataset su OpenML.

    Returns:
        Tupla ``(X, y)`` dove:
            * ``X`` e' un ``np.ndarray`` di tipo ``float32`` con shape
              ``(n_samples, n_features)``.
            * ``y`` e' un ``np.ndarray`` di interi (0/1) con shape
              ``(n_samples,)``.

    Raises:
        RuntimeError: Se il download o l'elaborazione del dataset falliscono.
    """
    _ensure_cache_dir()
    cache_file = _cache_path(dataset_id)

    # --- 1. Caching: se gia' scaricato, ricarica da disco -----------------
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as fh:
                cached = pickle.load(fh)
            X, y, name = cached["X"], cached["y"], cached["name"]
            print(
                f"[CACHE] Dataset '{name}' caricato dalla cache | "
                f"campioni: {X.shape[0]} | feature: {X.shape[1]}"
            )
            _print_dataset_verification(name, X, y)
            return X, y
        except (pickle.PickleError, KeyError, EOFError) as exc:
            print(
                f"[WARN] Cache corrotta per dataset_id={dataset_id} "
                f"({exc}). Riscarico da OpenML."
            )

    # --- 2. Download da OpenML --------------------------------------------
    # Nota: l'indicatore categorico viene memorizzato come mappa
    # nome_colonna -> bool, cosi' eventuali rimozioni di colonne (target, id)
    # non disallineano gli indici.
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        name = dataset.name
        target_attr = dataset.default_target_attribute

        if target_attr is not None:
            X_df, y_series, categorical_indicator, attr_names = dataset.get_data(
                target=target_attr,
                dataset_format="dataframe",
            )
            cat_map = dict(zip(attr_names, categorical_indicator))
        else:
            # Alcuni dataset su OpenML non hanno una colonna target dichiarata.
            # In questi casi scarichiamo l'intero dataframe e usiamo per
            # convenzione l'ultima colonna come target.
            full_df, _, categorical_indicator, attr_names = dataset.get_data(
                dataset_format="dataframe",
            )
            if full_df is None or full_df.shape[1] < 2:
                raise RuntimeError(
                    f"Il dataset id={dataset_id} non contiene colonne "
                    f"sufficienti per separare feature e target."
                )
            cat_map = dict(zip(attr_names, categorical_indicator))
            target_col = full_df.columns[-1]
            y_series = full_df[target_col]
            X_df = full_df.drop(columns=[target_col])
            print(
                f"[INFO] Dataset '{name}' privo di target dichiarato: "
                f"uso l'ultima colonna ('{target_col}') come target."
            )
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - vogliamo un messaggio chiaro
        raise RuntimeError(
            f"Impossibile scaricare il dataset con id={dataset_id} da OpenML: "
            f"{exc}"
        ) from exc

    if X_df is None or y_series is None:
        raise RuntimeError(
            f"Il dataset id={dataset_id} non contiene feature o target validi."
        )

    # Rimuove eventuali colonne identificative (es. 'id', 'index'): sono
    # indici progressivi che non hanno valore predittivo e, se correlati
    # all'ordinamento delle classi, introdurrebbero data leakage.
    id_like = [
        c for c in X_df.columns
        if str(c).strip().lower() in ("id", "index", "unnamed: 0")
    ]
    if id_like:
        X_df = X_df.drop(columns=id_like)
        print(f"[INFO] Rimosse colonne identificative: {id_like}")

    try:
        # --- 3. Encoding delle feature categoriche ------------------------
        X_df = X_df.copy()

        # Distinzione numerico/categorico robusta rispetto al backend dei
        # tipi (in pandas 3.0 le stringhe hanno dtype 'str' Arrow, non
        # 'object'). Una colonna e' trattata come numerica se NON e' marcata
        # categorica da OpenML e i suoi valori non nulli sono convertibili a
        # numero per almeno il 50%; altrimenti e' categorica. Cosi' colonne
        # come 'pcv' ("44","38") restano numeriche e 'rbc' ("normal") diventa
        # categorica, indipendentemente dal dtype originale.
        categorical_cols = []
        for col in X_df.columns:
            numeric_try = pd.to_numeric(X_df[col], errors="coerce")
            orig_notna = X_df[col].notna()
            convertible = (
                orig_notna.sum() > 0
                and float(numeric_try[orig_notna].notna().mean()) >= 0.5
            )
            if not cat_map.get(col, False) and convertible:
                X_df[col] = numeric_try
            else:
                categorical_cols.append(col)

        if categorical_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            )
            # Uniforma i tipi a stringa (rimuovendo spazi superflui) prima
            # dell'encoding ordinale.
            cat_data = X_df[categorical_cols].astype(str).apply(
                lambda s: s.str.strip()
            )
            X_df[categorical_cols] = encoder.fit_transform(cat_data)

        # Converte tutto in numerico (eventuali residui -> NaN).
        X_df = X_df.apply(pd.to_numeric, errors="coerce")

        # --- 4. Imputazione dei valori mancanti ---------------------------
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X_df.values)
        X = X.astype(np.float32)

        # --- 5. Encoding del target in binario 0/1 ------------------------
        # Si rimuovono spazi/tabulazioni accidentali (es. 'ckd\t' vs 'ckd')
        # che altrimenti verrebbero contati come classi distinte.
        y_clean = y_series.astype(str).str.strip()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_clean)
        y = y.astype(int)

        n_classes = len(np.unique(y))
        if n_classes != 2:
            print(
                f"[WARN] Il dataset '{name}' ha {n_classes} classi "
                f"(atteso 2 per classificazione binaria)."
            )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Errore durante l'elaborazione del dataset '{name}' "
            f"(id={dataset_id}): {exc}"
        ) from exc

    # --- 6. Stampa informazioni e salva in cache --------------------------
    print(
        f"[OPENML] Dataset '{name}' scaricato | "
        f"campioni: {X.shape[0]} | feature: {X.shape[1]}"
    )
    _print_dataset_verification(name, X, y)

    try:
        with open(cache_file, "wb") as fh:
            pickle.dump({"X": X, "y": y, "name": name}, fh)
    except OSError as exc:
        print(f"[WARN] Impossibile salvare la cache per '{name}': {exc}")

    return X, y


# ---------------------------------------------------------------------------
# 2. get_finetune_data
# ---------------------------------------------------------------------------

def get_finetune_data(
    dataset_name: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica un dataset di fine-tuning e lo divide in train/validation.

    Args:
        dataset_name: Nome del dataset, deve essere una chiave di
            ``FINETUNE_DATASETS``.
        val_size: Frazione di dati da destinare al validation set
            (default ``0.2``).
        random_state: Seed per la riproducibilita' dello split (default ``42``).

    Returns:
        Tupla ``(X_train, y_train, X_val, y_val)`` di ``np.ndarray``.

    Raises:
        ValueError: Se ``dataset_name`` non e' presente in ``FINETUNE_DATASETS``.
    """
    if dataset_name not in FINETUNE_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' non trovato in FINETUNE_DATASETS. "
            f"Dataset disponibili: {list(FINETUNE_DATASETS.keys())}"
        )

    dataset_id = FINETUNE_DATASETS[dataset_name]
    X, y = load_dataset(dataset_id)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# 3. get_evaluation_data
# ---------------------------------------------------------------------------

def get_evaluation_data(
    dataset_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica un dataset di valutazione e lo divide in train/test.

    Args:
        dataset_name: Nome del dataset, deve essere una chiave di
            ``EVALUATION_DATASETS``.
        test_size: Frazione di dati da destinare al test set (default ``0.2``).
        random_state: Seed per la riproducibilita' dello split (default ``42``).

    Returns:
        Tupla ``(X_train, y_train, X_test, y_test)`` di ``np.ndarray``.

    Raises:
        ValueError: Se ``dataset_name`` non e' presente in
            ``EVALUATION_DATASETS``.
    """
    if dataset_name not in EVALUATION_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' non trovato in EVALUATION_DATASETS. "
            f"Dataset disponibili: {list(EVALUATION_DATASETS.keys())}"
        )

    dataset_id = EVALUATION_DATASETS[dataset_name]
    X, y = load_dataset(dataset_id)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 4. get_all_finetune_data
# ---------------------------------------------------------------------------

def get_all_finetune_data(
    val_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Carica e prepara tutti i dataset di fine-tuning.

    Per ogni dataset in ``FINETUNE_DATASETS`` chiama :func:`get_finetune_data`
    e ne raccoglie gli split in un dizionario annidato.

    Args:
        val_size: Frazione di dati da destinare al validation set
            (default ``0.2``).
        random_state: Seed per la riproducibilita' degli split (default ``42``).

    Returns:
        Dizionario con la struttura::

            {
                "diabetes": {
                    "X_train": ..., "y_train": ...,
                    "X_val": ...,   "y_val": ...
                },
                "breast_cancer": { ... },
                ...
            }

        I dataset che falliscono il caricamento vengono saltati con un
        messaggio di avviso e non compaiono nel dizionario risultante.
    """
    all_data: Dict[str, Dict[str, np.ndarray]] = {}

    for dataset_name in FINETUNE_DATASETS:
        try:
            X_train, y_train, X_val, y_val = get_finetune_data(
                dataset_name,
                val_size=val_size,
                random_state=random_state,
            )
            all_data[dataset_name] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
            }
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERROR] Impossibile caricare il dataset di fine-tuning "
                f"'{dataset_name}': {exc}"
            )

    return all_data


# ---------------------------------------------------------------------------
# 5. print_dataset_summary
# ---------------------------------------------------------------------------

def print_dataset_summary(datasets: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Stampa una tabella riassuntiva dei dataset di fine-tuning.

    Per ogni dataset vengono riportati:
        * nome
        * numero di campioni di training
        * numero di campioni di validation
        * numero di feature
        * class balance, ovvero la percentuale di campioni appartenenti alla
          classe positiva (etichetta ``1``) sull'intero dataset.

    Args:
        datasets: Dizionario nel formato restituito da
            :func:`get_all_finetune_data`.

    Returns:
        ``None``. La funzione stampa direttamente su standard output.
    """
    if not datasets:
        print("Nessun dataset disponibile per il riepilogo.")
        return

    header = (
        f"{'dataset':<18} | {'n_train':>8} | {'n_val':>7} | "
        f"{'n_features':>10} | {'class_balance':>14}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("RIEPILOGO DATASET DI FINE-TUNING")
    print(separator)
    print(header)
    print(separator)

    for name, splits in datasets.items():
        try:
            y_train = splits["y_train"]
            y_val = splits["y_val"]
            X_train = splits["X_train"]

            n_train = int(X_train.shape[0])
            n_val = int(splits["X_val"].shape[0])
            n_features = int(X_train.shape[1])

            # Class balance calcolato sull'intero dataset (train + val).
            y_all = np.concatenate([y_train, y_val])
            positive_ratio = float(np.mean(y_all == 1)) * 100.0

            print(
                f"{name:<18} | {n_train:>8} | {n_val:>7} | "
                f"{n_features:>10} | {positive_ratio:>13.2f}%"
            )
        except (KeyError, AttributeError, ValueError) as exc:
            print(f"{name:<18} | dati non validi: {exc}")

    print(separator + "\n")


# ---------------------------------------------------------------------------
# Blocco di test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST DEL MODULO data_loader.py")
    print("=" * 70)

    # --- Test 1: load_dataset --------------------------------------------
    print("\n[TEST 1] load_dataset() su 'diabetes'...")
    try:
        X, y = load_dataset(FINETUNE_DATASETS["diabetes"])
        print(f"  -> X shape: {X.shape}, dtype: {X.dtype}")
        print(f"  -> y shape: {y.shape}, classi: {np.unique(y)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FALLITO] {exc}")

    # --- Test 2: get_finetune_data ---------------------------------------
    print("\n[TEST 2] get_finetune_data() su 'diabetes'...")
    try:
        X_train, y_train, X_val, y_val = get_finetune_data("diabetes")
        print(f"  -> train: {X_train.shape}, val: {X_val.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FALLITO] {exc}")

    # --- Test 3: get_evaluation_data -------------------------------------
    print("\n[TEST 3] get_evaluation_data() su 'credit_g'...")
    try:
        X_tr, y_tr, X_te, y_te = get_evaluation_data("credit_g")
        print(f"  -> train: {X_tr.shape}, test: {X_te.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FALLITO] {exc}")

    # --- Test 4: gestione errori (ValueError) ----------------------------
    print("\n[TEST 4] Gestione dataset inesistente...")
    try:
        get_finetune_data("dataset_inesistente")
        print("  [FALLITO] Avrebbe dovuto lanciare ValueError")
    except ValueError as exc:
        print(f"  -> ValueError correttamente sollevato: {exc}")

    # --- Test 5: get_all_finetune_data + print_dataset_summary -----------
    print("\n[TEST 5] get_all_finetune_data() + print_dataset_summary()...")
    try:
        all_data = get_all_finetune_data()
        print_dataset_summary(all_data)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FALLITO] {exc}")

    print("=" * 70)
    print("TEST COMPLETATI")
    print("=" * 70)
