"""
context_engineering.py
======================

Funzioni **pure** per il *context engineering* di TabPFN-2.5: invece di toccare
i pesi (come fa il LoRA), si interviene sul **contesto** che il modello riceve
in-context. E' il terzo asse del progetto, complementare a:

    * fine-tuning LoRA  -> modifica i pesi (si e' visto: near-optimal, non aiuta);
    * pipeline esterna  -> modifica l'output (calibrazione/soglia/ensembling);
    * context engineering -> modifica l'input/contesto (questo modulo).

Razionale: il meccanismo adattivo di TabPFN e' l'attenzione sul set di contesto.
Quindi la leva naturale non e' aggiornare i pesi ma **scegliere e bilanciare**
gli esempi che compongono il contesto. Qui stanno gli strumenti di base:

    * ``select_by_clustering``   -> riduce il contesto ai K esempi piu'
      rappresentativi via KMeans per-classe (instance selection);
    * ``balance_context_smote``  -> ribilancia il contesto con SMOTE;
    * ``balance_context_undersample`` -> ribilancia con random undersampling.

Tutte le funzioni sono pure (array NumPy in ingresso/uscita), non importano
TabPFN e sono testabili su CPU (vedi il blocco ``__main__``). Gli esperimenti
che le usano stanno in ``context_experiments.py``.

Dipendenze:
    pip install scikit-learn numpy    (+ imbalanced-learn per le funzioni SMOTE/
    undersampling)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Instance selection via clustering per-classe
# ---------------------------------------------------------------------------

def select_by_clustering(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    random_state: int = 0,
) -> np.ndarray:
    """
    Seleziona i ``k`` esempi piu' rappresentativi del contesto via KMeans.

    Strategia (pensata per preservare la distribuzione delle classi):
        1. Le feature sono standardizzate (``StandardScaler``) cosi' il
           clustering non e' dominato dalle colonne a range piu' ampio. Lo
           scaler e' fittato sugli stessi dati passati (il contesto).
        2. Il budget ``k`` e' ripartito tra le classi in modo proporzionale
           alla loro frequenza: ``k_c = round(k * n_c / n)`` (minimo 1). Questo
           mantiene la stessa proporzione di classi del contesto originale.
        3. Per ogni classe si esegue un KMeans con ``k_c`` cluster; per ogni
           centroide si prende l'**esempio reale piu' vicino** (non il
           centroide sintetico), evitando duplicati.

    A differenza del semplice random subsampling, i punti scelti coprono le
    regioni dense dello spazio delle feature (uno per "modo" della
    distribuzione), quindi il contesto ridotto resta informativo.

    Args:
        X: Feature del contesto, shape ``(n_samples, n_features)``.
        y: Etichette binarie (0/1), shape ``(n_samples,)``.
        k: Numero totale di esempi da selezionare. Se ``k >= n_samples`` la
           funzione restituisce tutti gli indici (nessuna riduzione).
        random_state: Seme per la riproducibilita' del KMeans.

    Returns:
        Array ordinato di indici (interi, riferiti alle righe di ``X``/``y``)
        degli esempi selezionati. La lunghezza e' ``<= k`` (puo' essere
        leggermente inferiore per arrotondamenti o classi piu' piccole della
        loro quota).

    Raises:
        ValueError: Se ``k <= 0`` o gli input non sono coerenti.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X ({X.shape[0]}) e y ({y.shape[0]}) hanno lunghezze diverse."
        )
    if k <= 0:
        raise ValueError(f"k deve essere positivo, ricevuto {k}.")

    n = X.shape[0]
    if k >= n:
        # Nessuna riduzione possibile: restituisce tutti gli indici.
        return np.arange(n)

    # Standardizzazione: il clustering lavora su distanze euclidee, quindi le
    # feature devono essere sulla stessa scala.
    Xs = StandardScaler().fit_transform(X)

    classes, counts = np.unique(y, return_counts=True)
    selected: list[int] = []

    for cls, cnt in zip(classes, counts):
        # Quota di budget per questa classe, proporzionale alla sua frequenza.
        k_c = int(round(k * cnt / n))
        k_c = max(1, min(k_c, cnt))  # almeno 1, non piu' dei disponibili

        idx_c = np.where(y == cls)[0]
        Xc = Xs[idx_c]

        if k_c >= len(idx_c):
            # La quota copre tutta la classe: prendila per intero.
            selected.extend(idx_c.tolist())
            continue

        km = KMeans(n_clusters=k_c, random_state=random_state, n_init=10)
        km.fit(Xc)

        # Per ogni centroide, l'esempio reale piu' vicino (senza ripetizioni).
        taken: set[int] = set()
        for center in km.cluster_centers_:
            dists = np.linalg.norm(Xc - center, axis=1)
            for j in np.argsort(dists):
                real_idx = int(idx_c[j])
                if real_idx not in taken:
                    taken.add(real_idx)
                    break
        selected.extend(sorted(taken))

    return np.array(sorted(set(selected)), dtype=int)


# ---------------------------------------------------------------------------
# 2. Ribilanciamento del contesto: SMOTE
# ---------------------------------------------------------------------------

def balance_context_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ribilancia il contesto sovracampionando la classe minoritaria con SMOTE.

    SMOTE (Synthetic Minority Over-sampling Technique) genera nuovi esempi
    sintetici della classe minoritaria interpolando tra vicini reali, fino a
    pareggiare le classi. Applicato **solo** al contesto (train), mai a
    calibrazione o test, per non introdurre leakage.

    Args:
        X: Feature del contesto, shape ``(n_samples, n_features)``.
        y: Etichette binarie (0/1), shape ``(n_samples,)``.
        random_state: Seme per la riproducibilita'.

    Returns:
        Tupla ``(X_res, y_res)`` con il contesto ribilanciato (classi ~50/50).

    Raises:
        ImportError: Se ``imbalanced-learn`` non e' installato.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:  # pragma: no cover - dipendenza esterna
        raise ImportError(
            "balance_context_smote richiede 'imbalanced-learn'. "
            "Installa con: pip install imbalanced-learn"
        ) from exc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int).ravel()

    # k_neighbors non puo' superare (n_minoritari - 1): lo adattiamo per i
    # contesti molto sbilanciati/piccoli, altrimenti SMOTE solleverebbe errore.
    counts = np.bincount(y)
    n_minority = int(counts[counts > 0].min())
    k_neighbors = max(1, min(5, n_minority - 1))

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    return np.asarray(X_res, dtype=float), np.asarray(y_res).astype(int)


# ---------------------------------------------------------------------------
# 3. Ribilanciamento del contesto: random undersampling
# ---------------------------------------------------------------------------

def balance_context_undersample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ribilancia il contesto sottocampionando la classe maggioritaria.

    Rimuove esempi a caso dalla classe maggioritaria fino a pareggiare le
    classi. A differenza di SMOTE non introduce dati sintetici, ma scarta
    informazione (il contesto risultante e' piu' piccolo). Applicato solo al
    contesto (train), mai a calibrazione o test.

    Args:
        X: Feature del contesto, shape ``(n_samples, n_features)``.
        y: Etichette binarie (0/1), shape ``(n_samples,)``.
        random_state: Seme per la riproducibilita'.

    Returns:
        Tupla ``(X_res, y_res)`` con il contesto ribilanciato (classi ~50/50).

    Raises:
        ImportError: Se ``imbalanced-learn`` non e' installato.
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as exc:  # pragma: no cover - dipendenza esterna
        raise ImportError(
            "balance_context_undersample richiede 'imbalanced-learn'. "
            "Installa con: pip install imbalanced-learn"
        ) from exc

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int).ravel()

    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    return np.asarray(X_res, dtype=float), np.asarray(y_res).astype(int)


# ---------------------------------------------------------------------------
# Blocco di test (CPU, senza TabPFN): verifica le funzioni pure
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST helper di context_engineering.py (dati sintetici, no TabPFN)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    # Dataset sbilanciato: 900 negativi, 100 positivi, 5 feature con scale
    # molto diverse (per testare la standardizzazione nel clustering).
    n_neg, n_pos = 900, 100
    X_neg = rng.normal(0, 1, size=(n_neg, 5)) * np.array([1, 10, 100, 0.1, 1])
    X_pos = rng.normal(3, 1, size=(n_pos, 5)) * np.array([1, 10, 100, 0.1, 1])
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * n_neg + [1] * n_pos)

    # --- select_by_clustering -------------------------------------------
    for k in (100, 200, 500):
        idx = select_by_clustering(X, y, k=k, random_state=42)
        frac_pos = float(np.mean(y[idx] == 1))
        print(
            f"\n[select k={k}] selezionati={len(idx)} "
            f"(<= {k}) | frazione positivi={frac_pos:.3f} "
            f"(atteso ~0.100, distribuzione preservata)"
        )
        assert len(idx) <= k, "Non deve selezionare piu' di k esempi."
        assert len(set(idx.tolist())) == len(idx), "Indici duplicati."

    # k >= n: nessuna riduzione.
    idx_all = select_by_clustering(X, y, k=5000, random_state=0)
    assert len(idx_all) == len(y), "k>=n deve restituire tutti gli indici."
    print(f"\n[select k=5000>=n] restituiti tutti i {len(idx_all)} indici. OK")

    # --- balance_context_smote / undersample (se imblearn c'e') ---------
    try:
        Xs, ys = balance_context_smote(X, y, random_state=0)
        c = np.bincount(ys)
        print(f"\n[SMOTE] {np.bincount(y)} -> {c} (atteso classi pari). OK")
        assert c[0] == c[1], "SMOTE deve pareggiare le classi."

        Xu, yu = balance_context_undersample(X, y, random_state=0)
        cu = np.bincount(yu)
        print(f"[undersample] {np.bincount(y)} -> {cu} "
              f"(atteso pari, totale ridotto). OK")
        assert cu[0] == cu[1], "L'undersampling deve pareggiare le classi."
    except ImportError as exc:
        print(f"\n[SKIP] imbalanced-learn non installato: {exc}")

    print("\n" + "=" * 70)
    print("TUTTI I TEST DEGLI HELPER SUPERATI")
    print("=" * 70)
