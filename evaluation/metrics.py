"""
metrics.py
==========

Metriche di valutazione per la classificazione binaria usate nel progetto
TabPFN.

Il modulo raccoglie:
    * Metriche di discriminazione: AUC-ROC, F1-score (macro).
    * Metriche di calibrazione: Brier Score, Expected Calibration Error (ECE).
    * Funzioni di alto livello per valutare un modello, aggregare piu'
      risultati in un DataFrame, salvarli su CSV e stamparli in forma di
      tabella comparativa.

Dipendenze:
    pip install scikit-learn numpy pandas
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Funzioni di supporto interne
# ---------------------------------------------------------------------------

def _validate_probabilities(y_prob: np.ndarray) -> np.ndarray:
    """
    Valida un vettore di probabilita' per la classe positiva.

    Controlla che il vettore non contenga valori ``NaN``/``inf`` e che tutti
    i valori siano compresi nell'intervallo ``[0, 1]``.

    Args:
        y_prob: Array di probabilita' della classe positiva.

    Returns:
        L'array convertito in ``np.ndarray`` di tipo ``float``.

    Raises:
        ValueError: Se ``y_prob`` contiene NaN/inf o valori fuori da ``[0, 1]``.
    """
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.size == 0:
        raise ValueError("y_prob e' vuoto: impossibile calcolare la metrica.")

    if np.isnan(y_prob).any():
        raise ValueError(
            "y_prob contiene valori NaN. Verifica le probabilita' predette "
            "dal modello prima di calcolare le metriche."
        )

    if np.isinf(y_prob).any():
        raise ValueError(
            "y_prob contiene valori infiniti (inf). Le probabilita' devono "
            "essere finite e comprese in [0, 1]."
        )

    out_of_range = (y_prob < 0.0) | (y_prob > 1.0)
    if out_of_range.any():
        n_bad = int(out_of_range.sum())
        raise ValueError(
            f"y_prob contiene {n_bad} valori fuori dall'intervallo [0, 1]. "
            f"Min={y_prob.min():.4f}, Max={y_prob.max():.4f}. Assicurati di "
            f"passare le probabilita' (predict_proba), non i logit."
        )

    return y_prob


def _validate_labels(y_true: np.ndarray) -> np.ndarray:
    """
    Valida un vettore di etichette vere binarie.

    Args:
        y_true: Array di etichette vere (attese 0/1).

    Returns:
        L'array convertito in ``np.ndarray`` di tipo ``int``.

    Raises:
        ValueError: Se ``y_true`` e' vuoto o contiene valori NaN.
    """
    y_true = np.asarray(y_true)

    if y_true.size == 0:
        raise ValueError("y_true e' vuoto: impossibile calcolare la metrica.")

    if np.isnan(np.asarray(y_true, dtype=float)).any():
        raise ValueError("y_true contiene valori NaN.")

    return y_true.astype(int)


# ---------------------------------------------------------------------------
# 1. compute_auc_roc
# ---------------------------------------------------------------------------

def compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calcola l'AUC-ROC (Area Under the ROC Curve).

    L'AUC-ROC misura la capacita' del modello di distinguere tra classe
    positiva e negativa: 1.0 indica separazione perfetta, 0.5 indica un
    classificatore casuale.

    Args:
        y_true: Array delle etichette vere (0 o 1), shape ``(n_samples,)``.
        y_prob: Array delle probabilita' della classe positiva (colonna 1 di
            ``predict_proba``), shape ``(n_samples,)``.

    Returns:
        Valore AUC-ROC (float) arrotondato a 4 decimali.

    Raises:
        ValueError: Se ``y_prob`` non e' valido o se ``y_true`` contiene una
            sola classe (AUC non definita).
    """
    y_true = _validate_labels(y_true)
    y_prob = _validate_probabilities(y_prob)

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "AUC-ROC non definita: y_true contiene una sola classe. "
            "Servono sia campioni positivi sia negativi."
        )

    return round(float(roc_auc_score(y_true, y_prob)), 4)


# ---------------------------------------------------------------------------
# 2. compute_f1
# ---------------------------------------------------------------------------

def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcola l'F1-score macro-averaged.

    L'F1 macro calcola l'F1 per ciascuna classe e ne fa la media non pesata,
    trattando quindi le due classi con la stessa importanza
    (utile in presenza di sbilanciamento).

    Args:
        y_true: Array delle etichette vere (0 o 1), shape ``(n_samples,)``.
        y_pred: Array delle classi predette (0 o 1), **non** le probabilita',
            shape ``(n_samples,)``.

    Returns:
        Valore F1 macro (float) arrotondato a 4 decimali.

    Raises:
        ValueError: Se ``y_true``/``y_pred`` sono vuoti o di lunghezza diversa.
    """
    y_true = _validate_labels(y_true)
    y_pred = np.asarray(y_pred).astype(int)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true ({y_true.shape[0]}) e y_pred ({y_pred.shape[0]}) hanno "
            f"lunghezze diverse."
        )

    return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)


# ---------------------------------------------------------------------------
# 3. compute_brier_score
# ---------------------------------------------------------------------------

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calcola il Brier Score.

    Il Brier Score e' l'errore quadratico medio tra probabilita' predette e
    label reali: ``mean((y_prob - y_true)^2)``. Valori piu' bassi indicano una
    migliore calibrazione (0 = perfetto, 1 = pessimo).

    Args:
        y_true: Array delle etichette vere (0 o 1), shape ``(n_samples,)``.
        y_prob: Array delle probabilita' della classe positiva,
            shape ``(n_samples,)``.

    Returns:
        Valore Brier Score (float) arrotondato a 4 decimali.

    Raises:
        ValueError: Se ``y_prob`` non e' valido o le lunghezze non coincidono.
    """
    y_true = _validate_labels(y_true)
    y_prob = _validate_probabilities(y_prob)

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"y_true ({y_true.shape[0]}) e y_prob ({y_prob.shape[0]}) hanno "
            f"lunghezze diverse."
        )

    return round(float(np.mean((y_prob - y_true) ** 2)), 4)


# ---------------------------------------------------------------------------
# 4. compute_ece
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calcola l'Expected Calibration Error (ECE).

    Le probabilita' predette vengono suddivise in ``n_bins`` bin di uguale
    ampiezza sull'intervallo ``[0, 1]``. Per ogni bin si calcola il valore
    assoluto della differenza tra la confidenza media (``mean(y_prob)``) e
    l'accuratezza media (``mean(y_true)``); il contributo di ogni bin e'
    pesato per la frazione di campioni che contiene::

        ECE = sum_b (|bin_b| / N) * |mean(y_prob_b) - mean(y_true_b)|

    Valori piu' bassi indicano una calibrazione migliore.

    Args:
        y_true: Array delle etichette vere (0 o 1), shape ``(n_samples,)``.
        y_prob: Array delle probabilita' della classe positiva,
            shape ``(n_samples,)``.
        n_bins: Numero di bin di uguale ampiezza (default ``10``).

    Returns:
        Valore ECE (float) arrotondato a 4 decimali.

    Raises:
        ValueError: Se ``y_prob`` non e' valido, le lunghezze non coincidono o
            ``n_bins`` non e' un intero positivo.
    """
    y_true = _validate_labels(y_true)
    y_prob = _validate_probabilities(y_prob)

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"y_true ({y_true.shape[0]}) e y_prob ({y_prob.shape[0]}) hanno "
            f"lunghezze diverse."
        )

    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError(f"n_bins deve essere un intero positivo, ricevuto {n_bins!r}.")

    n_samples = y_true.shape[0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # np.digitize assegna ogni probabilita' a un bin; right=True garantisce
    # che il valore esatto 1.0 cada nell'ultimo bin e non in uno fuori range.
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        avg_confidence = float(np.mean(y_prob[mask]))
        avg_accuracy = float(np.mean(y_true[mask]))
        ece += (bin_count / n_samples) * abs(avg_confidence - avg_accuracy)

    return round(float(ece), 4)


# ---------------------------------------------------------------------------
# 5. evaluate_model
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Valuta un modello calcolando tutte le metriche del modulo.

    Le classi predette ``y_pred`` vengono derivate dalle probabilita'
    applicando la soglia ``threshold`` (default ``0.5``).

    Args:
        y_true: Array delle etichette vere (0 o 1), shape ``(n_samples,)``.
        y_prob: Array delle probabilita' della classe positiva,
            shape ``(n_samples,)``.
        model_name: Nome del modello valutato (es. ``"TabPFN"``).
        dataset_name: Nome del dataset su cui e' avvenuta la valutazione.
        threshold: Soglia per binarizzare le probabilita' in classi
            (default ``0.5``).

    Returns:
        Dizionario con le metriche::

            {
                "model":       model_name,
                "dataset":     dataset_name,
                "auc_roc":     ...,
                "f1":          ...,
                "brier_score": ...,
                "ece":         ...,
            }

    Raises:
        ValueError: Se gli input non sono validi (vedi le singole metriche).
    """
    y_true = _validate_labels(y_true)
    y_prob = _validate_probabilities(y_prob)

    # Classi predette dalla soglia.
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "model": model_name,
        "dataset": dataset_name,
        "auc_roc": compute_auc_roc(y_true, y_prob),
        "f1": compute_f1(y_true, y_pred),
        "brier_score": compute_brier_score(y_true, y_prob),
        "ece": compute_ece(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# 6. evaluate_multiple_datasets
# ---------------------------------------------------------------------------

def evaluate_multiple_datasets(results_list: List[Dict[str, object]]) -> pd.DataFrame:
    """
    Aggrega in un DataFrame i risultati di piu' valutazioni.

    Args:
        results_list: Lista di dizionari come quelli restituiti da
            :func:`evaluate_model`.

    Returns:
        ``pd.DataFrame`` con una riga per risultato e una colonna aggiuntiva
        ``timestamp`` contenente data e ora dell'esperimento.

    Raises:
        ValueError: Se ``results_list`` e' vuoto o non e' una lista di dict.
    """
    if not isinstance(results_list, list) or len(results_list) == 0:
        raise ValueError(
            "results_list deve essere una lista non vuota di dizionari "
            "(output di evaluate_model)."
        )

    if not all(isinstance(item, dict) for item in results_list):
        raise ValueError("Tutti gli elementi di results_list devono essere dict.")

    df = pd.DataFrame(results_list)
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df


# ---------------------------------------------------------------------------
# 7. save_results
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, filepath: str) -> None:
    """
    Salva i risultati su CSV, appendendo se il file esiste gia'.

    Se ``filepath`` non esiste, viene creato (insieme alle cartelle mancanti)
    con l'header. Se esiste, le nuove righe vengono accodate senza header e
    senza sovrascrivere i dati precedenti.

    Args:
        df: DataFrame dei risultati da salvare.
        filepath: Percorso del file CSV di destinazione.

    Returns:
        ``None``. Stampa una conferma con il numero di righe aggiunte.

    Raises:
        ValueError: Se ``df`` non e' un DataFrame o e' vuoto.
        OSError: Se la scrittura su disco fallisce.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df deve essere un pandas.DataFrame.")
    if df.empty:
        raise ValueError("df e' vuoto: nessun risultato da salvare.")

    # Crea le eventuali cartelle mancanti del percorso.
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    file_exists = os.path.exists(filepath)

    try:
        df.to_csv(
            filepath,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )
    except OSError as exc:
        raise OSError(f"Impossibile salvare i risultati in '{filepath}': {exc}") from exc

    action = "aggiunte a" if file_exists else "salvate in"
    print(f"[SAVE] {len(df)} righe {action} '{filepath}'.")


# ---------------------------------------------------------------------------
# 8. print_comparison_table
# ---------------------------------------------------------------------------

def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Stampa una tabella comparativa dei modelli, raggruppata per dataset.

    Per ogni dataset viene stampato un blocco con una riga per modello e le
    relative metriche. Il valore migliore di ciascuna metrica all'interno del
    dataset viene evidenziato con un asterisco ``*``. La direzione di
    ottimalita' dipende dalla metrica:
        * AUC-ROC e F1: piu' alto e' meglio;
        * Brier Score ed ECE: piu' basso e' meglio.

    Args:
        df: DataFrame dei risultati (output di
            :func:`evaluate_multiple_datasets` o lettura da CSV).

    Returns:
        ``None``. La funzione stampa direttamente su standard output.

    Raises:
        ValueError: Se ``df`` non e' un DataFrame o mancano colonne richieste.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df deve essere un pandas.DataFrame.")
    if df.empty:
        print("Nessun risultato da confrontare.")
        return

    required = {"model", "dataset", "auc_roc", "f1", "brier_score", "ece"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel DataFrame: {sorted(missing)}.")

    # Metriche e direzione di ottimalita': True = piu' alto meglio.
    metrics = {
        "auc_roc": True,
        "f1": True,
        "brier_score": False,
        "ece": False,
    }

    header = (
        f"{'model':<18} | {'auc_roc':>9} | {'f1':>9} | "
        f"{'brier':>9} | {'ece':>9}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("TABELLA COMPARATIVA DEI MODELLI")
    print("=" * len(header))

    for dataset_name, group in df.groupby("dataset"):
        print(f"\nDataset: {dataset_name}")
        print(separator)
        print(header)
        print(separator)

        # Determina il valore migliore per ogni metrica in questo dataset.
        best_values = {}
        for metric, higher_is_better in metrics.items():
            col = pd.to_numeric(group[metric], errors="coerce")
            best_values[metric] = col.max() if higher_is_better else col.min()

        for _, row in group.iterrows():
            cells = []
            for metric in metrics:
                value = float(row[metric])
                marker = "*" if np.isclose(value, best_values[metric]) else " "
                cells.append(f"{value:>8.4f}{marker}")
            print(
                f"{str(row['model']):<18} | {cells[0]:>9} | {cells[1]:>9} | "
                f"{cells[2]:>9} | {cells[3]:>9}"
            )
        print(separator)

    print("\n(* = valore migliore per la metrica nel dataset)\n")


# ---------------------------------------------------------------------------
# Blocco di test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST DEL MODULO metrics.py (dati sintetici)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n = 500

    # --- Genera dati sintetici -------------------------------------------
    # y_true bilanciato; le probabilita' di un "buon" modello sono correlate
    # con la verita', quelle di un modello "casuale" sono indipendenti.
    y_true = rng.integers(0, 2, size=n)

    # Modello buono: probabilita' spinte verso la classe corretta + rumore.
    noise = rng.normal(0, 0.15, size=n)
    prob_good = np.clip(y_true * 0.7 + 0.15 + noise, 0.0, 1.0)

    # Modello casuale: probabilita' uniformi, scorrelate dal target.
    prob_random = rng.uniform(0.0, 1.0, size=n)

    # --- Test singole metriche -------------------------------------------
    print("\n[TEST] Metriche singole sul modello 'buono':")
    print(f"  AUC-ROC     : {compute_auc_roc(y_true, prob_good)}")
    print(f"  F1 (macro)  : {compute_f1(y_true, (prob_good >= 0.5).astype(int))}")
    print(f"  Brier Score : {compute_brier_score(y_true, prob_good)}")
    print(f"  ECE         : {compute_ece(y_true, prob_good)}")

    # --- Test evaluate_model ---------------------------------------------
    print("\n[TEST] evaluate_model() su due modelli:")
    res_good = evaluate_model(y_true, prob_good, "TabPFN", "synthetic")
    res_random = evaluate_model(y_true, prob_random, "RandomBaseline", "synthetic")
    print(f"  {res_good}")
    print(f"  {res_random}")

    # Secondo dataset sintetico per testare il raggruppamento.
    y_true2 = rng.integers(0, 2, size=n)
    prob_good2 = np.clip(y_true2 * 0.6 + 0.2 + rng.normal(0, 0.2, size=n), 0, 1)
    res_good2 = evaluate_model(y_true2, prob_good2, "TabPFN", "synthetic_2")
    res_random2 = evaluate_model(
        y_true2, rng.uniform(0, 1, size=n), "RandomBaseline", "synthetic_2"
    )

    # --- Test evaluate_multiple_datasets ---------------------------------
    print("\n[TEST] evaluate_multiple_datasets():")
    df = evaluate_multiple_datasets([res_good, res_random, res_good2, res_random2])
    print(df.to_string(index=False))

    # --- Test print_comparison_table -------------------------------------
    print_comparison_table(df)

    # --- Test save_results (append) --------------------------------------
    print("[TEST] save_results():")
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        "metrics_test.csv",
    )
    save_results(df, out_path)
    save_results(df, out_path)  # seconda chiamata: deve appendere

    # --- Test gestione errori --------------------------------------------
    print("\n[TEST] Gestione input non valido (NaN / fuori range):")
    for bad_prob, descr in [
        (np.array([0.1, np.nan, 0.5]), "con NaN"),
        (np.array([0.1, 1.5, 0.5]), "fuori range [0,1]"),
    ]:
        try:
            compute_auc_roc(np.array([0, 1, 1]), bad_prob)
            print(f"  [FALLITO] {descr}: avrebbe dovuto sollevare ValueError")
        except ValueError as exc:
            print(f"  -> ValueError ({descr}): {exc}")

    print("\n" + "=" * 70)
    print("TEST COMPLETATI")
    print("=" * 70)
