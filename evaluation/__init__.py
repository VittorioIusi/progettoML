"""Package per le metriche e la valutazione dei modelli del progetto TabPFN."""

from .metrics import (
    compute_auc_roc,
    compute_brier_score,
    compute_ece,
    compute_f1,
    evaluate_model,
    evaluate_multiple_datasets,
    print_comparison_table,
    save_results,
)

__all__ = [
    "compute_auc_roc",
    "compute_brier_score",
    "compute_ece",
    "compute_f1",
    "evaluate_model",
    "evaluate_multiple_datasets",
    "print_comparison_table",
    "save_results",
]
