"""Pipeline esterna per TabPFN: ensembling, calibrazione, ottimizzazione soglia."""

from .external_pipeline import (
    average_probas,
    fit_calibrator,
    optimize_threshold,
    run_calibration_threshold_experiment,
    run_ensemble_experiment,
)

__all__ = [
    "average_probas",
    "fit_calibrator",
    "optimize_threshold",
    "run_calibration_threshold_experiment",
    "run_ensemble_experiment",
]
