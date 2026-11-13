"""Evaluation package: metrics, model comparison and error analysis."""

from evaluation.comparison import compare_predictions, holm_bonferroni
from evaluation.error_analysis import error_distribution, hard_examples
from evaluation.metrics import (
    EvaluationError,
    binary_metrics,
    bootstrap_confidence_intervals,
    evaluate_predictions,
    expected_calibration_error,
)

__all__ = [
    "EvaluationError",
    "binary_metrics",
    "bootstrap_confidence_intervals",
    "compare_predictions",
    "error_distribution",
    "evaluate_predictions",
    "expected_calibration_error",
    "hard_examples",
    "holm_bonferroni",
]
