"""Shared evaluation metrics (Person 1).

Ground-truth labels are joined here and nowhere else. Prediction files must not
contain labels; ``evaluate_predictions`` fails if they do.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FORBIDDEN_PREDICTION_COLUMNS = {
    "label",
    "true_label",
    "ground_truth",
    "source_dataset",
    "source_task",
    "family_id",
    "text_type",
    "condition",
    "split",
}


class EvaluationError(ValueError):
    pass


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, *, bins: int = 10
) -> float | None:
    if len(y_true) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (y_score >= lower) & (y_score < upper if upper < 1.0 else y_score <= upper)
        if not mask.any():
            continue
        confidence = float(y_score[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += (mask.sum() / total) * abs(confidence - accuracy)
    return float(error)


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    if len(y_true_array) != len(y_score_array):
        raise EvaluationError("y_true and y_score length mismatch")
    if len(y_true_array) == 0:
        raise EvaluationError("cannot evaluate an empty set")
    predicted = (y_score_array >= threshold).astype(int)
    matrix = confusion_matrix(y_true_array, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
    return {
        "n": int(len(y_true_array)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, predicted)),
        "precision": float(precision_score(y_true_array, predicted, zero_division=0)),
        "recall": float(recall_score(y_true_array, predicted, zero_division=0)),
        "f1": float(f1_score(y_true_array, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y_true_array, predicted, average="macro", zero_division=0)),
        "roc_auc": _safe_auc(y_true_array, y_score_array),
        "brier_score": float(brier_score_loss(y_true_array, y_score_array)),
        "expected_calibration_error": expected_calibration_error(y_true_array, y_score_array),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "class_counts": {
            "human": int((y_true_array == 0).sum()),
            "ai": int((y_true_array == 1).sum()),
        },
    }


def _stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices: list[np.ndarray] = []
    for label in (0, 1):
        class_indices = np.flatnonzero(y_true == label)
        if class_indices.size == 0:
            continue
        indices.append(rng.choice(class_indices, size=class_indices.size, replace=True))
    return np.concatenate(indices)


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    metrics: Iterable[str] = ("macro_f1", "balanced_accuracy", "roc_auc"),
) -> dict[str, dict[str, float | None]]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(resamples):
        indices = _stratified_bootstrap_indices(y_true_array, rng)
        result = binary_metrics(y_true_array[indices], y_score_array[indices])
        for name in metrics:
            value = result.get(name)
            if value is not None:
                collected[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float | None]] = {}
    for name, values in collected.items():
        if not values:
            intervals[name] = {"low": None, "high": None, "n_resamples": 0}
            continue
        array = np.asarray(values)
        intervals[name] = {
            "low": float(np.quantile(array, alpha)),
            "high": float(np.quantile(array, 1.0 - alpha)),
            "mean": float(array.mean()),
            "n_resamples": int(array.size),
        }
    return intervals


def evaluate_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    strata: Sequence[str] = ("source_dataset", "source_task", "text_type"),
    min_cell_size: int = 30,
    bootstrap: bool = False,
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Join labels to predictions and produce overall + stratified metrics."""
    leaked = sorted(FORBIDDEN_PREDICTION_COLUMNS & set(predictions.columns))
    if leaked:
        raise EvaluationError(f"prediction frame contains target metadata: {leaked}")
    required = {"sample_id", "predicted_score"}
    missing = required - set(predictions.columns)
    if missing:
        raise EvaluationError(f"predictions missing columns {sorted(missing)}")

    merged = predictions.merge(
        labels[["sample_id", "label", *strata]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(predictions):
        raise EvaluationError(
            f"{len(predictions) - len(merged)} predictions have no ground-truth label; "
            "evaluation must cover the full split"
        )

    overall = binary_metrics(merged["label"], merged["predicted_score"])
    result: dict[str, Any] = {
        "overall": overall,
        "n": int(len(merged)),
        "strata": {},
        "small_cells": [],
    }
    if bootstrap:
        result["overall"]["confidence_intervals"] = bootstrap_confidence_intervals(
            merged["label"], merged["predicted_score"], resamples=resamples, seed=seed
        )

    for column in strata:
        if column not in merged:
            continue
        block: dict[str, Any] = {}
        for value, cell in merged.groupby(column, dropna=False):
            key = str(value)
            if len(cell) < min_cell_size:
                block[key] = {"n": int(len(cell)), "status": "insufficient_n"}
                result["small_cells"].append({"stratum": column, "value": key, "n": int(len(cell))})
                continue
            block[key] = binary_metrics(cell["label"], cell["predicted_score"])
        result["strata"][column] = block
    return result


def bootstrap_metric_delta(
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    *,
    metric: str = "macro_f1",
    threshold: float = 0.5,
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of metric(A) - metric(B) on identical resamples."""
    y_true_array = np.asarray(y_true, dtype=int)
    a = (np.asarray(score_a, dtype=float) >= threshold).astype(int)
    b = (np.asarray(score_b, dtype=float) >= threshold).astype(int)
    rng = np.random.default_rng(seed)

    def compute(predicted: np.ndarray, truth: np.ndarray) -> float:
        if metric == "macro_f1":
            from sklearn.metrics import f1_score

            return float(f1_score(truth, predicted, average="macro", zero_division=0))
        if metric == "accuracy":
            return float(accuracy_score(truth, predicted))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(truth, predicted))
        raise EvaluationError(f"unsupported delta metric {metric!r}")

    deltas: list[float] = []
    for _ in range(resamples):
        indices = _stratified_bootstrap_indices(y_true_array, rng)
        deltas.append(
            compute(a[indices], y_true_array[indices]) - compute(b[indices], y_true_array[indices])
        )
    array = np.asarray(deltas)
    return {
        "delta": compute(a, y_true_array) - compute(b, y_true_array),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "p_two_sided": float(min(1.0, 2.0 * min((array <= 0).mean(), (array >= 0).mean()))),
        "resamples": int(resamples),
    }
