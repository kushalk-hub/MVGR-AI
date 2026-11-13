"""Model comparison helpers (Person 1).

Comparisons are only valid on identical split IDs and identical sample sets.
Both are enforced here rather than left to convention.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics import (
    EvaluationError,
    bootstrap_metric_delta,
    binary_metrics,
)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(p_values) - rank) * float(p_values[index])
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def compare_predictions(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    metric: str = "macro_f1",
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    for name, frame in (("a", predictions_a), ("b", predictions_b)):
        if "split_id" not in frame:
            raise EvaluationError(f"predictions_{name} has no split_id")
    if predictions_a["split_id"].nunique() != 1 or predictions_b["split_id"].nunique() != 1:
        raise EvaluationError("each prediction file must target exactly one split_id")
    split_a = predictions_a["split_id"].iloc[0]
    split_b = predictions_b["split_id"].iloc[0]
    if split_a != split_b:
        raise EvaluationError(f"cannot compare different splits: {split_a} vs {split_b}")

    columns = ["sample_id", "predicted_score"]
    merged = (
        predictions_a[columns]
        .rename(columns={"predicted_score": "score_a"})
        .merge(
            predictions_b[columns].rename(columns={"predicted_score": "score_b"}),
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(labels[["sample_id", "label"]], on="sample_id", how="inner", validate="one_to_one")
    )
    if len(merged) != len(predictions_a) or len(merged) != len(predictions_b):
        raise EvaluationError("prediction files do not cover the same sample set")

    delta = bootstrap_metric_delta(
        merged["label"],
        merged["score_a"],
        merged["score_b"],
        metric=metric,
        resamples=resamples,
        seed=seed,
    )
    return {
        "split_id": split_a,
        "n": int(len(merged)),
        "metric": metric,
        "model_a": binary_metrics(merged["label"], merged["score_a"]),
        "model_b": binary_metrics(merged["label"], merged["score_b"]),
        "paired_delta": delta,
    }


def attach_holm(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    p_values = [float(result["paired_delta"]["p_two_sided"]) for result in results]
    adjusted = holm_bonferroni(p_values)
    for result, value in zip(results, adjusted):
        result["paired_delta"]["p_holm"] = value
        result["paired_delta"]["significant_at_0_05"] = bool(value < 0.05)
    return results
