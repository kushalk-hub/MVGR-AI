"""Error analysis (Person 1).

Labels are joined for inspection only. Text is included solely to let a human
review examples; it must never be fed back into a model during evaluation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.metrics import EvaluationError


def error_frame(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    merged = predictions.merge(labels, on="sample_id", how="inner", validate="one_to_one")
    merged["predicted_label"] = (merged["predicted_score"] >= threshold).astype(int)
    merged["error_type"] = "correct"
    merged.loc[
        (merged["predicted_label"] == 1) & (merged["label"] == 0), "error_type"
    ] = "false_positive"
    merged.loc[
        (merged["predicted_label"] == 0) & (merged["label"] == 1), "error_type"
    ] = "false_negative"
    merged["confidence"] = (merged["predicted_score"] - 0.5).abs() * 2.0
    return merged


def hard_examples(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    k: int = 20,
    columns: tuple[str, ...] = ("sample_id", "predicted_score", "error_type"),
) -> dict[str, list[dict[str, Any]]]:
    frame = error_frame(predictions, labels)
    result: dict[str, list[dict[str, Any]]] = {}
    for error_type in ("false_positive", "false_negative"):
        subset = frame[frame["error_type"] == error_type].sort_values(
            "confidence", ascending=False
        )
        keep = [column for column in columns if column in subset.columns]
        result[error_type] = subset.head(k)[keep].to_dict(orient="records")
    return result


def error_distribution(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    strata: tuple[str, ...] = ("source_dataset", "source_task", "text_type"),
) -> dict[str, dict[str, float]]:
    frame = error_frame(predictions, labels)
    if "label" not in frame:
        raise EvaluationError("labels were not joined")
    result: dict[str, dict[str, float]] = {}
    for column in strata:
        if column not in frame:
            continue
        block: dict[str, dict[str, float]] = {}
        for value, cell in frame.groupby(column, dropna=False):
            errors = cell[cell["error_type"] != "correct"]
            block[str(value)] = {
                "n": float(len(cell)),
                "error_rate": float(len(errors) / len(cell)),
                "false_positives": float((cell["error_type"] == "false_positive").sum()),
                "false_negatives": float((cell["error_type"] == "false_negative").sum()),
            }
        result[column] = block
    return result
