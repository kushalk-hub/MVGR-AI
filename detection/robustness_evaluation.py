"""Paraphrase-robustness evaluation.

The detector is evaluated separately on every text category present in the
test split:

1. original human text
2. original AI-generated text
3. paraphrased human text
4. paraphrased AI-generated text
5. deeply paraphrased AI-generated text
6. any further combination present in the data (e.g. BiScope sentences)

Categories are derived from the dataset's own ``text_type`` and
``paraphrase_status`` columns. **No paraphrased text is synthesised** — every
row scored here is a row of ``10000.csv``.

Within a single-category subset the "positive" class is AI, so:

* on a human-only subset, recall/precision are undefined and the meaningful
  number is the **false-positive rate** (how often a human text is called AI);
* on an AI-only subset, recall is the **detection rate** and precision is the
  fraction of the model’s positive calls that are correct.

Both are reported, with ``NaN`` where a metric is not defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from detection.dataset_loader import LABEL_AI, LABEL_HUMAN

#: Human-readable names for the categories, keyed by the raw ``text_type``.
CATEGORY_LABELS: dict[str, str] = {
    "original": "Original",
    "paraphrased": "Paraphrased",
    "deep_paraphrased": "Deeply paraphrased",
    "unknown": "Unspecified (BiScope)",
}


@dataclass
class CategoryResult:
    """One category's performance on the test split."""

    category: str
    text_type: str
    label: int
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    mean_ai_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "text_type": self.text_type,
            "label": self.label,
            "n": self.n,
            "accuracy": round(float(self.accuracy), 6),
            "precision": None if np.isnan(self.precision) else round(float(self.precision), 6),
            "recall": None if np.isnan(self.recall) else round(float(self.recall), 6),
            "f1": None if np.isnan(self.f1) else round(float(self.f1), 6),
            "false_positive_rate": round(float(self.false_positive_rate), 6),
            "false_negative_rate": round(float(self.false_negative_rate), 6),
            "mean_ai_probability": round(float(self.mean_ai_probability), 6),
        }


def category_name(text_type: str, label: int) -> str:
    """Human-readable category name, e.g. ``Paraphrased AI-generated``."""
    base = CATEGORY_LABELS.get(text_type, text_type)
    origin = "AI-generated" if label == LABEL_AI else "Human"
    return f"{base} {origin}"


def evaluate_categories(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    text_type_column: str = "text_type",
    label_column: str = "label",
) -> list[CategoryResult]:
    """Score one model's predictions per text category."""
    frame = metadata.reset_index(drop=True).copy()
    frame["_y_true"] = np.asarray(y_true)
    frame["_y_pred"] = np.asarray(y_pred)
    if y_proba is not None:
        frame["_y_proba"] = np.asarray(y_proba)

    results: list[CategoryResult] = []
    for (text_type, label), group in frame.groupby(
        [text_type_column, label_column], sort=True
    ):
        n = len(group)
        human = group["_y_true"] == LABEL_HUMAN
        ai = group["_y_true"] == LABEL_AI
        called_ai = group["_y_pred"] == LABEL_AI
        called_human = group["_y_pred"] == LABEL_HUMAN
        true_positive = int((ai & called_ai).sum())
        false_positive = int((human & called_ai).sum())
        false_negative = int((ai & called_human).sum())
        true_negative = int((human & called_human).sum())

        accuracy = (true_positive + true_negative) / n if n else float("nan")
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else float("nan")
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else float("nan")
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else float("nan")
        )
        false_positive_rate = (
            false_positive / (false_positive + true_negative)
            if (false_positive + true_negative)
            else float("nan")
        )
        false_negative_rate = (
            false_negative / (false_negative + true_positive)
            if (false_negative + true_positive)
            else float("nan")
        )
        mean_probability = (
            float(group["_y_proba"].mean()) if "_y_proba" in group.columns else float("nan")
        )

        results.append(
            CategoryResult(
                category=category_name(str(text_type), int(label)),
                text_type=str(text_type),
                label=int(label),
                n=n,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                mean_ai_probability=mean_probability,
            )
        )
    return results


def robustness_table(results: list[CategoryResult]) -> pd.DataFrame:
    """Turn the category results into a table."""
    frame = pd.DataFrame([result.to_dict() for result in results])
    if not frame.empty:
        frame = frame.sort_values(["text_type", "label"]).reset_index(drop=True)
    return frame


def summarise_robustness(frame: pd.DataFrame) -> dict[str, Any]:
    """Condense the per-category table into the headline findings.

    ``degradation`` compares each non-original category with the matching
    original category of the same label: a positive value means the category
    is detected *worse* than the original text of the same authorship.
    """
    summary: dict[str, Any] = {"categories": []}
    if frame.empty:
        return summary

    for _, row in frame.iterrows():
        summary["categories"].append(
            {
                "category": row["category"],
                "n": int(row["n"]),
                "accuracy": row["accuracy"],
                "false_positive_rate": row["false_positive_rate"],
                "false_negative_rate": row["false_negative_rate"],
            }
        )

    originals = {
        int(row["label"]): row
        for _, row in frame[frame["text_type"] == "original"].iterrows()
    }
    degradations: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        if row["text_type"] == "original":
            continue
        reference = originals.get(int(row["label"]))
        if reference is None:
            continue
        degradations.append(
            {
                "category": row["category"],
                "reference_category": reference["category"],
                "accuracy_delta": round(
                    float(row["accuracy"] - reference["accuracy"]), 6
                ),
                "fpr_delta": round(
                    float(row["false_positive_rate"] - reference["false_positive_rate"]), 6
                ),
                "fnr_delta": round(
                    float(row["false_negative_rate"] - reference["false_negative_rate"]), 6
                ),
            }
        )
    summary["degradation_vs_original"] = degradations

    human_rows = frame[frame["label"] == LABEL_HUMAN]
    ai_rows = frame[frame["label"] == LABEL_AI]
    if not human_rows.empty:
        summary["worst_human_false_positive_rate"] = float(
            human_rows["false_positive_rate"].max()
        )
    if not ai_rows.empty:
        summary["worst_ai_detection_rate"] = float(
            (1.0 - ai_rows["false_negative_rate"]).min()
        )
    summary["robustness_claim_supported"] = bool(
        not human_rows.empty
        and not ai_rows.empty
        and float(human_rows["false_positive_rate"].max()) <= 0.25
        and float((1.0 - ai_rows["false_negative_rate"]).min()) >= 0.75
    )
    return summary


__all__ = [
    "CATEGORY_LABELS",
    "CategoryResult",
    "category_name",
    "evaluate_categories",
    "robustness_table",
    "summarise_robustness",
]
