"""Evaluate the trained detectors and draw the comparison plots.

All metrics come from the **untouched test split**. Every model is scored with
the same metric set so the comparison is apples-to-apples:

accuracy, precision, recall, F1, ROC-AUC, PR-AUC, the confusion matrix,
false-positive rate, false-negative rate, specificity and balanced accuracy.

Because the dataset is imbalanced (roughly 78% AI), balanced accuracy and the
false-positive rate are the numbers that matter most: a detector that simply
answers "AI" everywhere scores 0.785 accuracy while being useless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from detection.dataset_loader import LABEL_AI, LABEL_HUMAN

#: Matplotlib must not open a window on a headless Windows box.
_MPL_BACKEND = "Agg"


def _plt() -> Any:
    import matplotlib

    matplotlib.use(_MPL_BACKEND)
    import matplotlib.pyplot as plt

    return plt


@dataclass
class EvaluationResult:
    """One model's metrics on one evaluation split."""

    model: str
    label: str
    split: str
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    support: dict[str, int]
    y_true: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    y_proba: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "label": self.label,
            "split": self.split,
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "confusion_matrix": self.confusion_matrix,
            "support": self.support,
        }


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Compute the full metric set for one model."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    matrix = confusion_matrix(y_true, y_pred, labels=[LABEL_HUMAN, LABEL_AI])
    tn, fp, fn, tp = matrix.ravel()
    total = tn + fp + fn + tp
    metrics["true_negative"] = float(tn)
    metrics["false_positive"] = float(fp)
    metrics["false_negative"] = float(fn)
    metrics["true_positive"] = float(tp)
    metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
    metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    metrics["precision_positive"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
    metrics["n_samples"] = float(total)

    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_proba)))
        from sklearn.metrics import average_precision_score

        metrics["pr_auc"] = float(average_precision_score(y_true, np.asarray(y_proba)))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def evaluate_model(
    model_name: str,
    label: str,
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    split: str = "test",
) -> EvaluationResult:
    """Predict with ``estimator`` and score it against ``y``."""
    from detection.train_models import positive_probability

    y_pred = np.asarray(estimator.predict(X)).astype(int)
    try:
        y_proba = positive_probability(estimator, X)
    except ValueError:
        y_proba = None
    metrics = compute_metrics(y, y_pred, y_proba)
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y, y_pred, labels=[LABEL_HUMAN, LABEL_AI])
    support = {
        "human": int((y == LABEL_HUMAN).sum()),
        "ai": int((y == LABEL_AI).sum()),
    }
    return EvaluationResult(
        model=model_name,
        label=label,
        split=split,
        metrics=metrics,
        confusion_matrix=matrix.tolist(),
        support=support,
        y_true=np.asarray(y).astype(int),
        y_pred=y_pred,
        y_proba=np.asarray(y_proba) if y_proba is not None else np.array([]),
    )


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(
    results: list[EvaluationResult], path: str | Path
) -> str:
    """One confusion matrix per model, drawn side by side."""
    plt = _plt()
    count = len(results)
    fig, axes = plt.subplots(1, count, figsize=(4.2 * count, 4.0), squeeze=False)
    for axis, result in zip(axes[0], results, strict=True):
        matrix = np.asarray(result.confusion_matrix)
        axis.imshow(matrix, cmap="Blues")
        threshold = matrix.max() / 2.0 if matrix.max() else 0.5
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        axis.set_xticks([0, 1], ["Human", "AI"])
        axis.set_yticks([0, 1], ["Human", "AI"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{result.label}\nacc {result.metrics['accuracy']:.3f} "
            f"| bal.acc {result.metrics['balanced_accuracy']:.3f}"
        )
    fig.suptitle("Confusion matrices (test split)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_roc_curves(results: list[EvaluationResult], path: str | Path) -> str:
    """ROC curves for every model on the test split."""
    from sklearn.metrics import roc_curve

    plt = _plt()
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    for result in results:
        if result.y_proba.size == 0:
            continue
        fpr, tpr, _ = roc_curve(result.y_true, result.y_proba)
        axis.plot(
            fpr,
            tpr,
            label=f"{result.label} (AUC {result.metrics['roc_auc']:.3f})",
        )
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves (test split)")
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_precision_recall_curves(
    results: list[EvaluationResult], path: str | Path
) -> str:
    """Precision-recall curves for every model on the test split."""
    from sklearn.metrics import precision_recall_curve

    plt = _plt()
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    for result in results:
        if result.y_proba.size == 0:
            continue
        precision, recall, _ = precision_recall_curve(result.y_true, result.y_proba)
        axis.plot(
            recall,
            precision,
            label=f"{result.label} (AP {result.metrics['pr_auc']:.3f})",
        )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves (test split)")
    axis.legend(loc="lower left", fontsize=9)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_model_comparison(results: list[EvaluationResult], path: str | Path) -> str:
    """Grouped bar chart of the headline metrics."""
    plt = _plt()
    names = [result.label for result in results]
    keys = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]
    titles = ["Accuracy", "Balanced accuracy", "F1", "ROC-AUC", "PR-AUC"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.2), squeeze=False)
    for axis, key, title in zip(axes[0], keys, titles, strict=True):
        values = [result.metrics.get(key, float("nan")) for result in results]
        bars = axis.bar(names, values, color="#4c72b0")
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", rotation=30, labelsize=8)
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=7,
                )
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle("Model comparison (test split)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_feature_importance(
    importances: dict[str, pd.Series], path: str | Path, top_n: int = 25
) -> str:
    """Horizontal bar charts of the most influential features per model."""
    plt = _plt()
    count = len(importances)
    fig, axes = plt.subplots(1, count, figsize=(6.0 * count, 6.4), squeeze=False)
    for axis, (name, series) in zip(axes[0], importances.items(), strict=True):
        ordered = series.sort_values(ascending=True).tail(top_n)
        axis.barh(ordered.index, ordered.values, color="#55a868")
        axis.set_title(f"{name} (top {len(ordered)})")
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(axis="x", alpha=0.3)
    fig.suptitle("Feature importance (trained on the training split)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def fit_and_score(
    spec: Any,
    params: dict[str, Any] | None,
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    columns: list[str],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    *,
    split: str = "test",
    scale: bool = True,
    calibration_frame: pd.DataFrame | None = None,
    calibration_labels: np.ndarray | None = None,
) -> tuple[Any, Any, EvaluationResult]:
    """Fit the preprocessing on train, train the model on train, score it.

    The preprocessing is fitted on the training rows only and then applied
    unchanged to the evaluation rows, so no evaluation value can influence the
    model. A margin classifier is calibrated on ``calibration_frame`` (the
    validation split); when none is given the calibrator is fitted in-sample
    on the training rows, which affects the probability scale but not the
    ranking.
    """
    from detection.preprocessing import fit_preprocessor
    from detection.train_models import fit_final

    preprocessor = fit_preprocessor(train_frame, columns, scale=scale)
    X_train = preprocessor.transform(train_frame)
    X_eval = preprocessor.transform(eval_frame)
    estimator = fit_final(spec, X_train, np.asarray(y_train), params)

    if getattr(spec, "probability_strategy", "native") == "margin_calibrated":
        if calibration_frame is not None and calibration_labels is not None:
            X_calibration = preprocessor.transform(calibration_frame)
            estimator.fit_calibrator(X_calibration, np.asarray(calibration_labels))
        else:
            estimator.fit_calibrator(X_train, np.asarray(y_train))

    result = evaluate_model(
        spec.name, spec.label, estimator, X_eval, np.asarray(y_eval), split=split
    )
    return estimator, preprocessor, result


def results_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    """Flatten the metric sets into one table."""
    rows = []
    for result in results:
        row = {"model": result.model, "label": result.label, "split": result.split}
        row.update(result.metrics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "roc_auc" in frame.columns:
        frame = frame.sort_values("roc_auc", ascending=False, na_position="last")
    return frame.reset_index(drop=True)


def write_results(
    results: list[EvaluationResult], directory: str | Path
) -> dict[str, str]:
    """Write the metric table as CSV and JSON."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    frame = results_frame(results)
    csv_path = target / "model_comparison.csv"
    frame.to_csv(csv_path, index=False)
    json_path = target / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump([result.to_dict() for result in results], handle, indent=2)
        handle.write("\n")
    return {"csv": str(csv_path), "json": str(json_path)}


__all__ = [
    "EvaluationResult",
    "compute_metrics",
    "evaluate_model",
    "plot_confusion_matrices",
    "plot_feature_importance",
    "plot_model_comparison",
    "plot_precision_recall_curves",
    "plot_roc_curves",
    "results_frame",
    "write_results",
]
