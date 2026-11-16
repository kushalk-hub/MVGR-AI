"""Interpretability: what drives the detector's decisions.

Three views of importance are produced, because no single one is sufficient:

* **Logistic Regression coefficients** — signed, additive effects. The most
  interpretable view, but it can only express linear boundaries.
* **Random Forest / XGBoost importances** — how much a feature reduces the
  tree impurity (RF) or the loss (XGBoost). Useful, but they are *not*
  causal and are biased towards high-cardinality features.
* **SHAP values** — per-sample attributions from the tree structure, so the
  contribution of a feature to one prediction can be read directly.

Finally, the **error profile**: the mean feature value of the true positives,
true negatives, false positives and false negatives. Features that separate
the errors from the correct decisions are the ones worth investigating.

Feature importance describes what the model *used*. It says nothing about
what causes a text to be machine-generated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from detection.dataset_loader import LABEL_AI, LABEL_HUMAN


def logistic_coefficients(estimator: Any, feature_names: list[str]) -> pd.Series:
    """Signed coefficients of a fitted logistic regression."""
    coefficients = np.asarray(estimator.coef_).ravel()
    return pd.Series(coefficients, index=feature_names).sort_values(ascending=False)


def random_forest_importances(estimator: Any, feature_names: list[str]) -> pd.Series:
    """Impurity-based importances of a fitted random forest."""
    importances = np.asarray(estimator.feature_importances_).ravel()
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def xgboost_importances(estimator: Any, feature_names: list[str]) -> pd.Series:
    """Gain-based importances of a fitted XGBoost model."""
    booster = getattr(estimator, "get_booster", None)
    if booster is None:
        return pd.Series(dtype=float)
    try:
        gain = booster().get_score(importance_type="gain")
    except Exception:  # noqa: BLE001 - older/newer xgboost versions differ
        gain = booster().get_score()
    if not gain:
        return pd.Series(dtype=float)
    # When the estimator is fit on a numpy array, xgboost names the features
    # ``f0, f1, ...`` by column position; when it is fit on a DataFrame it uses
    # the column names. Support both so the importances are never all zero.
    if all(key.startswith("f") and key[1:].isdigit() for key in gain):
        mapped = {
            name: float(gain.get(f"f{index}", 0.0))
            for index, name in enumerate(feature_names)
        }
    else:
        mapped = {name: float(gain.get(name, 0.0)) for name in feature_names}
    return pd.Series(mapped).sort_values(ascending=False)


def shap_values(
    estimator: Any, X: np.ndarray, feature_names: list[str], *, max_samples: int = 500
) -> tuple[pd.Series, np.ndarray | None]:
    """Mean absolute SHAP value per feature (and the per-sample matrix).

    Uses the tree explainer, which is exact for the tree ensembles. If SHAP
    is unavailable the function returns empty results instead of a substitute
    that would be mistaken for it.
    """
    try:
        import shap
    except ImportError:
        return pd.Series(dtype=float), None
    try:
        explainer = shap.TreeExplainer(estimator)
        sample = X[:max_samples]
        values = explainer.shap_values(sample)
    except Exception:  # noqa: BLE001 - explainer unavailable for this model
        return pd.Series(dtype=float), None

    if isinstance(values, list):  # older shap returns one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:  # (samples, features, classes)
        values = values[:, :, -1]
    mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False), values


def error_profile(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    *,
    id_column: str = "sample_id",
) -> dict[str, Any]:
    """Compare the feature profile of the correct and the wrong decisions.

    Returns, for every feature, the mean value in the true-positive,
    true-negative, false-positive and false-negative groups, plus the gap
    between the error groups and their correct counterparts.
    """
    data = frame.reindex(columns=feature_names).copy()
    data["_y_true"] = np.asarray(y_true)
    data["_y_pred"] = np.asarray(y_pred)

    groups = {
        "true_positive": (data["_y_true"] == LABEL_AI) & (data["_y_pred"] == LABEL_AI),
        "true_negative": (data["_y_true"] == LABEL_HUMAN) & (data["_y_pred"] == LABEL_HUMAN),
        "false_positive": (data["_y_true"] == LABEL_HUMAN) & (data["_y_pred"] == LABEL_AI),
        "false_negative": (data["_y_true"] == LABEL_AI) & (data["_y_pred"] == LABEL_HUMAN),
    }
    means = {
        name: data.loc[mask, feature_names].mean(numeric_only=True)
        for name, mask in groups.items()
    }
    counts = {name: int(mask.sum()) for name, mask in groups.items()}

    rows = []
    for feature in feature_names:
        row: dict[str, Any] = {"feature": feature}
        for name, series in means.items():
            value = series.get(feature, np.nan)
            row[f"{name}_mean"] = (
                None if not np.isfinite(value) else round(float(value), 6)
            )
        fp = row.get("false_positive_mean")
        tn = row.get("true_negative_mean")
        fn = row.get("false_negative_mean")
        tp = row.get("true_positive_mean")
        if fp is not None and tn is not None and np.isfinite(fp) and np.isfinite(tn):
            row["fp_minus_tn"] = round(float(fp - tn), 6)
        else:
            row["fp_minus_tn"] = None
        if fn is not None and tp is not None and np.isfinite(fn) and np.isfinite(tp):
            row["fn_minus_tp"] = round(float(fn - tp), 6)
        else:
            row["fn_minus_tp"] = None
        rows.append(row)

    table = pd.DataFrame(rows)
    return {
        "group_counts": counts,
        "table": table,
        "largest_fp_shifts": table.reindex(
            table["fp_minus_tn"].abs().sort_values(ascending=False).index
        ).head(15)[["feature", "false_positive_mean", "true_negative_mean", "fp_minus_tn"]]
        if not table.empty
        else table,
        "largest_fn_shifts": table.reindex(
            table["fn_minus_tp"].abs().sort_values(ascending=False).index
        ).head(15)[["feature", "false_negative_mean", "true_positive_mean", "fn_minus_tp"]]
        if not table.empty
        else table,
    }


def family_summary(importance: pd.Series) -> dict[str, int]:
    """How many of the top features come from each family."""
    from detection.feature_alignment import feature_family_of

    counts: dict[str, int] = {}
    for column in importance.index:
        family = feature_family_of(str(column))
        counts[family] = counts.get(family, 0) + 1
    return counts


__all__ = [
    "error_profile",
    "family_summary",
    "logistic_coefficients",
    "random_forest_importances",
    "shap_values",
    "xgboost_importances",
]
