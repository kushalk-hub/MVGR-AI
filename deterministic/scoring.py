"""Deterministic scoring rule with validation-selected features and thresholds.

Stage A is feature extraction. Stage B is a transparent scoring rule:

1. per-feature direction and discriminative strength are estimated on the
   validation split only;
2. at most ``top_k`` features are kept;
3. a single scalar score is the sigmoid of the mean signed z-score;
4. the decision threshold is chosen on validation by balanced accuracy;
5. the resulting rule is frozen with a content hash before test evaluation.

Caveat: selection and thresholding both use the validation split. That makes
the validation estimate mildly optimistic. It never uses the test split, and
the frozen artifact records this limitation explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from shared.contracts.validators import utc_now

DEFAULT_TOP_K = 12
DEFAULT_MIN_AUC = 0.55
CLIP = 3.0


class ScoringError(ValueError):
    pass


def _clean_matrix(values: pd.DataFrame) -> pd.DataFrame:
    return values.replace([np.inf, -np.inf], np.nan)


def select_features(
    features: pd.DataFrame,
    labels: Sequence[int],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_auc: float = DEFAULT_MIN_AUC,
) -> list[dict[str, Any]]:
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        raise ScoringError("validation split must contain both classes")
    selected: list[dict[str, Any]] = []
    clean = _clean_matrix(features)
    for column in clean.columns:
        values = clean[column].to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
            continue
        auc = float(roc_auc_score(y[mask], values[mask]))
        strength = abs(auc - 0.5)
        if strength < abs(min_auc - 0.5):
            continue
        selected.append({"name": column, "auc": auc, "direction": 1.0 if auc >= 0.5 else -1.0})
    selected.sort(key=lambda item: (-abs(item["auc"] - 0.5), item["name"]))
    return selected[:top_k]


def _standardize_block(
    matrix: np.ndarray, means: np.ndarray, stds: np.ndarray, medians: np.ndarray
) -> np.ndarray:
    filled = np.where(np.isfinite(matrix), matrix, medians)
    scaled = (filled - means) / stds
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.clip(scaled, -CLIP, CLIP)


def score_matrix(matrix: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    if matrix.shape[1] != len(rule["features"]):
        raise ScoringError(
            f"score matrix has {matrix.shape[1]} columns, rule expects {len(rule['features'])}"
        )
    means = np.asarray([feature["mean"] for feature in rule["features"]], dtype=float)
    stds = np.asarray([feature["std"] for feature in rule["features"]], dtype=float)
    medians = np.asarray([feature["median"] for feature in rule["features"]], dtype=float)
    directions = np.asarray([feature["direction"] for feature in rule["features"]], dtype=float)
    scaled = _standardize_block(matrix, means, stds, medians) * directions
    aggregate = scaled.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-aggregate))


def fit_scoring_rule(
    train_features: pd.DataFrame,
    train_labels: Sequence[int],
    validation_features: pd.DataFrame,
    validation_labels: Sequence[int],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_auc: float = DEFAULT_MIN_AUC,
    config_id: str = "cfg_unknown",
) -> dict[str, Any]:
    """Fit and freeze a scoring rule. Persist with :func:`save_rule`."""
    if list(train_features.columns) != list(validation_features.columns):
        raise ScoringError("train and validation feature columns differ")
    selected = select_features(validation_features, validation_labels, top_k=top_k, min_auc=min_auc)
    if not selected:
        raise ScoringError("no feature passed the validation selection threshold")

    clean_train = _clean_matrix(train_features)
    features: list[dict[str, Any]] = []
    for item in selected:
        values = clean_train[item["name"]].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ScoringError(f"feature {item['name']} has no finite training values")
        features.append(
            {
                "name": item["name"],
                "auc": item["auc"],
                "direction": item["direction"],
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)) if float(np.std(finite)) > 0 else 1.0,
                "median": float(np.median(finite)),
            }
        )

    rule: dict[str, Any] = {
        "feature_set_version": str(train_features.attrs.get("feature_set_version", "features.v1")),
        "features": features,
        "selection_split": "validation",
        "threshold_split": "validation",
        "validation_optimism": True,
        "created_at": utc_now(),
        "config_id": config_id,
        "threshold": 0.5,
        "clip": CLIP,
        "top_k": top_k,
        "min_auc": min_auc,
    }

    columns = [feature["name"] for feature in features]
    validation_scores = score_matrix(validation_features[columns].to_numpy(dtype=float), rule)
    candidates = np.unique(
        np.concatenate(
            [
                np.quantile(validation_scores, np.linspace(0.05, 0.95, 91)),
                np.asarray([0.5]),
            ]
        )
    )
    y = np.asarray(validation_labels, dtype=int)
    best_threshold, best_balanced, best_recall = 0.5, -1.0, -1.0
    for threshold in candidates:
        predicted = (validation_scores >= threshold).astype(int)
        balanced = float(balanced_accuracy_score(y, predicted))
        recall = float(((predicted == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        if balanced > best_balanced + 1e-12 or (
            abs(balanced - best_balanced) <= 1e-12 and recall > best_recall
        ):
            best_threshold, best_balanced, best_recall = float(threshold), balanced, recall
    rule["threshold"] = best_threshold
    rule["validation_metrics"] = {
        "balanced_accuracy": best_balanced,
        "recall_at_threshold": best_recall,
        "auc_of_score": float(roc_auc_score(y, validation_scores))
        if len(np.unique(y)) > 1
        else None,
    }
    rule["rule_hash"] = hashlib.sha256(
        json.dumps(
            {key: rule[key] for key in ("features", "threshold", "clip", "feature_set_version")},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return rule


def apply_rule(
    features: pd.DataFrame, rule: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    columns = [feature["name"] for feature in rule["features"]]
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ScoringError(f"features missing for scoring rule: {missing}")
    scores = score_matrix(_clean_matrix(features[columns]).to_numpy(dtype=float), rule)
    predicted = (scores >= float(rule["threshold"])).astype(int)
    return scores, predicted


def save_rule(rule: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_rule(path: str | Path) -> dict[str, Any]:
    rule = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = rule.get("rule_hash")
    actual = hashlib.sha256(
        json.dumps(
            {key: rule[key] for key in ("features", "threshold", "clip", "feature_set_version")},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if expected != actual:
        raise ScoringError("frozen scoring rule hash does not match its content")
    return rule


def require_frozen_rule(path: str | Path) -> dict[str, Any]:
    rule = load_rule(path)
    if not rule.get("rule_hash"):
        raise ScoringError("scoring rule is not frozen")
    return rule
