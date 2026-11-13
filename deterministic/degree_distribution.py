"""Degree-distribution features.

All statistics are computed on the undirected simple projection of the graph.
Undefined values are returned as ``None``.
"""

from __future__ import annotations

import numpy as np

DEGREE_FEATURES = (
    "degree_mean",
    "degree_std",
    "degree_min",
    "degree_max",
    "degree_median",
    "degree_q25",
    "degree_q75",
    "degree_cv",
    "degree_gini",
    "degree_entropy",
    "degree_max_ratio",
)


def gini(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    total = float(values.sum())
    if total <= 0:
        return None
    sorted_values = np.sort(values.astype(float))
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_values)) / (n * total) - (n + 1.0) / n)


def entropy(values: np.ndarray, *, normalize: bool = False) -> float | None:
    if values.size == 0:
        return None
    counts = np.asarray(values, dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    probabilities = counts[counts > 0] / total
    result = float(-np.sum(probabilities * np.log(probabilities)))
    if normalize:
        if values.size < 2:
            return None
        return result / float(np.log(values.size))
    return result


def degree_features(degrees: np.ndarray) -> dict[str, float | None]:
    if degrees.size == 0:
        return {name: None for name in DEGREE_FEATURES}
    mean = float(degrees.mean())
    std = float(degrees.std())
    total = float(degrees.sum())
    return {
        "degree_mean": mean,
        "degree_std": std,
        "degree_min": float(degrees.min()),
        "degree_max": float(degrees.max()),
        "degree_median": float(np.median(degrees)),
        "degree_q25": float(np.quantile(degrees, 0.25)),
        "degree_q75": float(np.quantile(degrees, 0.75)),
        "degree_cv": float(std / mean) if mean > 0 else None,
        "degree_gini": gini(degrees),
        "degree_entropy": entropy(degrees, normalize=True),
        "degree_max_ratio": float(degrees.max() / total) if total > 0 else None,
    }


__all__ = ["DEGREE_FEATURES", "degree_features", "entropy", "gini"]
