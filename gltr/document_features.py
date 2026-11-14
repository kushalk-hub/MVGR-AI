"""Document-level GLTR feature aggregation.

Combines the token-probability, token-rank and entropy signals into one flat,
stable feature mapping per document. Padding and special tokens must already be
excluded by the caller (``valid`` mask), so the statistics describe real text
tokens only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gltr.entropy_analysis import next_token_entropies
from gltr.token_probability import next_token_probabilities
from gltr.token_rank import token_ranks

DEFAULT_RANK_THRESHOLDS: tuple[int, ...] = (10, 100, 1000)


def feature_names(rank_thresholds: tuple[int, ...] = DEFAULT_RANK_THRESHOLDS) -> list[str]:
    names = [
        "gltr_token_count",
        "gltr_mean_prob",
        "gltr_median_prob",
        "gltr_std_prob",
        "gltr_min_prob",
        "gltr_max_prob",
        "gltr_mean_log_prob",
        "gltr_perplexity",
        "gltr_mean_rank",
        "gltr_median_rank",
        "gltr_mean_log_rank",
        "gltr_max_rank",
    ]
    names += [f"gltr_frac_rank_le_{threshold}" for threshold in rank_thresholds]
    names.append(f"gltr_frac_rank_gt_{max(rank_thresholds)}")
    names += ["gltr_entropy_mean", "gltr_entropy_median", "gltr_entropy_std"]
    return names


def _empty(rank_thresholds: tuple[int, ...]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"gltr_token_count": 0}
    for name in feature_names(rank_thresholds):
        result.setdefault(name, None)
    return result


def aggregate_from_arrays(
    probabilities: np.ndarray,
    ranks: np.ndarray,
    entropies: np.ndarray,
    *,
    rank_thresholds: tuple[int, ...] = DEFAULT_RANK_THRESHOLDS,
) -> dict[str, float | int | None]:
    """Build the document features from already-aligned per-token arrays."""
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    ranks = np.asarray(ranks, dtype=np.float64).ravel()
    entropies = np.asarray(entropies, dtype=np.float64).ravel()
    if probabilities.size == 0:
        return _empty(rank_thresholds)

    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    log_ranks = np.log(np.clip(ranks, 1, None))

    features: dict[str, float | int | None] = {
        "gltr_token_count": int(probabilities.size),
        "gltr_mean_prob": float(probabilities.mean()),
        "gltr_median_prob": float(np.median(probabilities)),
        "gltr_std_prob": float(probabilities.std(ddof=0)),
        "gltr_min_prob": float(probabilities.min()),
        "gltr_max_prob": float(probabilities.max()),
        "gltr_mean_log_prob": float(log_probabilities.mean()),
        "gltr_perplexity": float(np.exp(-log_probabilities.mean())),
        "gltr_mean_rank": float(ranks.mean()),
        "gltr_median_rank": float(np.median(ranks)),
        "gltr_mean_log_rank": float(log_ranks.mean()),
        "gltr_max_rank": float(ranks.max()),
        "gltr_entropy_mean": float(entropies.mean()),
        "gltr_entropy_median": float(np.median(entropies)),
        "gltr_entropy_std": float(entropies.std(ddof=0)),
    }
    for threshold in rank_thresholds:
        features[f"gltr_frac_rank_le_{threshold}"] = float((ranks <= threshold).mean())
    features[f"gltr_frac_rank_gt_{max(rank_thresholds)}"] = float(
        (ranks > max(rank_thresholds)).mean()
    )
    return features


def aggregate_document_features(
    logits: np.ndarray,
    input_ids: np.ndarray,
    *,
    rank_thresholds: tuple[int, ...] = DEFAULT_RANK_THRESHOLDS,
    valid: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    logits = np.asarray(logits, dtype=np.float64)
    input_ids = np.asarray(input_ids)
    if logits.ndim != 2 or logits.shape[0] < 2:
        return _empty(rank_thresholds)

    probabilities, _ = next_token_probabilities(logits, input_ids)
    ranks = token_ranks(logits, input_ids)
    entropies = next_token_entropies(logits, input_ids)

    if valid is not None:
        mask = np.asarray(valid, dtype=bool).ravel()
        if mask.shape[0] == probabilities.shape[0] and mask.any():
            probabilities = probabilities[mask]
            ranks = ranks[mask]
            entropies = entropies[mask]

    return aggregate_from_arrays(
        probabilities, ranks, entropies, rank_thresholds=rank_thresholds
    )


__all__ = [
    "DEFAULT_RANK_THRESHOLDS",
    "aggregate_document_features",
    "aggregate_from_arrays",
    "feature_names",
]
