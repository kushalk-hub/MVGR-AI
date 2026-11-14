"""Token rank under a causal language model (GLTR-style).

The rank of the realised next token is ``1 + (number of vocabulary items with a
strictly higher logit)``. Rank 1 means the token was the model's top choice.
Ranks are computed from logits (monotonic with probability) so ties are handled
deterministically: a tie counts as lower rank (not strictly greater).
"""

from __future__ import annotations

import numpy as np

from gltr.token_probability import causal_shift


def token_ranks(logits: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
    """Return the per-position rank of the realised token, shape ``[..., seq-1]``."""
    shifted_logits, targets = causal_shift(logits, input_ids)
    target_logits = np.take_along_axis(shifted_logits, targets[..., None], axis=-1)[..., 0]
    greater = np.sum(shifted_logits > target_logits[..., None], axis=-1)
    return greater + 1


def rank_statistics(
    ranks: np.ndarray, thresholds: tuple[int, ...] = (10, 100, 1000)
) -> dict[str, float | None]:
    values = np.asarray(ranks, dtype=np.float64).ravel()
    if values.size == 0:
        result: dict[str, float | None] = {
            "count": 0,
            "mean": None,
            "median": None,
            "mean_log": None,
            "max": None,
        }
        for threshold in thresholds:
            result[f"frac_le_{threshold}"] = None
        result[f"frac_gt_{max(thresholds)}"] = None
        return result

    result = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_log": float(np.log(values).mean()),
        "max": float(values.max()),
    }
    for threshold in thresholds:
        result[f"frac_le_{threshold}"] = float((values <= threshold).mean())
    result[f"frac_gt_{max(thresholds)}"] = float((values > max(thresholds)).mean())
    return result


__all__ = ["rank_statistics", "token_ranks"]
