"""Next-token entropy under a causal language model (GLTR-style).

Entropy is measured over the shifted distribution, i.e. the entropy of the
prediction for each realised next token. High entropy means the model was
uncertain at that position.
"""

from __future__ import annotations

import numpy as np

from gltr.token_probability import causal_shift, softmax


def next_token_entropies(logits: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
    shifted_logits, _ = causal_shift(logits, input_ids)
    probabilities = softmax(shifted_logits, axis=-1)
    return -np.sum(probabilities * np.log(probabilities + 1e-12), axis=-1)


def entropy_statistics(entropies: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(entropies, dtype=np.float64).ravel()
    if values.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


__all__ = ["entropy_statistics", "next_token_entropies"]
