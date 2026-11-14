"""Token probability under a causal language model (GLTR-style).

The causal shift is explicit: logits at position ``i`` predict the token at
position ``i + 1``. Probabilities are the softmax of the shifted logits gathered
at the true next token. Pure NumPy so the math is unit-testable without torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def causal_shift(logits: np.ndarray, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift so position ``i``'s logits are aligned with token ``i + 1``."""
    return logits[..., :-1, :], input_ids[..., 1:]


def next_token_probabilities(
    logits: np.ndarray, input_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(target_probabilities, targets)`` for the next-token prediction.

    ``logits`` is ``[..., seq, vocab]`` and ``input_ids`` is ``[..., seq]``. The
    returned arrays have shape ``[..., seq - 1]``: one probability per predicted
    position.
    """
    shifted_logits, targets = causal_shift(logits, input_ids)
    probabilities = softmax(shifted_logits, axis=-1)
    target_probabilities = np.take_along_axis(
        probabilities, targets[..., None], axis=-1
    )[..., 0]
    return target_probabilities, targets


def probability_statistics(probabilities: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(probabilities, dtype=np.float64).ravel()
    if values.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


__all__ = [
    "causal_shift",
    "next_token_probabilities",
    "probability_statistics",
    "softmax",
]
