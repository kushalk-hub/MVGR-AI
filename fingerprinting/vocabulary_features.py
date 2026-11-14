"""Module A(a) — vocabulary features.

Pure lexical-diversity statistics computed from the shared ``tokenizer.v1``
token stream, so they stay consistent with the graph views. Undefined ratios for
empty text are emitted as ``None`` (never imputed).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from shared.preprocessing.text import Token

from fingerprinting.stats_utils import (
    counter_from,
    histogram_shares,
    mean,
    safe_div,
    stdev,
)

DEFAULT_WORD_LENGTH_BINS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
DEFAULT_MSTTR_SEGMENT = 50


def _msttr(surfaces: Sequence[str], segment: int) -> float | None:
    """Mean segmental type-token ratio over fixed-size segments."""
    if segment <= 0 or len(surfaces) < segment:
        return None
    segment_ttrs = [
        len(set(surfaces[index : index + segment])) / segment
        for index in range(0, len(surfaces) - segment + 1, segment)
    ]
    return mean(segment_ttrs)


def extract(
    text: str,
    *,
    tokens: Sequence[Token],
    sentences: Sequence[str],
    doc: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    config = config or {}
    bins = tuple(config.get("word_length_bins", DEFAULT_WORD_LENGTH_BINS))
    segment = int(config.get("msttr_segment_words", DEFAULT_MSTTR_SEGMENT))

    surfaces = [token.text for token in tokens]
    total = len(surfaces)
    counts = counter_from(surfaces)
    types = len(counts)
    hapax = sum(1 for value in counts.values() if value == 1)
    word_lengths = [len(surface) for surface in surfaces]

    features: dict[str, float | int | None] = {
        "vocab_total_words": total,
        "vocab_unique_words": types,
        "vocab_type_token_ratio": safe_div(types, total),
        "vocab_root_type_token_ratio": (
            None if total == 0 else types / (total ** 0.5)
        ),
        "vocab_hapax_count": hapax,
        "vocab_hapax_ratio": safe_div(hapax, total),
        "vocab_richness_herdan": (
            None if total < 2 or types < 2 else math.log(types) / math.log(total)
        ),
        "vocab_avg_word_length": mean(word_lengths),
        "vocab_word_length_std": stdev(word_lengths),
        "vocab_stopword_ratio": safe_div(
            sum(1 for token in tokens if token.is_stopword), total
        ),
        "vocab_lexical_diversity_msttr": _msttr(surfaces, segment),
        "vocab_unique_word_ratio": safe_div(types, total),
    }

    histogram = histogram_shares(word_lengths, bins)
    for index, share in enumerate(histogram):
        features[f"vocab_word_length_hist_{index}"] = share

    # A compact, always-defined length summary for downstream fusion.
    features["vocab_max_word_length"] = max(word_lengths) if word_lengths else None
    features["vocab_min_word_length"] = min(word_lengths) if word_lengths else None
    return features


__all__ = ["DEFAULT_MSTTR_SEGMENT", "DEFAULT_WORD_LENGTH_BINS", "extract"]
