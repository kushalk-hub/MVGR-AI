"""Module A(b) — sentence features.

Sentence boundaries come from ``shared.preprocessing.text.split_sentences``
(``tokenizer.v1``), so they are identical to the ones used by the graph views.
Sentence length is measured in word tokens; a punctuation-only fragment carries
no word tokens and is counted as length 0 rather than dropped silently.
"""

from __future__ import annotations

from typing import Any, Sequence

from shared.preprocessing.text import Token

from fingerprinting.stats_utils import (
    histogram_shares,
    mean,
    median,
    safe_div,
    stdev,
    variance,
)

DEFAULT_SENTENCE_BINS = (1, 5, 10, 15, 20, 25, 30, 40, 50, 75)
DEFAULT_LONG_WORDS = 25
DEFAULT_SHORT_WORDS = 8


def _sentence_lengths(tokens: Sequence[Token], sentence_count: int) -> list[int]:
    lengths = [0] * max(sentence_count, 0)
    for token in tokens:
        index = token.sentence_index
        if 0 <= index < len(lengths):
            lengths[index] += 1
        else:  # defensive: extend if a token carries an out-of-range index
            lengths.extend([0] * (index - len(lengths) + 1))
            lengths[index] += 1
    return lengths


def extract(
    text: str,
    *,
    tokens: Sequence[Token],
    sentences: Sequence[str],
    doc: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    config = config or {}
    bins = tuple(config.get("sentence_length_bins", DEFAULT_SENTENCE_BINS))
    long_words = int(config.get("long_sentence_words", DEFAULT_LONG_WORDS))
    short_words = int(config.get("short_sentence_words", DEFAULT_SHORT_WORDS))

    lengths = _sentence_lengths(tokens, len(sentences))
    count = len(sentences)

    features: dict[str, float | int | None] = {
        "sent_count": count,
        "sent_avg_length": mean(lengths),
        "sent_median_length": median(lengths),
        "sent_max_length": max(lengths) if lengths else None,
        "sent_min_length": min(lengths) if lengths else None,
        "sent_length_variance": variance(lengths),
        "sent_length_std": stdev(lengths),
        "sent_length_range": (max(lengths) - min(lengths)) if lengths else None,
        "sent_long_ratio": safe_div(
            sum(1 for value in lengths if value > long_words), count
        ),
        "sent_short_ratio": safe_div(
            sum(1 for value in lengths if value < short_words), count
        ),
    }

    histogram = histogram_shares(lengths, bins)
    for index, share in enumerate(histogram):
        features[f"sent_length_hist_{index}"] = share
    return features


__all__ = [
    "DEFAULT_LONG_WORDS",
    "DEFAULT_SENTENCE_BINS",
    "DEFAULT_SHORT_WORDS",
    "extract",
]
