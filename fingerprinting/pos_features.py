"""Module A(d) — part-of-speech features.

POS tags come from the local spaCy model (``en_core_web_sm``), the same model
used by the syntactic graph view. Ratios are computed over non-whitespace
tokens. When no spaCy document is supplied (model unavailable), every feature is
``None`` rather than zero.
"""

from __future__ import annotations

from typing import Any, Sequence

from fingerprinting.stats_utils import (
    counter_from,
    entropy,
    normalized_entropy,
    safe_div,
)

# Universal POS tag inventory (spaCy ``token.pos_``).
UPOS_TAGS: tuple[str, ...] = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)

# Named ratio features map a friendly name to the tags they aggregate.
RATIO_TAGS: dict[str, tuple[str, ...]] = {
    "noun": ("NOUN",),
    "verb": ("VERB", "AUX"),
    "adjective": ("ADJ",),
    "adverb": ("ADV",),
    "pronoun": ("PRON",),
    "determiner": ("DET",),
    "preposition": ("ADP",),
    "conjunction": ("CCONJ", "SCONJ"),
    "proper_noun": ("PROPN",),
    "numeral": ("NUM",),
    "particle": ("PART",),
    "punctuation": ("PUNCT",),
}


def _pos_counts(doc: Any) -> list[str]:
    return [token.pos_ for token in doc if not token.is_space]


def extract(
    text: str,
    *,
    tokens: Sequence[Any],
    sentences: Sequence[str],
    doc: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    if doc is None:
        keys = [f"pos_{name}_ratio" for name in RATIO_TAGS]
        keys += [f"pos_share_{tag.lower()}" for tag in UPOS_TAGS]
        keys += ["pos_token_count", "pos_entropy", "pos_normalized_entropy"]
        return {key: None for key in keys}

    tags = _pos_counts(doc)
    total = len(tags)
    counts = counter_from(tags)

    features: dict[str, float | int | None] = {"pos_token_count": total}
    for name, group in RATIO_TAGS.items():
        features[f"pos_{name}_ratio"] = safe_div(
            sum(counts.get(tag, 0) for tag in group), total
        )
    for tag in UPOS_TAGS:
        features[f"pos_share_{tag.lower()}"] = safe_div(counts.get(tag, 0), total)

    features["pos_entropy"] = entropy(counts.values())
    features["pos_normalized_entropy"] = normalized_entropy(counts.values())
    return features


__all__ = ["RATIO_TAGS", "UPOS_TAGS", "extract"]
