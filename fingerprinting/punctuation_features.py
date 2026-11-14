"""Module A(c) — punctuation features.

Counts are taken from the normalized text (punctuation is not part of the
``tokenizer.v1`` token stream). Frequencies are reported per 1,000 characters so
they are comparable across documents of different length.
"""

from __future__ import annotations

from typing import Any, Sequence

from fingerprinting.stats_utils import safe_div

# Each bucket is a set of characters counted together. Straight and curly
# variants are grouped so the feature does not depend on the source encoding.
PUNCTUATION_GROUPS: dict[str, str] = {
    "comma": ",",
    "period": ".",
    "semicolon": ";",
    "colon": ":",
    "question": "?",
    "exclamation": "!",
    "parenthesis": "()[]{}",
    "quotation": "\"\u201c\u201d\u2018\u2019\u00ab\u00bb",
    "apostrophe": "'\u2019\u02bc",
    "hyphen": "-\u2010\u2011\u2012\u2013\u2014",
    "ellipsis": "\u2026",
    "slash": "/\\",
}

# Any character that is not a letter, digit or whitespace is "punctuation" for
# the density metric. This is intentionally broader than the buckets above.
def _is_punct_char(char: str) -> bool:
    return not (char.isalnum() or char.isspace())


def extract(
    text: str,
    *,
    tokens: Sequence[Any],
    sentences: Sequence[str],
    doc: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    length = len(text)
    per_kchar = 1000.0 / length if length else None
    n_tokens = len(tokens)

    features: dict[str, float | int | None] = {}
    total_bucketed = 0
    for name, charset in PUNCTUATION_GROUPS.items():
        count = sum(1 for char in text if char in charset)
        total_bucketed += count
        features[f"punct_{name}_count"] = count
        features[f"punct_{name}_per_kchar"] = (
            None if per_kchar is None else count * per_kchar
        )

    punct_chars = sum(1 for char in text if _is_punct_char(char))
    features["punct_total_count"] = punct_chars
    features["punct_density"] = safe_div(punct_chars, length)
    features["punct_per_token"] = safe_div(punct_chars, n_tokens)
    features["punct_non_bucketed_count"] = max(0, punct_chars - total_bucketed)
    return features


__all__ = ["PUNCTUATION_GROUPS", "extract"]
