"""Small numeric and text helpers for the fingerprinting feature branches.

Everything here is pure, deterministic and dependency-free apart from numpy, so
it can be unit-tested without spaCy, a language model or a database.

Null policy (matches ``features.v1``): a quantity that is undefined for a given
input is returned as ``None`` and later written as an empty CSV cell. It is
never imputed, and it is never silently replaced with zero.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

# A vowel-group heuristic for syllable counting. It is an approximation used by
# the readability formulas; it is deterministic and needs no external
# dictionary, which keeps the whole branch offline.
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]{1,2}")
_SILENT_E_RE = re.compile(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$")
_LEADING_Y_RE = re.compile(r"^y")
_NON_ALPHA_RE = re.compile(r"[^a-z]")


def safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """``numerator / denominator`` or ``None`` when the denominator is 0/None."""
    if numerator is None or denominator is None:
        return None
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator) / denominator


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def stdev(values: Sequence[float]) -> float | None:
    """Population standard deviation, ``None`` for fewer than two values."""
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=0))


def variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.var(values, ddof=0))


def entropy(counts: Iterable[float]) -> float | None:
    """Shannon entropy in nats; ``None`` when there is no mass."""
    values = [float(value) for value in counts if value and value > 0]
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        return None
    return float(-sum((value / total) * math.log(value / total) for value in values))


def normalized_entropy(counts: Iterable[float]) -> float | None:
    """Entropy divided by ``log(n_categories)``; ``None`` below two categories."""
    values = [float(value) for value in counts if value and value > 0]
    if len(values) < 2:
        return None
    raw = entropy(values)
    if raw is None:
        return None
    return raw / math.log(len(values))


def shares(counts: Mapping[str, float], *, keys: Sequence[str] | None = None) -> dict[str, float | None]:
    """Return ``{key: proportion}`` for ``keys`` (or all keys), ``None`` if empty."""
    total = sum(float(value) for value in counts.values())
    if total <= 0:
        return {key: None for key in (keys if keys is not None else counts.keys())}
    keys = list(keys) if keys is not None else list(counts.keys())
    return {key: float(counts.get(key, 0.0)) / total for key in keys}


def histogram_shares(values: Sequence[float], bins: Sequence[float]) -> list[float | None]:
    """Share of values in each ``[bins[i], bins[i+1])`` bin.

    The final bin is closed. Returns one ``None`` per bin when there are no
    values, so the caller can distinguish "no data" from "empty bin".
    """
    if not values:
        return [None] * (len(bins) - 1)
    array = np.asarray(values, dtype=float)
    edges = np.asarray(bins, dtype=float)
    counts, _ = np.histogram(array, bins=edges)
    total = counts.sum()
    if total == 0:
        return [None] * (len(edges) - 1)
    return [float(count) / float(total) for count in counts]


def counter_from(items: Iterable[str]) -> Counter[str]:
    return Counter(str(item) for item in items)


def count_syllables(word: str) -> int:
    """Deterministic English syllable estimate for one word.

    Documented approximation: vowel groups, minus a trailing silent ``e``/``es``
    or ``ed``, with a floor of one. It is not a pronunciation dictionary.
    """
    cleaned = _NON_ALPHA_RE.sub("", str(word).lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1
    cleaned = _SILENT_E_RE.sub("", cleaned)
    cleaned = _LEADING_Y_RE.sub("", cleaned)
    groups = _VOWEL_GROUP_RE.findall(cleaned)
    return max(1, len(groups))


__all__ = [
    "counter_from",
    "count_syllables",
    "entropy",
    "histogram_shares",
    "mean",
    "median",
    "normalized_entropy",
    "safe_div",
    "shares",
    "stdev",
    "variance",
]
