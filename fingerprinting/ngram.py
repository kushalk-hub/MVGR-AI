"""N-gram (shingle) fingerprints (Person 1).

Shingles are contiguous token sequences. Jaccard similarity measures overlap;
it is evidence of textual reuse, not proof of common authorship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from shared.preprocessing.text import normalize_text, tokenize

METHOD_VERSION = "ngram.v1"


@dataclass(frozen=True)
class NgramFingerprint:
    size: int
    shingles: frozenset[tuple[str, ...]]

    def jaccard(self, other: "NgramFingerprint") -> float:
        if not self.shingles or not other.shingles:
            return 0.0
        union = self.shingles | other.shingles
        return len(self.shingles & other.shingles) / len(union)

    def containment(self, other: "NgramFingerprint") -> float:
        """Share of this fingerprint's shingles present in ``other``."""
        if not self.shingles:
            return 0.0
        return len(self.shingles & other.shingles) / len(self.shingles)


def shingles(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("shingle size must be positive")
    tokens = [token.text for token in tokenize(normalize_text(text))]
    if len(tokens) < size:
        return frozenset()
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def fingerprint(text: str, *, size: int = 5) -> NgramFingerprint:
    return NgramFingerprint(size=size, shingles=shingles(text, size))


def jaccard(left: str | NgramFingerprint, right: str | NgramFingerprint, *, size: int = 5) -> float:
    left_fp = left if isinstance(left, NgramFingerprint) else fingerprint(left, size=size)
    right_fp = right if isinstance(right, NgramFingerprint) else fingerprint(right, size=size)
    return left_fp.jaccard(right_fp)


def similarity_evidence(
    left: str,
    right: str,
    *,
    left_id: str | None = None,
    right_id: str | None = None,
    size: int = 5,
) -> dict[str, object]:
    left_fp = fingerprint(left, size=size)
    right_fp = fingerprint(right, size=size)
    score = left_fp.jaccard(right_fp)
    return {
        "fingerprint_type": "ngram",
        "method_version": METHOD_VERSION,
        "shingle_size": size,
        "left_id": left_id,
        "right_id": right_id,
        "similarity": score,
        "containment_left_in_right": left_fp.containment(right_fp),
        "containment_right_in_left": right_fp.containment(left_fp),
        "interpretation": "textual_overlap_not_authorship_proof",
    }
