"""Exact text fingerprints (Person 1).

SHA-256 identifies an exact byte sequence. The normalized digest ignores
case and insignificant whitespace. Neither digest says anything about
authorship or semantic similarity.
"""

from __future__ import annotations

import hashlib

from shared.preprocessing.text import normalize_for_hashing, normalize_text

METHOD_VERSION = "sha256.v1"
NORMALIZED_METHOD_VERSION = "sha256_normalized_v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_normalized(text: str) -> str:
    return hashlib.sha256(normalize_for_hashing(text).encode("utf-8")).hexdigest()


def exact_fingerprint(text: str, sample_id: str | None = None) -> dict[str, object]:
    return {
        "fingerprint_type": "sha256",
        "method_version": METHOD_VERSION,
        "sample_id": sample_id,
        "sha256": sha256_text(text),
        "normalized_sha256": sha256_normalized(text),
        "characters": len(text),
        "normalized_characters": len(normalize_text(text)),
    }


def match(left: str, right: str) -> dict[str, object]:
    left_exact = sha256_text(left)
    right_exact = sha256_text(right)
    left_normalized = sha256_normalized(left)
    right_normalized = sha256_normalized(right)
    if left_exact == right_exact:
        status = "exact_match"
    elif left_normalized == right_normalized:
        status = "normalized_match"
    else:
        status = "no_exact_match"
    return {
        "fingerprint_type": "sha256",
        "match_status": status,
        "exact_match": left_exact == right_exact,
        "normalized_match": left_normalized == right_normalized,
    }
