"""Text fingerprinting package (Person 1).

Similarity is reported as evidence with explicit methods and matched IDs.
It is never interpreted as proof of common authorship.
"""

from fingerprinting.index import ReferenceIndex, build_index
from fingerprinting.minhash import MinHash, MinHashError
from fingerprinting.ngram import (
    NgramFingerprint,
    fingerprint,
    jaccard,
    shingles,
    similarity_evidence,
)
from fingerprinting.sha256 import (
    exact_fingerprint,
    match,
    sha256_normalized,
    sha256_text,
)
from fingerprinting.stylometry import StylometricExtractor
from fingerprinting.writeprints import WriteprintsExtractor

__all__ = [
    "MinHash",
    "MinHashError",
    "NgramFingerprint",
    "ReferenceIndex",
    "StylometricExtractor",
    "WriteprintsExtractor",
    "build_index",
    "exact_fingerprint",
    "fingerprint",
    "jaccard",
    "match",
    "sha256_normalized",
    "sha256_text",
    "shingles",
    "similarity_evidence",
]
