"""MinHash fingerprints (Person 1).

MinHash estimates Jaccard similarity between shingle sets; banded LSH makes
candidate retrieval sub-quadratic. It is a similarity method, not an authorship
classifier. A high score means the two texts share n-grams, nothing more.

Implementation notes
--------------------
* Shingles are hashed with ``zlib.crc32`` (deterministic, fast, 32-bit).
* Permutations use a fixed universal hash family over the Mersenne prime
  2^31 - 1, seeded for reproducibility.
* An empty text has no shingles and returns the sentinel signature ``-1``;
  it never matches anything, including another empty text.
"""

from __future__ import annotations

import zlib
from typing import Iterable

import numpy as np

METHOD_VERSION = "minhash.v1"
MERSENNE_PRIME = 2_147_483_647


class MinHashError(ValueError):
    pass


def _shingle_hashes(text: str, size: int) -> np.ndarray:
    from shared.preprocessing.text import normalize_text, tokenize

    tokens = [token.text for token in tokenize(normalize_text(text))]
    if len(tokens) < size:
        return np.empty(0, dtype=np.int64)
    hashes = np.empty(len(tokens) - size + 1, dtype=np.int64)
    for index in range(len(tokens) - size + 1):
        shingle = "\x1f".join(tokens[index : index + size])
        hashes[index] = zlib.crc32(shingle.encode("utf-8")) & 0xFFFFFFFF
    return hashes


class MinHash:
    def __init__(
        self,
        *,
        permutations: int = 128,
        seed: int = 42,
        shingle_size: int = 5,
        rows_per_band: int = 8,
    ) -> None:
        if permutations <= 0:
            raise MinHashError("permutations must be positive")
        if rows_per_band <= 0 or permutations % rows_per_band != 0:
            raise MinHashError("rows_per_band must divide permutations")
        self.permutations = permutations
        self.seed = seed
        self.shingle_size = shingle_size
        self.rows_per_band = rows_per_band
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, MERSENNE_PRIME, size=permutations, dtype=np.int64)
        self._b = rng.integers(0, MERSENNE_PRIME, size=permutations, dtype=np.int64)

    def signature_from_hashes(self, hashes: np.ndarray) -> np.ndarray:
        if hashes.size == 0:
            return np.full(self.permutations, -1, dtype=np.int64)
        combined = (
            self._a[:, None] * hashes[None, :].astype(np.int64) + self._b[:, None]
        ) % MERSENNE_PRIME
        return combined.min(axis=1).astype(np.int64)

    def signature(self, text: str) -> np.ndarray:
        return self.signature_from_hashes(_shingle_hashes(text, self.shingle_size))

    def similarity_from_signature(self, left: np.ndarray, right: np.ndarray) -> float:
        if left.size != right.size:
            raise MinHashError("signatures must have equal length")
        if (left < 0).any() or (right < 0).any():
            return 0.0
        return float(np.mean(left == right))

    def similarity(self, left: str, right: str) -> float:
        return self.similarity_from_signature(self.signature(left), self.signature(right))

    def bands(self, signature: np.ndarray) -> list[tuple[int, ...]]:
        if signature.size != self.permutations:
            raise MinHashError("signature length does not match permutation count")
        reshaped = signature.reshape(-1, self.rows_per_band)
        return [tuple(int(value) for value in row) for row in reshaped]

    @property
    def band_count(self) -> int:
        return self.permutations // self.rows_per_band

    def candidate_buckets(self, signatures: Iterable[np.ndarray]) -> dict[tuple[int, tuple[int, ...]], list[int]]:
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        for position, signature in enumerate(signatures):
            for band_index, key in enumerate(self.bands(signature)):
                buckets.setdefault((band_index, key), []).append(position)
        return buckets
