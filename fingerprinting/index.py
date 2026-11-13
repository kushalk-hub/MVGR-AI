"""Reference fingerprint index (Person 1).

The index is deliberately restricted to an explicit list of reference sample
IDs. In particular, a reference index used by the online pipeline must be built
from training data only; indexing the test split would turn similarity into a
label leak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from fingerprinting.minhash import MinHash, MinHashError
from fingerprinting.ngram import jaccard as ngram_jaccard
from shared.contracts.validators import utc_now
from shared.preprocessing.text import normalize_for_hashing, sha256_text


class IndexError_(ValueError):
    pass


@dataclass
class ReferenceIndex:
    sample_ids: list[str]
    signatures: np.ndarray
    normalized_hashes: list[str]
    minhash: MinHash
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sample_ids=np.asarray(self.sample_ids, dtype=object),
            signatures=np.asarray(self.signatures, dtype=np.int64),
            normalized_hashes=np.asarray(self.normalized_hashes, dtype=object),
            metadata=np.asarray([json.dumps(self.metadata)], dtype=object),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceIndex":
        with np.load(Path(path), allow_pickle=True) as payload:
            sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            signatures = np.asarray(payload["signatures"], dtype=np.int64)
            hashes = [str(value) for value in payload["normalized_hashes"].tolist()]
            metadata = json.loads(str(payload["metadata"][0]))
        minhash = MinHash(
            permutations=int(metadata["permutations"]),
            seed=int(metadata["seed"]),
            shingle_size=int(metadata["shingle_size"]),
            rows_per_band=int(metadata["rows_per_band"]),
        )
        return cls(sample_ids, signatures, hashes, minhash, metadata)

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        threshold: float = 0.5,
        exclude_ids: Iterable[str] = (),
        exact_text_lookup: Callable[[str], str | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve similar references.

        ``exact_text_lookup`` optionally maps a reference ID to its raw text so
        the candidate can be re-ranked with exact shingle Jaccard.
        """
        if not self.sample_ids:
            return []
        excluded = set(exclude_ids)
        signature = self.minhash.signature(text)
        if (signature < 0).any():
            return []
        normalized = sha256_text(normalize_for_hashing(text))

        candidates: set[int] = set()
        if getattr(self, "_buckets", None) is None:
            self._buckets = self.minhash.candidate_buckets(self.signatures)
        buckets = self._buckets
        for band_index, key in enumerate(self.minhash.bands(signature)):
            candidates.update(buckets.get((band_index, key), []))
        # Fall back to a full scan for very small indexes, where LSH bands are
        # sparse and would lose valid candidates.
        if len(self.sample_ids) <= max(64, self.minhash.band_count):
            candidates = set(range(len(self.sample_ids)))

        results: list[dict[str, Any]] = []
        normalized_matches = 0
        for position in candidates:
            reference_id = self.sample_ids[position]
            if reference_id in excluded:
                continue
            if self.normalized_hashes[position] == normalized:
                normalized_matches += 1
            estimate = self.minhash.similarity_from_signature(signature, self.signatures[position])
            if estimate < threshold:
                continue
            entry: dict[str, Any] = {
                "reference_id": reference_id,
                "method": "minhash",
                "score": estimate,
                "minhash_score": estimate,
                "normalized_exact": self.normalized_hashes[position] == normalized,
            }
            if exact_text_lookup is not None:
                reference_text = exact_text_lookup(reference_id)
                if reference_text is not None:
                    exact = ngram_jaccard(text, reference_text, size=self.minhash.shingle_size)
                    entry["ngram_jaccard"] = exact
                    if exact > entry["score"]:
                        entry["score"] = exact
                        entry["method"] = "shingle_jaccard"
            results.append(entry)

        results.sort(key=lambda item: (-float(item["score"]), str(item["reference_id"])))
        return results[:top_k]


def build_index(
    *,
    frame: pd.DataFrame,
    reference_ids: Iterable[str],
    permutations: int = 128,
    seed: int = 42,
    shingle_size: int = 5,
    rows_per_band: int = 8,
    source_description: str = "",
) -> ReferenceIndex:
    """Build an index over exactly ``reference_ids``.

    The caller is responsible for proving that those IDs belong to the training
    split. The index records the restriction in its metadata for later audit.
    """
    allowed = {str(value) for value in reference_ids}
    subset = frame[frame["sample_id"].astype(str).isin(allowed)].copy()
    missing = allowed - set(subset["sample_id"].astype(str))
    if missing:
        raise IndexError_(f"{len(missing)} reference ids were not found in the frame")
    if subset.empty:
        raise IndexError_("reference index would be empty")
    if subset["sample_id"].astype(str).duplicated().any():
        raise IndexError_("duplicate reference sample ids")

    minhash = MinHash(
        permutations=permutations,
        seed=seed,
        shingle_size=shingle_size,
        rows_per_band=rows_per_band,
    )
    signatures = np.stack([minhash.signature(text) for text in subset["text"]])
    hashes = [
        sha256_text(normalize_for_hashing(text))
        for text in subset["text"]
    ]
    metadata = {
        "method_version": "minhash.v1",
        "created_at": utc_now(),
        "permutations": permutations,
        "seed": seed,
        "shingle_size": shingle_size,
        "rows_per_band": rows_per_band,
        "reference_count": int(len(subset)),
        "reference_scope": "explicit_id_list",
        "source_description": source_description,
        "note": "Reference scope must be train-only for inference-time use.",
    }
    return ReferenceIndex(
        sample_ids=subset["sample_id"].astype(str).tolist(),
        signatures=signatures,
        normalized_hashes=hashes,
        minhash=minhash,
        metadata=metadata,
    )
