"""Deterministic, content-addressed artifact identifiers.

Every artifact in MVGR-AI must be independently identifiable and reproducible
from its inputs. IDs are therefore derived from a canonical JSON serialization
of the artifact's defining payload, never from timestamps or random values.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ID_LENGTH = 16


def canonical_json(payload: Any) -> str:
    """Stable JSON used for hashing (sorted keys, no insignificant whitespace)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest(payload: Any, *, length: int = ID_LENGTH) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]


def content_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def config_id(config: dict[str, Any]) -> str:
    """Identity of a resolved run configuration."""
    return f"cfg_{digest(config)}"


def split_id(manifest_id: str) -> str:
    return f"split_{digest({'manifest_id': manifest_id})}"


def graph_build_id(
    *,
    dataset_version: str,
    split_id_value: str,
    graph_spec_version: str,
    config: dict[str, Any],
    sample_ids: list[str],
) -> str:
    """Identity of a graph build.

    ``sample_ids`` must be sorted so that ingestion order cannot change the ID.
    """
    return "gb_" + digest(
        {
            "dataset_version": dataset_version,
            "split_id": split_id_value,
            "graph_spec_version": graph_spec_version,
            "config": config,
            "sample_ids": sorted(sample_ids),
        },
        length=20,
    )


def experiment_id(
    *,
    config: dict[str, Any],
    split_id_value: str,
    graph_build_id_value: str,
) -> str:
    return (
        "exp_"
        + digest(
            {
                "config": config,
                "split_id": split_id_value,
                "graph_build_id": graph_build_id_value,
            },
            length=20,
        )
    )


def report_id(text_sha256: str, model_bundle_id: str) -> str:
    return "rep_" + digest({"text": text_sha256, "model_bundle": model_bundle_id}, length=20)


def is_prefixed_id(value: str, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix + "_"):
        return False
    suffix = value[len(prefix) + 1 :]
    return len(suffix) >= 12 and all(c in "0123456789abcdef" for c in suffix)
