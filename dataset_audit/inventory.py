"""Dataset inventory and provenance (Person 1).

Read-only with respect to the canonical dataset. Every count emitted here is
computed from the file, never copied from a report or hard-coded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from shared.contracts.versions import TOKENIZER_VERSION
from shared.paths import canonical_dataset_path, dataset_config, source_dataset_path
from shared.preprocessing.text import normalize_for_hashing, sha256_text
from shared.validation import require_columns

REQUIRED_COLUMNS = (
    "sample_id",
    "text",
    "label",
    "source_dataset",
    "source_task",
    "text_type",
    "paraphrase_status",
    "domain",
    "ai_model",
    "split",
    "group_id",
    "source_row_id",
)

AUDIT_COLUMNS = (
    "sample_id",
    "text",
    "label",
    "source_dataset",
    "source_task",
    "text_type",
    "paraphrase_status",
    "domain",
    "ai_model",
    "split",
    "group_id",
    "source_row_id",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_provenance(
    *,
    config: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare the canonical contract copy against the immutable upstream file.

    The canonical copy under ``data/processed`` must carry the same SHA-256 as the
    upstream ``datasets/processed`` file. Any divergence -- or a missing upstream
    file -- fails closed: callers must not audit a dataset whose provenance is
    unproven.
    """
    config = config or dataset_config()
    dataset_path = Path(dataset_path) if dataset_path else canonical_dataset_path(config)
    source_path = Path(source_path) if source_path else source_dataset_path(config)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {dataset_path}")
    if not source_path.exists():
        raise FileNotFoundError(
            f"Upstream source dataset not found: {source_path}. The canonical copy "
            "cannot be verified against an immutable source."
        )

    canonical_sha = file_sha256(dataset_path)
    source_sha = file_sha256(source_path)
    recorded_sha = config["dataset"].get("sha256")
    match = canonical_sha == source_sha
    if not match:
        raise ValueError(
            "Provenance mismatch: canonical copy "
            f"{dataset_path} sha256={canonical_sha} does not match upstream "
            f"{source_path} sha256={source_sha}. The contract copy is not the "
            "immutable source; refusing to audit."
        )
    if recorded_sha and recorded_sha != canonical_sha:
        raise ValueError(
            f"Recorded dataset.sha256={recorded_sha} does not match the file "
            f"sha256={canonical_sha}; the dataset changed after it was recorded."
        )
    return {
        "canonical_path": str(dataset_path),
        "source_path": str(source_path),
        "sha256": canonical_sha,
        "source_sha256": source_sha,
        "recorded_sha256": recorded_sha,
        "match": True,
    }


def load_dataset(
    *,
    config: dict[str, Any] | None = None,
    path: str | Path | None = None,
    columns: tuple[str, ...] = AUDIT_COLUMNS,
) -> pd.DataFrame:
    config = config or dataset_config()
    path = Path(path) if path else canonical_dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {path}")
    frame = pd.read_parquet(path, columns=list(columns))
    require_columns(frame.columns, REQUIRED_COLUMNS, where=f"dataset:{path.name}")
    return frame


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.fillna("<missing>").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def build_inventory(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    config = config or dataset_config()
    dataset_path = Path(dataset_path) if dataset_path else canonical_dataset_path(config)

    frame = frame.copy()
    frame["text"] = frame["text"].fillna("")
    normalized_hashes = frame["text"].map(normalize_for_hashing).map(sha256_text)
    lengths = frame["text"].str.len()

    provenance = verify_provenance(
        config=config, dataset_path=dataset_path, source_path=source_dataset_path(config)
    )

    duplicate_rows = int(normalized_hashes.duplicated().sum())
    duplicate_groups = int(normalized_hashes[normalized_hashes.duplicated()].nunique())
    empty_rows = int((lengths == 0).sum())
    short_rows = int((lengths < 40).sum())

    label_by_source = (
        frame.groupby(["source_dataset", "label"], dropna=False).size().unstack(fill_value=0)
    )
    label_by_task = (
        frame.groupby(["source_task", "label"], dropna=False).size().unstack(fill_value=0)
    )
    label_by_split = (
        frame.groupby(["split", "label"], dropna=False).size().unstack(fill_value=0)
    )

    return {
        "audit_versions": {
            "tokenizer": TOKENIZER_VERSION,
            "dataset_version": config["dataset"]["version"],
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": provenance["sha256"],
            "size_bytes": dataset_path.stat().st_size,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "provenance": provenance,
        },
        "labels": {
            "semantics": config["dataset"]["label_semantics"],
            "counts": _value_counts(frame["label"]),
            "null_count": int(frame["label"].isna().sum()),
        },
        "sources": {
            "counts": _value_counts(frame["source_dataset"]),
            "label_crosstab": label_by_source.to_dict(orient="index"),
        },
        "tasks": {
            "counts": _value_counts(frame["source_task"]),
            "label_crosstab": label_by_task.to_dict(orient="index"),
        },
        "conditions": {
            "text_type_counts": _value_counts(frame["text_type"]),
            "paraphrase_status_counts": _value_counts(frame["paraphrase_status"]),
        },
        "splits": {
            "counts": _value_counts(frame["split"]),
            "label_crosstab": label_by_split.to_dict(orient="index"),
        },
        "lineage": {
            "unique_group_ids": int(frame["group_id"].fillna("").nunique()),
            "rows_without_group_id": int(frame["group_id"].fillna("").eq("").sum()),
            "group_size_counts": _value_counts(frame["group_id"].value_counts()),
        },
        "text": {
            "length": {
                "mean": float(lengths.mean()),
                "std": float(lengths.std()),
                "min": int(lengths.min()),
                "p25": float(lengths.quantile(0.25)),
                "median": float(lengths.median()),
                "p75": float(lengths.quantile(0.75)),
                "max": int(lengths.max()),
            },
            "empty_rows": empty_rows,
            "shorter_than_40_chars": short_rows,
            "duplicate_normalized_rows": duplicate_rows,
            "duplicate_normalized_groups": duplicate_groups,
            "unique_normalized_texts": int(normalized_hashes.nunique()),
        },
        "warnings": _inventory_warnings(frame, duplicate_rows, empty_rows),
    }


def _inventory_warnings(frame: pd.DataFrame, duplicate_rows: int, empty_rows: int) -> list[str]:
    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} rows share normalized text with another row; "
            "deduplication policy must be applied before splitting."
        )
    if empty_rows:
        warnings.append(f"{empty_rows} rows have empty text.")
    if frame["ai_model"].fillna("Unknown").eq("Unknown").all():
        warnings.append("No generator metadata is available for any row; model-specific claims are unsupported.")
    if (frame["text_type"] == "unknown").any():
        warnings.append("Some rows have text_type=unknown; condition-specific claims must exclude them.")
    return warnings
