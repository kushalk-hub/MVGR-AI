"""Load, audit and split the 10,000-row source dataset.

The deterministic branch is trained on ``10000.csv``. This module is the single
place that knows about that file: it loads it, audits it, and resolves the
train/validation/test assignment.

Split policy
------------
The repository already owns a split policy (``dataset_audit.split_manifest``)
and a candidate manifest (``man_20ddf272c292efbc8c8c``, source-scoped to
PADBen). The manifest covers 7,157 of the 10,000 rows; the remaining BiScope
rows are not in it, so they are assigned with the *same* policy
(:func:`dataset_audit.split_manifest.assign_by_family`), the same seed and the
same ratios.

That gives one assignment for all 10,000 rows that

* never splits a paraphrase family (``group_id``) across two splits,
* never splits a duplicate text across two splits,
* keeps the existing candidate manifest untouched,
* uses the repository's own policy for the rows the manifest does not cover.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from shared.paths import REPO_ROOT

DATASET_FILENAME = "10000.csv"
DEFAULT_DATASET_PATH = REPO_ROOT / DATASET_FILENAME

#: Column used to group paraphrase families together.
FAMILY_COLUMN = "group_id"
#: Column holding the pre-existing (per-row) split of the source CSV.
CSV_SPLIT_COLUMN = "split"
#: Column holding the label (``0`` human, ``1`` AI).
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
ID_COLUMN = "sample_id"

#: The candidate manifest reused for the rows it covers.
DEFAULT_MANIFEST_ID = "man_20ddf272c292efbc8c8c"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "data" / "manifests" / "candidate" / DEFAULT_MANIFEST_ID

SPLITS = ("train", "validation", "test")

#: Label convention used throughout the deterministic branch.
LABEL_HUMAN = 0
LABEL_AI = 1


class DatasetError(RuntimeError):
    """Raised when the source dataset cannot be used as specified."""


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Read the source CSV exactly as-is (no type coercion, no NA filling)."""
    csv_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not csv_path.exists():
        raise DatasetError(f"source dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [
        column
        for column in (ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN, FAMILY_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise DatasetError(f"{csv_path} is missing required columns: {missing}")
    return frame


def dataset_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else DEFAULT_DATASET_PATH


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def audit_dataset(path: str | Path | None = None) -> dict[str, Any]:
    """Return a complete, JSON-serialisable audit of the source dataset."""
    csv_path = dataset_path(path)
    frame = load_dataset(csv_path)

    texts = frame[TEXT_COLUMN].astype(str)
    stripped = texts.str.strip()
    empty_text = stripped == ""
    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")

    invalid_reasons: dict[str, int] = {}
    for reason, mask in (
        ("empty_or_whitespace_text", empty_text),
        ("missing_or_unparsable_label", labels.isna()),
    ):
        count = int(mask.sum())
        if count:
            invalid_reasons[reason] = count
    invalid_mask = empty_text | labels.isna()

    normalized = stripped.str.replace(r"\s+", " ", regex=True)
    label_counts = {
        str(int(k)): int(v)
        for k, v in labels.value_counts().sort_index().items()
        if pd.notna(k)
    }

    families = frame[FAMILY_COLUMN].astype(str)
    family_sizes = families.value_counts()
    multi_families = family_sizes[family_sizes > 1]

    csv_split = (
        frame[CSV_SPLIT_COLUMN].astype(str)
        if CSV_SPLIT_COLUMN in frame.columns
        else pd.Series(dtype=str)
    )
    family_splits = frame.groupby(FAMILY_COLUMN)[CSV_SPLIT_COLUMN].nunique()
    family_labels = frame.groupby(FAMILY_COLUMN)[LABEL_COLUMN].nunique()

    audit: dict[str, Any] = {
        "dataset_file": csv_path.name,
        "dataset_path": str(csv_path),
        "dataset_sha256": _file_sha256(csv_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "valid_records": int(len(frame) - int(invalid_mask.sum())),
        "invalid_records": int(invalid_mask.sum()),
        "invalid_reasons": invalid_reasons,
        "label_counts": label_counts,
        "label_human": int((labels == 0).sum()),
        "label_ai": int((labels == 1).sum()),
        "imbalance_ratio": (
            round(float((labels == 1).sum() / max(1, (labels == 0).sum())), 4)
        ),
        "text_type_counts": _counts(frame["text_type"]) if "text_type" in frame.columns else {},
        "paraphrase_status_counts": _counts(frame["paraphrase_status"])
        if "paraphrase_status" in frame.columns
        else {},
        "graph_category_counts": _counts(frame["graph_category"])
        if "graph_category" in frame.columns
        else {},
        "source_dataset_counts": _counts(frame["source_dataset"])
        if "source_dataset" in frame.columns
        else {},
        "source_task_counts": (
            _counts(frame["source_task"]) if "source_task" in frame.columns else {}
        ),
        "domain_counts": _counts(frame["domain"]) if "domain" in frame.columns else {},
        "ai_model_counts": _counts(frame["ai_model"]) if "ai_model" in frame.columns else {},
        "text_length_chars": {
            "min": int(texts.str.len().min()),
            "median": float(texts.str.len().median()),
            "mean": round(float(texts.str.len().mean()), 2),
            "max": int(texts.str.len().max()),
        },
        "missing_values_by_column": {
            str(column): int((frame[column].astype(str) == "").sum())
            for column in frame.columns
        },
        "duplicate_sample_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "duplicate_texts_exact": int(texts.duplicated().sum()),
        "duplicate_texts_normalized": int(normalized.duplicated().sum()),
        "families": {
            "key": FAMILY_COLUMN,
            "unique": int(families.nunique()),
            "families_with_multiple_members": int(len(multi_families)),
            "rows_in_multi_member_families": int(
                families.isin(multi_families.index).sum()
            ),
            "families_spanning_multiple_labels": int((family_labels > 1).sum()),
        },
        "source_csv_split_distribution": _counts(csv_split),
        "source_csv_families_spanning_multiple_splits": int((family_splits > 1).sum()),
    }
    return audit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# split resolution
# --------------------------------------------------------------------------- #
def read_manifest_splits(manifest_dir: str | Path | None = None) -> dict[str, set[str]]:
    """Read the id files of a candidate manifest."""
    directory = Path(manifest_dir) if manifest_dir else DEFAULT_MANIFEST_DIR
    splits: dict[str, set[str]] = {}
    for split in SPLITS:
        path = directory / f"{split}.ids.txt"
        if not path.exists():
            raise DatasetError(f"manifest id file not found: {path}")
        with open(path, encoding="utf-8") as handle:
            splits[split] = {line.strip() for line in handle if line.strip()}
    return splits


def resolve_split(
    frame: pd.DataFrame,
    *,
    manifest_dir: str | Path | None = None,
    dataset_config: dict[str, Any] | None = None,
    manifest_id: str = DEFAULT_MANIFEST_ID,
) -> pd.DataFrame:
    """Attach a resolved ``model_split`` column to the dataset.

    Rows covered by the candidate manifest keep the manifest's assignment. The
    remaining rows are assigned with the repository's own family-disjoint
    policy, using the manifest's seed and ratios.
    """
    from dataset_audit.split_manifest import assign_by_family
    from shared.paths import dataset_config as load_dataset_config

    manifest_splits = read_manifest_splits(manifest_dir)
    manifest_ids = set().union(*manifest_splits.values())

    result = frame.copy()
    result["model_split"] = ""

    covered = result[ID_COLUMN].isin(manifest_ids)
    for split, ids in manifest_splits.items():
        mask = covered & result[ID_COLUMN].isin(ids)
        result.loc[mask, "model_split"] = split

    uncovered = result[~covered].copy()
    if not uncovered.empty:
        cfg = dataset_config or load_dataset_config()
        families = uncovered[[ID_COLUMN, FAMILY_COLUMN]].rename(
            columns={FAMILY_COLUMN: "family_id"}
        )
        assignment = assign_by_family(
            uncovered[[ID_COLUMN, "source_dataset", LABEL_COLUMN]].assign(
                LABEL_COLUMN=pd.to_numeric(
                    uncovered[LABEL_COLUMN], errors="coerce"
                ).fillna(0).astype(int)
            ),
            families,
            config=cfg,
        )
        mapping = dict(zip(assignment[ID_COLUMN], assignment["split"], strict=True))
        result.loc[~covered, "model_split"] = result.loc[~covered, ID_COLUMN].map(mapping)

    if (result["model_split"] == "").any():
        unresolved = result.loc[result["model_split"] == "", ID_COLUMN].tolist()[:5]
        raise DatasetError(f"could not assign a split to {unresolved}")

    result["model_split"] = pd.Categorical(result["model_split"], categories=list(SPLITS))
    return result


def split_integrity_report(
    frame: pd.DataFrame, split_column: str = "model_split"
) -> dict[str, Any]:
    """Verify that no family, duplicate text or label cell is broken by the split."""
    report: dict[str, Any] = {"split_column": split_column}
    report["counts"] = _counts(frame[split_column])

    def _by(column: str) -> dict[str, dict[str, int]]:
        if column not in frame.columns:
            return {}
        return {
            str(split): _counts(values)
            for split, values in frame.groupby(split_column, observed=True)[column]
        }

    report["counts_by_label"] = _by(LABEL_COLUMN)
    report["counts_by_source"] = _by("source_dataset")
    report["counts_by_text_type"] = _by("text_type")

    family_splits = frame.groupby(FAMILY_COLUMN)[split_column].nunique()
    report["families_spanning_multiple_splits"] = int((family_splits > 1).sum())
    report["rows_in_families_spanning_splits"] = int(
        frame[FAMILY_COLUMN].isin(family_splits[family_splits > 1].index).sum()
    )

    if TEXT_COLUMN in frame.columns:
        normalized = (
            frame[TEXT_COLUMN]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        dupes = normalized.duplicated(keep=False)
        if dupes.any():
            spanning = frame.loc[dupes].groupby(normalized[dupes])[split_column].nunique()
            report["duplicate_texts_spanning_splits"] = int((spanning > 1).sum())
        else:
            report["duplicate_texts_spanning_splits"] = 0
    else:
        report["duplicate_texts_spanning_splits"] = 0

    report["leakage_free"] = bool(
        report["families_spanning_multiple_splits"] == 0
        and report["duplicate_texts_spanning_splits"] == 0
    )
    return report


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    """The resolved dataset plus its audit and split integrity report."""

    frame: pd.DataFrame
    audit: dict[str, Any]
    split_report: dict[str, Any]
    dataset_file: str = DATASET_FILENAME
    manifest_id: str = DEFAULT_MANIFEST_ID
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return [str(value) for value in self.frame[ID_COLUMN].tolist()]

    def split_frame(self, split: str) -> pd.DataFrame:
        return self.frame[self.frame["model_split"] == split].copy()

    def counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.frame["model_split"].value_counts().items()}


def load_bundle(
    path: str | Path | None = None,
    *,
    manifest_dir: str | Path | None = None,
    manifest_id: str = DEFAULT_MANIFEST_ID,
) -> DatasetBundle:
    """Load the dataset, audit it and resolve the leakage-safe split."""
    frame = load_dataset(path)
    audit = audit_dataset(path)
    resolved = resolve_split(frame, manifest_dir=manifest_dir, manifest_id=manifest_id)
    report = split_integrity_report(resolved)
    return DatasetBundle(
        frame=resolved,
        audit=audit,
        split_report=report,
        dataset_file=audit["dataset_file"],
        manifest_id=manifest_id,
        metadata={
            "manifest_dir": str(Path(manifest_dir) if manifest_dir else DEFAULT_MANIFEST_DIR),
        },
    )


__all__ = [
    "DATASET_FILENAME",
    "DEFAULT_MANIFEST_ID",
    "LABEL_AI",
    "LABEL_HUMAN",
    "DatasetBundle",
    "DatasetError",
    "audit_dataset",
    "dataset_path",
    "load_bundle",
    "load_dataset",
    "read_manifest_splits",
    "resolve_split",
    "split_integrity_report",
]
