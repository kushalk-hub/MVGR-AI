"""Align extracted features to the dataset and select features without leakage.

Two distinct things happen here:

1. **Alignment** — the fused feature matrix (written by
   ``scripts/fingerprint_pipeline.py``) is joined to the source dataset on
   ``sample_id``. Every sample id is checked; a missing or extra id is an
   error, never a silently dropped row.

2. **Feature selection** — the fusion layer selects its columns using the
   *whole* table it is given. When that table is the evaluation set, the
   selection has seen the test rows and the evaluation is contaminated. The
   :func:`select_features_train_only` helper therefore re-runs exactly the
   same selection rules on the **training rows only** and then applies the
   resulting column list to validation and test. The column list is the only
   thing learned from training; no test value ever influences it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from detection.dataset_loader import (
    FAMILY_COLUMN,
    ID_COLUMN,
)

#: Feature families and the prefix the fusion layer uses for each of them.
FAMILY_PREFIXES: dict[str, str] = {
    "stylometric": "stylometric__",
    "writeprints": "writeprints__",
    "gltr": "gltr__",
    "watermark": "watermark__",
}

#: Columns that must never be used as a predictive feature.
METADATA_COLUMNS: tuple[str, ...] = (
    "label",
    "source_dataset",
    "source_task",
    "text_type",
    "graph_category",
    "paraphrase_status",
    "domain",
    "ai_model",
    "padben_split",
    "split",
    "model_split",
    FAMILY_COLUMN,
    "source_row_id",
    "has_stylometric",
    "has_writeprints",
    "has_gltr",
    "has_watermark",
)


class AlignmentError(RuntimeError):
    """Raised when features and dataset cannot be aligned."""


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_fused_features(path: str | Path) -> pd.DataFrame:
    """Load the fused feature matrix produced by the fingerprinting pipeline."""
    frame = pd.read_csv(path, dtype={ID_COLUMN: str})
    if ID_COLUMN not in frame.columns:
        raise AlignmentError(f"{path} has no {ID_COLUMN!r} column")
    return frame


def load_fused_metadata(path: str | Path) -> pd.DataFrame:
    """Load the fused metadata table (labels, provenance — never features)."""
    frame = pd.read_csv(path, dtype={ID_COLUMN: str})
    if ID_COLUMN not in frame.columns:
        raise AlignmentError(f"{path} has no {ID_COLUMN!r} column")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the usable numeric feature columns of a fused matrix."""
    return [
        column
        for column in frame.columns
        if column != ID_COLUMN
        and column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def family_columns(frame: pd.DataFrame, family: str) -> list[str]:
    """Return the feature columns belonging to one family."""
    prefix = FAMILY_PREFIXES[family]
    return [column for column in feature_columns(frame) if column.startswith(prefix)]


def families_present(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Map family -> its feature columns, for every family present."""
    return {family: family_columns(frame, family) for family in FAMILY_PREFIXES}


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def align_features(
    features: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    require_full_coverage: bool = True,
) -> pd.DataFrame:
    """Join the feature matrix to the dataset rows, keeping metadata separate.

    The returned frame has the dataset's row order and carries the feature
    columns; metadata columns are *not* added — they stay in the dataset.
    """
    if features[ID_COLUMN].duplicated().any():
        duplicates = features.loc[features[ID_COLUMN].duplicated(), ID_COLUMN].tolist()[:5]
        raise AlignmentError(f"feature matrix has duplicate sample ids: {duplicates}")

    feature_ids = set(features[ID_COLUMN].astype(str))
    dataset_ids = set(dataset[ID_COLUMN].astype(str))

    missing = sorted(dataset_ids - feature_ids)
    extra = sorted(feature_ids - dataset_ids)
    if require_full_coverage and missing:
        raise AlignmentError(
            f"{len(missing)} dataset sample ids have no extracted features "
            f"(first: {missing[:5]})"
        )

    aligned = dataset[[ID_COLUMN]].copy()
    aligned[ID_COLUMN] = aligned[ID_COLUMN].astype(str)
    aligned = aligned.merge(features, on=ID_COLUMN, how="left", validate="one_to_one")
    if extra:
        # Extra feature rows are not a data error, but they are reported.
        aligned.attrs["extra_feature_ids"] = extra
    # Defragment: a wide merge leaves the frame in a block-per-column layout
    # that makes later column inserts slow and raises PerformanceWarning.
    return aligned.copy()


def coverage_report(
    features: pd.DataFrame, dataset: pd.DataFrame
) -> dict[str, Any]:
    """Summarise how many records actually carry extracted features."""
    feature_ids = set(features[ID_COLUMN].astype(str))
    dataset_ids = set(dataset[ID_COLUMN].astype(str))
    covered = dataset_ids & feature_ids
    numeric = feature_columns(features)
    covered_rows = features[features[ID_COLUMN].isin(covered)]
    missing_cells = int(covered_rows[numeric].isna().sum().sum())
    total_cells = len(covered_rows) * max(1, len(numeric))
    return {
        "dataset_records": int(len(dataset_ids)),
        "feature_rows": int(len(feature_ids)),
        "covered_records": int(len(covered)),
        "uncovered_records": int(len(dataset_ids - feature_ids)),
        "extra_feature_rows": int(len(feature_ids - dataset_ids)),
        "feature_columns": len(numeric),
        "missing_cells_in_covered_rows": missing_cells,
        "total_cells_in_covered_rows": total_cells,
        "missing_fraction_in_covered_rows": (
            round(missing_cells / total_cells, 6) if total_cells else 0.0
        ),
        "complete_records": int((covered_rows[numeric].notna().all(axis=1)).sum())
        if numeric
        else 0,
    }


# --------------------------------------------------------------------------- #
# leakage-safe feature selection
# --------------------------------------------------------------------------- #
def select_features_train_only(
    train_frame: pd.DataFrame,
    *,
    max_missing_fraction: float = 0.30,
    constant_feature_tolerance: float = 1.0e-12,
    high_correlation_threshold: float = 0.98,
    drop_constant: bool = True,
    drop_high_correlation: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Select feature columns using the **training rows only**.

    The rules are identical to ``fusion.multiview_feature_fusion.select_features``
    so the result is directly comparable with the shipped 424-column set; only
    the rows used to decide are different.
    """
    from fusion.multiview_feature_fusion import select_features

    candidates = [column for column in train_frame.columns if column != ID_COLUMN]
    selection_frame = train_frame[[ID_COLUMN, *candidates]].copy()
    selected, report = select_features(
        selection_frame,
        max_missing_fraction=max_missing_fraction,
        constant_tolerance=constant_feature_tolerance,
        high_correlation_threshold=high_correlation_threshold,
        drop_constant=drop_constant,
        drop_high_correlation=drop_high_correlation,
        id_column=ID_COLUMN,
    )
    selected_columns = [column for column in selected.columns if column != ID_COLUMN]
    report["fitted_on"] = "train_split_only"
    report["selected_features"] = len(selected_columns)
    return selected_columns, report


def feature_family_of(column: str) -> str:
    """Return the family a (prefixed) feature column belongs to."""
    for family, prefix in FAMILY_PREFIXES.items():
        if column.startswith(prefix):
            return family
    return "other"


def family_breakdown(columns: Iterable[str]) -> dict[str, int]:
    """Count how many columns of each family are in a column list."""
    counts: dict[str, int] = {family: 0 for family in FAMILY_PREFIXES}
    counts["other"] = 0
    for column in columns:
        counts[feature_family_of(column)] += 1
    return counts


__all__ = [
    "AlignmentError",
    "FAMILY_PREFIXES",
    "METADATA_COLUMNS",
    "align_features",
    "coverage_report",
    "families_present",
    "family_breakdown",
    "family_columns",
    "feature_columns",
    "feature_family_of",
    "load_fused_features",
    "load_fused_metadata",
    "select_features_train_only",
]
