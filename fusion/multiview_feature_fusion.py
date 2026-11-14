"""Module E — multi-view feature fusion.

Combines the graph branches with the new fingerprinting branches into one
sample-keyed feature matrix while keeping identifiers and metadata strictly
separate (mirrors the graph pipeline's leakage policy).

Branches
--------
* ``lexical_graph`` / ``syntactic_graph`` — the existing per-view graph features,
  re-prefixed ``glex_`` / ``gsyn_`` so they cannot collide with the stylometric
  ``syn_`` features.
* ``stylometric`` — Module A.
* ``writeprints`` — Module B.
* ``gltr`` — Module C (absent when no local model is available).
* ``watermark`` — auxiliary **evidence**, written to metadata only by default.

Guarantees
----------
* one row per ``sample_id``; duplicate ids are rejected;
* identifier / metadata columns never enter the feature matrix;
* a missing branch is skipped and recorded, never imputed as zeros;
* feature selection (missing / constant / correlated) is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

BRANCH_ORDER: tuple[str, ...] = (
    "lexical_graph",
    "syntactic_graph",
    "stylometric",
    "writeprints",
    "gltr",
    "watermark",
)

GRAPH_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "graph_id",
    "build_id",
    "source_row_id",
    "group_id",
)

GRAPH_METADATA_COLUMNS: tuple[str, ...] = (
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
)

WATERMARK_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "watermark_status",
    "watermark_provider",
    "watermark_z_score",
    "watermark_green_count",
    "watermark_green_proportion",
    "watermark_expected_proportion",
    "watermark_p_value",
    "watermark_detected",
)


class FusionError(ValueError):
    pass


@dataclass
class FusionResult:
    features: pd.DataFrame
    metadata: pd.DataFrame
    schema: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# graph branch preparation
# --------------------------------------------------------------------------- #
def split_graph_branches(
    graph_features: pd.DataFrame,
    *,
    identifier_columns: tuple[str, ...] = GRAPH_IDENTIFIER_COLUMNS,
    metadata_columns: tuple[str, ...] = GRAPH_METADATA_COLUMNS,
) -> dict[str, pd.DataFrame]:
    """Return ``{"lexical_graph": df, "syntactic_graph": df}`` keyed by sample_id.

    Only numeric, non-identifier, non-metadata columns are kept, prefixed so the
    two views cannot collide.
    """
    if "view" not in graph_features.columns:
        raise FusionError("graph_features is missing the 'view' column")
    excluded = set(identifier_columns) | set(metadata_columns) | {
        "view", "graph_spec_version", "spacy_version", "model_name", "parser_status",
    }
    frames: dict[str, pd.DataFrame] = {}
    for view, prefix, branch in (
        ("lexical_cooccurrence", "glex", "lexical_graph"),
        ("syntactic_dependency", "gsyn", "syntactic_graph"),
    ):
        subset = graph_features[graph_features["view"] == view].copy()
        if subset.empty:
            frames[branch] = pd.DataFrame(columns=["sample_id"])
            continue
        numeric = [
            column
            for column in subset.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(subset[column])
        ]
        subset = subset[["sample_id", *numeric]].copy()
        subset = subset.rename(columns={column: f"{prefix}_{column}" for column in numeric})
        # A sample has at most one row per view; guard against silent duplication.
        if subset["sample_id"].duplicated().any():
            raise FusionError(f"graph view {view!r} has duplicate sample rows")
        frames[branch] = subset.reset_index(drop=True)
    return frames


# --------------------------------------------------------------------------- #
# feature selection / normalization
# --------------------------------------------------------------------------- #
def select_features(
    features: pd.DataFrame,
    *,
    max_missing_fraction: float = 0.30,
    constant_tolerance: float = 1.0e-12,
    high_correlation_threshold: float = 0.98,
    drop_constant: bool = True,
    drop_high_correlation: bool = True,
    id_column: str = "sample_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_columns = [column for column in features.columns if column != id_column]
    report: dict[str, Any] = {
        "candidate_features": len(feature_columns),
        "dropped_missing": [],
        "dropped_constant": [],
        "dropped_correlated": [],
    }

    missing_fraction = features[feature_columns].isna().mean()
    keep = [column for column in feature_columns if missing_fraction[column] <= max_missing_fraction]
    report["dropped_missing"] = sorted(set(feature_columns) - set(keep))

    if drop_constant and keep:
        numeric = features[keep].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            spread = np.nanmax(numeric, axis=0) - np.nanmin(numeric, axis=0)
        constant = [column for column, value in zip(keep, spread) if not (value > constant_tolerance)]
        report["dropped_constant"] = sorted(constant)
        keep = [column for column in keep if column not in set(constant)]

    if drop_high_correlation and len(keep) > 1:
        matrix = features[keep].to_numpy(dtype=float)
        # Pairwise correlation on complete cases; NaN pairs are treated as 0.
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(np.nan_to_num(matrix, nan=0.0), rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        drop: set[str] = set()
        for i in range(len(keep)):
            if keep[i] in drop:
                continue
            for j in range(i + 1, len(keep)):
                if keep[j] in drop:
                    continue
                if abs(corr[i, j]) >= high_correlation_threshold:
                    drop.add(keep[j])
        report["dropped_correlated"] = sorted(drop)
        keep = [column for column in keep if column not in drop]

    report["selected_features"] = len(keep)
    selected = features[[id_column, *keep]].copy()
    return selected, report


def normalize_features(
    features: pd.DataFrame, method: str, *, id_column: str = "sample_id"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method = (method or "none").lower()
    feature_columns = [column for column in features.columns if column != id_column]
    report: dict[str, Any] = {"method": method, "columns": {}}
    if method in ("none", ""):
        return features.copy(), report
    if method not in ("zscore", "robust", "minmax"):
        raise FusionError(f"unknown normalization method {method!r}")

    output = features.copy()
    for column in feature_columns:
        values = output[column].to_numpy(dtype=float)
        if method == "zscore":
            center = np.nanmean(values)
            scale = np.nanstd(values, ddof=0)
        elif method == "robust":
            center = np.nanmedian(values)
            q1, q3 = np.nanpercentile(values, [25, 75])
            scale = (q3 - q1) / 1.349
        else:  # minmax
            center = np.nanmin(values)
            scale = np.nanmax(values) - np.nanmin(values)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
            center = 0.0
        output[column] = (values - center) / scale
        report["columns"][column] = {"center": float(center), "scale": float(scale)}
    return output, report


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #
def _prefixed_frame(frame: pd.DataFrame, prefix: str, id_column: str = "sample_id") -> pd.DataFrame:
    columns = {column: f"{prefix}{column}" for column in frame.columns if column != id_column}
    return frame.rename(columns=columns)


def fuse_multiview(
    *,
    branch_frames: dict[str, pd.DataFrame | None],
    metadata: pd.DataFrame,
    watermark_frame: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    id_column: str = "sample_id",
) -> FusionResult:
    """Merge branch frames into one feature matrix plus a separate metadata table."""
    config = config or {}
    selection_cfg = config.get("feature_selection", {})
    identifier_columns = tuple(config.get("identifier_columns", ("sample_id", "graph_id", "build_id")))
    metadata_columns = tuple(config.get("metadata_columns", GRAPH_METADATA_COLUMNS))

    if metadata[id_column].duplicated().any():
        raise FusionError("metadata contains duplicate sample ids")
    sample_ids = metadata[[id_column]].copy()

    availability: dict[str, bool] = {}
    feature_blocks: list[pd.DataFrame] = []
    branch_feature_columns: dict[str, list[str]] = {}

    for branch in BRANCH_ORDER:
        if branch == "watermark":
            continue
        frame = branch_frames.get(branch)
        if frame is None or frame.empty or not any(column != id_column for column in frame.columns):
            availability[branch] = False
            branch_feature_columns[branch] = []
            continue
        availability[branch] = True
        block = _prefixed_frame(frame, f"{branch}__", id_column=id_column)
        feature_blocks.append(block)
        branch_feature_columns[branch] = [
            column for column in block.columns if column != id_column
        ]

    merged = sample_ids.copy()
    for block in feature_blocks:
        merged = merged.merge(block, on=id_column, how="left")

    # Metadata + availability + optional watermark evidence.
    out_metadata = metadata.copy()
    for branch, present in availability.items():
        out_metadata[f"has_{branch}"] = int(present)
    out_metadata["has_watermark"] = int(watermark_frame is not None and not watermark_frame.empty)

    if watermark_frame is not None and not watermark_frame.empty:
        evidence = watermark_frame.copy()
        out_metadata = out_metadata.merge(evidence, on=id_column, how="left")
    if config.get("watermark_as_feature", False) and watermark_frame is not None:
        numeric_evidence = [
            column
            for column in WATERMARK_EVIDENCE_COLUMNS
            if column in watermark_frame.columns
            and pd.api.types.is_numeric_dtype(watermark_frame[column])
        ]
        if numeric_evidence:
            extra = watermark_frame[[id_column, *numeric_evidence]].rename(
                columns={column: f"watermark__{column}" for column in numeric_evidence}
            )
            merged = merged.merge(extra, on=id_column, how="left")
            branch_feature_columns["watermark"] = [
                column for column in extra.columns if column != id_column
            ]
            availability["watermark"] = True
        else:
            availability["watermark"] = False
    else:
        availability["watermark"] = False

    # Never allow an identifier to slip into the feature matrix.
    leaked = [column for column in merged.columns if column in set(identifier_columns) and column != id_column]
    if leaked:
        raise FusionError(f"identifier columns leaked into features: {leaked}")

    selected, selection_report = select_features(
        merged,
        max_missing_fraction=float(selection_cfg.get("max_missing_fraction", 0.30)),
        constant_tolerance=float(selection_cfg.get("constant_feature_tolerance", 1.0e-12)),
        high_correlation_threshold=float(selection_cfg.get("high_correlation_threshold", 0.98)),
        drop_constant=bool(selection_cfg.get("drop_constant", True)),
        drop_high_correlation=bool(selection_cfg.get("drop_high_correlation", True)),
        id_column=id_column,
    )
    normalized, normalization_report = normalize_features(
        selected, str(config.get("normalization", "none")), id_column=id_column
    )

    schema = build_schema(
        features=normalized,
        metadata=out_metadata,
        availability=availability,
        branch_feature_columns=branch_feature_columns,
        selection_report=selection_report,
        normalization_report=normalization_report,
        config=config,
        id_column=id_column,
    )
    report = {
        "samples": int(len(normalized)),
        "features": int(len([c for c in normalized.columns if c != id_column])),
        "availability": availability,
        "selection": selection_report,
        "normalization": normalization_report,
    }
    return FusionResult(features=normalized, metadata=out_metadata, schema=schema, report=report)


def build_schema(
    *,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    availability: dict[str, bool],
    branch_feature_columns: dict[str, list[str]],
    selection_report: dict[str, Any],
    normalization_report: dict[str, Any],
    config: dict[str, Any],
    id_column: str = "sample_id",
) -> dict[str, Any]:
    from shared.contracts.versions import FINGERPRINT_FEATURE_VERSION, FUSION_VERSION

    experiments = config.get("experiments", {})
    experiment_inputs: dict[str, list[str]] = {}
    for key, definition in experiments.items():
        branches = definition.get("branches", []) if isinstance(definition, dict) else []
        columns: list[str] = []
        for branch in branches:
            columns.extend(branch_feature_columns.get(branch, []))
        experiment_inputs[key] = columns

    return {
        "feature_set_version": FINGERPRINT_FEATURE_VERSION,
        "fusion_version": FUSION_VERSION,
        "id_column": id_column,
        "null_policy": "undefined features are emitted as empty cells (NaN), never imputed",
        "branches": list(BRANCH_ORDER),
        "branch_availability": availability,
        "branch_feature_columns": branch_feature_columns,
        "feature_count": int(len([c for c in features.columns if c != id_column])),
        "features": [c for c in features.columns if c != id_column],
        "metadata_columns": [c for c in metadata.columns],
        "identifier_policy": "identifier columns are never features",
        "metadata_policy": "metadata columns are preserved for joins/stratification, never features",
        "watermark_policy": "watermark is auxiliary evidence in metadata, never a feature by default",
        "selection": selection_report,
        "normalization": {"method": normalization_report.get("method", "none")},
        "experiments": experiment_inputs,
        "leakage_note": (
            "Labels, source, task, family, condition and split never enter the "
            "feature matrix; similarity and watermark signals are evidence, not "
            "authorship proof."
        ),
    }


__all__ = [
    "BRANCH_ORDER",
    "FusionError",
    "FusionResult",
    "build_schema",
    "fuse_multiview",
    "normalize_features",
    "select_features",
    "split_graph_branches",
]
