"""ML-ready dataset preparation for the MVGR-AI graph features (Phase D, E).

Consumes the tables written by ``scripts/extract_graph_features.py`` and produces
a model-ready feature matrix plus a separate metadata/provenance table:

* ``data/processed/ml_ready_features.csv``  numeric features, one row per sample
* ``data/processed/ml_ready_metadata.csv``  identifiers, target and provenance
* ``data/processed/ml_dataset_report.json`` full preparation report

What this stage does
--------------------
* separates *predictive features* from *identifiers* and *metadata* using the
  schema written by the extractor (never by guessing),
* reports missing-value statistics and drops features above the configured
  missing fraction,
* drops constant and near-constant features,
* drops one feature from each highly-correlated pair (threshold from config),
* encodes the target label and preserves it only in the metadata table,
* runs the leakage checks and writes their result into the report.

What this stage deliberately does NOT do
----------------------------------------
* It never trains or evaluates a model.
* It never invents a formal train/validation/test split. A formal split requires
  an *approved* manifest (``leakage.approved_manifest_pointer``). If that file
  is absent the formal split is skipped and the blocker is reported. The pilot's
  preserved ``split`` column is carried through for pipeline validation only and
  is explicitly marked as non-formal.

Usage
-----
    python scripts/prepare_ml_dataset.py
    python scripts/prepare_ml_dataset.py --graph-features data/processed/graph_features.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.paths import load_config, resolve  # noqa: E402

NON_FEATURE_ROLES = {"identifier", "metadata", "descriptive"}


# --------------------------------------------------------------------------- #
# feature selection
# --------------------------------------------------------------------------- #
def schema_feature_columns(schema: dict[str, Any], table: str) -> tuple[list[str], list[str]]:
    """Return (predictive, non_predictive) columns for ``table`` from the schema."""
    columns = schema["tables"][table]["columns"]
    predictive = [c["name"] for c in columns if c.get("predictive")]
    non_predictive = [c["name"] for c in columns if not c.get("predictive")]
    return predictive, non_predictive


def numeric_feature_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Keep candidates that are present and numeric; drop non-numeric surprises."""
    keep: list[str] = []
    for column in candidates:
        if column not in frame.columns:
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            keep.append(column)
    return keep


def missing_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    if not columns:
        return {}
    return {c: float(frame[c].isna().mean()) for c in columns}


def constant_features(frame: pd.DataFrame, columns: list[str], tolerance: float) -> list[str]:
    """Columns whose non-null values vary by at most ``tolerance``."""
    constant: list[str] = []
    for column in columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            constant.append(column)
            continue
        if float(np.nanmax(values) - np.nanmin(values)) <= tolerance:
            constant.append(column)
    return constant


def high_correlation_drops(
    frame: pd.DataFrame, columns: list[str], threshold: float
) -> list[tuple[str, str, float]]:
    """Greedily drop one column from every pair correlated above ``threshold``.

    Columns are considered in the given order; when a pair exceeds the threshold
    the *later* column is dropped (the earlier one is retained). Returns the list
    of ``(kept, dropped, correlation)`` triples.
    """
    if len(columns) < 2:
        return []
    matrix = frame[columns].to_numpy(dtype=float)
    # Pairwise complete-observation correlation; NaN where undefined.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False)
    corr = np.abs(np.nan_to_num(corr, nan=0.0))
    dropped: list[tuple[str, str, float]] = []
    removed: set[int] = set()
    for i in range(len(columns)):
        if i in removed:
            continue
        for j in range(i + 1, len(columns)):
            if j in removed:
                continue
            value = float(corr[i, j])
            if value >= threshold:
                removed.add(j)
                dropped.append((columns[i], columns[j], value))
    return dropped


def select_features(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    max_missing_fraction: float,
    constant_tolerance: float,
    correlation_threshold: float,
) -> dict[str, Any]:
    """Apply the missing / constant / correlation gates in order."""
    candidates = numeric_feature_columns(frame, candidates)
    missing = missing_summary(frame, candidates)
    kept_after_missing = [c for c in candidates if missing.get(c, 0.0) <= max_missing_fraction]
    dropped_missing = [c for c in candidates if c not in set(kept_after_missing)]

    constant = constant_features(frame, kept_after_missing, constant_tolerance)
    kept_after_constant = [c for c in kept_after_missing if c not in set(constant)]

    correlated = high_correlation_drops(frame, kept_after_constant, correlation_threshold)
    dropped_corr = {d for _, d, _ in correlated}
    final = [c for c in kept_after_constant if c not in dropped_corr]

    return {
        "candidates": candidates,
        "selected": final,
        "dropped_missing": dropped_missing,
        "dropped_constant": constant,
        "dropped_correlated": correlated,
        "missing_fraction": missing,
    }


# --------------------------------------------------------------------------- #
# leakage checks
# --------------------------------------------------------------------------- #
def check_graph_view_split_consistency(graph_frame: pd.DataFrame) -> dict[str, Any]:
    """Both graph views of a sample must share one split value."""
    if "split" not in graph_frame.columns:
        return {"checked": False, "reason": "no split column"}
    grouped = graph_frame.groupby("sample_id")["split"].nunique(dropna=False)
    violations = grouped[grouped > 1]
    return {
        "checked": True,
        "samples_with_multiple_splits": int(len(violations)),
        "example_violations": violations.index[:10].tolist(),
        "passed": bool(len(violations) == 0),
    }


def check_group_split_purity(metadata: pd.DataFrame, group_column: str = "group_id") -> dict[str, Any]:
    """Related samples (same group) must not cross splits where lineage exists."""
    if group_column not in metadata.columns or "split" not in metadata.columns:
        return {"checked": False, "reason": f"missing {group_column} or split"}
    valid = metadata[metadata[group_column].notna()]
    grouped = valid.groupby(group_column)["split"].nunique(dropna=False)
    violations = grouped[grouped > 1]
    return {
        "checked": True,
        "groups_with_multiple_splits": int(len(violations)),
        "example_violations": [str(v) for v in violations.index[:10]],
        "passed": bool(len(violations) == 0),
    }


def check_no_identifier_features(
    feature_columns: list[str], identifier_columns: list[str]
) -> dict[str, Any]:
    leaked = [c for c in feature_columns if c in set(identifier_columns)]
    return {"checked": True, "leaked_identifiers": leaked, "passed": not leaked}


def check_no_metadata_features(
    feature_columns: list[str], metadata_columns: list[str]
) -> dict[str, Any]:
    leaked = [c for c in feature_columns if c in set(metadata_columns)]
    return {"checked": True, "leaked_metadata": leaked, "passed": not leaked}


def check_label_not_a_feature(feature_columns: list[str], label_column: str) -> dict[str, Any]:
    return {
        "checked": True,
        "label_in_features": label_column in set(feature_columns),
        "passed": label_column not in set(feature_columns),
    }


def check_one_row_per_sample(features: pd.DataFrame) -> dict[str, Any]:
    duplicates = int(features["sample_id"].duplicated().sum()) if "sample_id" in features else 0
    return {
        "checked": True,
        "duplicate_sample_rows": duplicates,
        "n_samples": int(len(features)),
        "passed": duplicates == 0,
    }


def check_feature_metadata_alignment(
    features: pd.DataFrame, metadata: pd.DataFrame
) -> dict[str, Any]:
    if "sample_id" not in features.columns or "sample_id" not in metadata.columns:
        return {"checked": False, "reason": "sample_id missing from one table"}
    same = features["sample_id"].tolist() == metadata["sample_id"].tolist()
    return {
        "checked": True,
        "row_aligned": bool(same),
        "passed": bool(same),
    }


def approved_manifest_status(config: dict[str, Any]) -> dict[str, Any]:
    """Report whether a formal split may be generated (approved manifest present)."""
    pointer = config["leakage"]["approved_manifest_pointer"]
    path = resolve(pointer)
    if not path.is_file():
        return {
            "approved_manifest_found": False,
            "path": pointer,
            "formal_split_generated": False,
            "reason": (
                "No approved split manifest exists. Formal train/validation/test "
                "generation is intentionally skipped; only the pilot's preserved "
                "split column is carried for pipeline validation and is NOT a "
                "formal split."
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    approved = str(payload.get("status", "")).lower() == "approved" and bool(payload.get("approvals"))
    return {
        "approved_manifest_found": True,
        "path": pointer,
        "manifest_id": payload.get("manifest_id"),
        "status": payload.get("status"),
        "approved": approved,
        "formal_split_generated": False,
        "reason": (
            "An approved manifest pointer exists; formal split generation is a "
            "separate, explicitly authorised step and is not performed here."
        ),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def prepare(config: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    outputs = config["outputs"]
    validation = config["validation"]
    leakage = config["leakage"]

    graph_features_path = resolve(outputs["graph_features"])
    multiview_path = resolve(outputs["sample_multiview_features"])
    schema_path = resolve(outputs["feature_schema"])
    for path in (graph_features_path, multiview_path, schema_path):
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}. Run scripts/extract_graph_features.py first."
            )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    graph_frame = pd.read_csv(graph_features_path)
    sample_frame = pd.read_csv(multiview_path)

    identifier_columns = list(leakage["identifier_columns"])
    metadata_columns = list(leakage["metadata_columns"])
    label_column = "label"

    candidates, non_predictive = schema_feature_columns(schema, "sample_multiview_features")

    selection = select_features(
        sample_frame,
        candidates,
        max_missing_fraction=float(validation["max_missing_fraction"]),
        constant_tolerance=float(validation["constant_feature_tolerance"]),
        correlation_threshold=float(validation["high_correlation_threshold"]),
    )
    selected = selection["selected"]

    # Metadata table: identifiers + metadata, one row per sample, row-aligned.
    ordered_metadata = [c for c in ["sample_id"] + metadata_columns if c in sample_frame.columns]
    metadata = sample_frame[ordered_metadata].copy()

    # Feature matrix: selected numeric features only, one row per sample.
    features = sample_frame[selected].copy()

    # Target encoding lives only in the metadata table.
    label_mapping: dict[str, int] = {}
    if label_column in metadata.columns:
        values = sorted(str(v) for v in metadata[label_column].dropna().unique())
        label_mapping = {value: index for index, value in enumerate(values)}
        metadata["label_encoded"] = metadata[label_column].map(
            lambda v: label_mapping.get(str(v), -1) if pd.notna(v) else -1
        ).astype(int)

    leakage_checks = {
        "graph_view_split_consistency": check_graph_view_split_consistency(graph_frame),
        "group_split_purity": check_group_split_purity(
            sample_frame[[c for c in ("group_id", "split") if c in sample_frame.columns]]
        ),
        "no_identifier_features": check_no_identifier_features(selected, identifier_columns),
        "no_metadata_features": check_no_metadata_features(selected, metadata_columns),
        "label_not_a_feature": check_label_not_a_feature(selected, label_column),
        "one_row_per_sample": check_one_row_per_sample(
            sample_frame[["sample_id"]] if "sample_id" in sample_frame else sample_frame
        ),
        "feature_metadata_alignment": check_feature_metadata_alignment(
            sample_frame[["sample_id"]] if "sample_id" in sample_frame else sample_frame,
            metadata,
        ),
    }
    leakage_passed = all(
        result.get("passed", True) for result in leakage_checks.values()
    )

    manifest = approved_manifest_status(config)

    # Persist outputs (features and metadata stay separate; never merged into a
    # single training file that would leak the target into the inputs).
    features_path = resolve(outputs["ml_features"])
    metadata_path = resolve(outputs["ml_metadata"])
    report_path = resolve(outputs["ml_report"])
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    metadata.to_csv(metadata_path, index=False)

    report: dict[str, Any] = {
        "generated_at": started.isoformat(),
        "generated_from": {
            "graph_features": outputs["graph_features"],
            "sample_multiview_features": outputs["sample_multiview_features"],
            "feature_schema": outputs["feature_schema"],
        },
        "build_id": schema.get("build_id"),
        "grain": "one row per sample",
        "n_samples": int(len(sample_frame)),
        "target": {
            "column": label_column,
            "encoding": label_mapping,
            "present_only_in_metadata": True,
        },
        "feature_selection": {
            "candidate_count": len(selection["candidates"]),
            "selected_count": len(selected),
            "dropped_missing": selection["dropped_missing"],
            "dropped_constant": selection["dropped_constant"],
            "dropped_correlated": [
                {"kept": kept, "dropped": dropped, "abs_correlation": round(value, 6)}
                for kept, dropped, value in selection["dropped_correlated"]
            ],
            "missing_fraction": selection["missing_fraction"],
            "selected_features": selected,
            "non_predictive_columns": non_predictive,
        },
        "metadata_columns": ordered_metadata,
        "identifier_columns": identifier_columns,
        "leakage_policy": leakage,
        "leakage_checks": leakage_checks,
        "leakage_passed": leakage_passed,
        "formal_split": manifest,
        "pilot_split_values": (
            sample_frame["split"].value_counts(dropna=False).to_dict()
            if "split" in sample_frame.columns else {}
        ),
        "training_performed": False,
        "notes": [
            "The 500-sample mixed pilot is for PIPELINE VALIDATION ONLY and is not "
            "an approved training set.",
            "No model was trained and no test-set metric was computed.",
            "The pilot 'split' column is preserved for validation only; it is not "
            "a formal, approved split.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {features_path}  rows={len(features)} cols={len(features.columns)}")
    print(f"Wrote {metadata_path}  rows={len(metadata)} cols={len(metadata.columns)}")
    print(f"Wrote {report_path}")
    print(f"Candidates={len(selection['candidates'])} selected={len(selected)} "
          f"(dropped: missing={len(selection['dropped_missing'])}, "
          f"constant={len(selection['dropped_constant'])}, "
          f"correlated={len(selection['dropped_correlated'])})")
    print(f"Leakage checks passed: {leakage_passed}")
    print(f"Formal split generated: {manifest['formal_split_generated']} "
          f"(approved manifest found: {manifest['approved_manifest_found']})")
    report["elapsed_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="graph_feature_config")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    prepare(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
