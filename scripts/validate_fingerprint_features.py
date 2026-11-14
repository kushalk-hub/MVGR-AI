"""Validate the offline fingerprinting / GLTR / watermark / fusion artifacts.

Read-only. It checks the artifacts written by ``scripts/fingerprint_pipeline.py``
against the fused schema and the project's leakage and null policies:

1.  Artifact presence and row counts (one row per pilot sample).
2.  Unique ``sample_id`` in every artifact (duplicate detection).
3.  Sample sets aligned across the branch tables and the fused tables.
4.  Fused feature columns match ``fingerprint_feature_schema.json`` exactly.
5.  No identifier or metadata column is present in the feature matrix.
6.  No branch's ratio/frequency features fall outside [0, 1].
7.  No infinities and no negative counts/frequencies.
8.  Missing-value fractions stay within the configured threshold.
9.  The watermark table never turns ``unsupported`` into a negative class.
10. GLTR availability is reported consistently with the extraction report.

Outputs
-------
``outputs/fingerprint_audit/fingerprint_validation_report.json``
``outputs/fingerprint_audit/fingerprint_validation_report.md``

Usage
-----
    python scripts/validate_fingerprint_features.py
    python scripts/validate_fingerprint_features.py --config fingerprint_config
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.paths import load_config, resolve  # noqa: E402

PASS = "pass"
WARN = "warn"
FAIL = "fail"

# Columns that may legitimately hold negative values. Flesch Reading Ease and
# Flesch-Kincaid Grade are both defined on the whole real line for very hard or
# very short texts.
SIGNED_SUFFIXES = ("read_flesch_reading_ease", "read_flesch_kincaid_grade")

# Columns that must lie in [0, 1]. Matched by suffix so branch-prefixed fused
# columns (e.g. ``stylometric__sent_long_ratio``) are covered too.
RATIO_SUFFIXES = (
    "_ratio", "_share", "_proportion", "_density", "_per_token",
)

# "ratio"-named features that are *not* bounded in [0, 1]. The root type-token
# ratio is the Guiraud index (types / sqrt(tokens)), which grows with vocabulary
# size and can exceed 1.
UNBOUNDED_RATIO_SUFFIXES = ("vocab_root_type_token_ratio",)


def _check(check_id: str, description: str, actual: Any, expected: Any, status: str,
           details: str = "") -> dict[str, Any]:
    return {
        "check": check_id,
        "description": description,
        "actual": actual,
        "expected": expected,
        "status": status,
        "details": details,
    }


def _read(path_like: str) -> pd.DataFrame | None:
    path = resolve(path_like)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _numeric_columns(frame: pd.DataFrame, id_column: str) -> list[str]:
    columns = []
    for column in frame.columns:
        if column == id_column:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
        else:
            coerced = pd.to_numeric(frame[column], errors="coerce")
            if coerced.notna().any():
                columns.append(column)
    return columns


def check_artifacts(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], dict[str, Any]]:
    outputs = config["outputs"]
    id_column = config["input"]["id_column"]
    pilot_path = resolve(config["input"]["pilot_csv"])
    pilot = pd.read_csv(pilot_path, dtype=str, keep_default_na=False)
    expected_samples = int(len(pilot))
    checks: list[dict[str, Any]] = []

    artifact_keys = {
        "stylometric": "stylometric",
        "writeprints": "writeprints",
        "gltr": "gltr",
        "watermark": "watermark",
        "fused": "fused",
        "fused_metadata": "fused_metadata",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, key in artifact_keys.items():
        table = _read(outputs[key])
        if table is None:
            checks.append(_check(
                f"artifact::{name}", "Artifact exists", "missing", "present", FAIL,
                f"expected at {outputs[key]}",
            ))
            continue
        tables[name] = table
        checks.append(_check(
            f"artifact::{name}", "Artifact exists", "present", "present", PASS,
            f"{len(table)} rows x {len(table.columns)} columns",
        ))

    for name in ("stylometric", "writeprints", "watermark", "fused", "fused_metadata"):
        table = tables.get(name)
        if table is None:
            continue
        rows = int(len(table))
        checks.append(_check(
            f"row_count::{name}",
            "One row per pilot sample",
            rows,
            expected_samples,
            PASS if rows == expected_samples else FAIL,
        ))
        duplicates = int(table[id_column].duplicated().sum()) if id_column in table.columns else -1
        checks.append(_check(
            f"unique_sample_id::{name}",
            "sample_id is unique (duplicate detection)",
            duplicates,
            0,
            PASS if duplicates == 0 else FAIL,
        ))

    # Sample sets aligned across branches and fused tables.
    if "fused" in tables:
        fused_ids = set(tables["fused"][id_column])
        for name in ("stylometric", "writeprints", "watermark", "fused_metadata"):
            table = tables.get(name)
            if table is None or id_column not in table.columns:
                continue
            ids = set(table[id_column])
            checks.append(_check(
                f"sample_set_alignment::{name}",
                "Sample ids match the fused table",
                {"only_here": len(ids - fused_ids), "only_fused": len(fused_ids - ids)},
                {"only_here": 0, "only_fused": 0},
                PASS if ids == fused_ids else FAIL,
            ))
    extras = {"expected_samples": expected_samples}
    return checks, tables, extras


def check_schema(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    id_column = config["input"]["id_column"]
    schema_path = resolve(config["outputs"]["schema"])
    if not schema_path.exists():
        return [_check("schema::present", "fingerprint_feature_schema.json exists", "missing", "present", FAIL)]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    fused = tables.get("fused")
    if fused is not None:
        declared = list(schema.get("features", []))
        actual = [c for c in fused.columns if c != id_column]
        missing = [c for c in declared if c not in actual]
        extra = [c for c in actual if c not in declared]
        checks.append(_check(
            "schema::feature_columns",
            "Fused feature columns match the schema exactly",
            {"n_columns": len(actual), "missing": missing, "extra": extra},
            {"n_columns": len(declared), "missing": [], "extra": []},
            PASS if (not missing and not extra) else FAIL,
        ))
        checks.append(_check(
            "schema::feature_count",
            "Schema feature_count matches the fused matrix",
            int(schema.get("feature_count", -1)),
            len(actual),
            PASS if int(schema.get("feature_count", -1)) == len(actual) else FAIL,
        ))

    for key in ("feature_set_version", "fusion_version", "id_column", "null_policy",
                "branch_availability", "experiments", "leakage_note"):
        present = key in schema
        checks.append(_check(
            f"schema::key::{key}", f"Schema declares '{key}'", present, True,
            PASS if present else FAIL,
        ))
    return checks


def check_leakage(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    id_column = config["input"]["id_column"]
    identifier_columns = set(config.get("fusion", {}).get("identifier_columns", ["sample_id"]))
    metadata_columns = set(config.get("fusion", {}).get("metadata_columns", []))

    fused = tables.get("fused")
    if fused is not None:
        feature_columns = set(c for c in fused.columns if c != id_column)
        leaked_ids = sorted(feature_columns & identifier_columns)
        leaked_md = sorted(feature_columns & metadata_columns)
        checks.append(_check(
            "leakage::identifiers", "No identifier column is a feature",
            leaked_ids, [], PASS if not leaked_ids else FAIL,
        ))
        checks.append(_check(
            "leakage::metadata", "No metadata/target column is a feature",
            leaked_md, [], PASS if not leaked_md else FAIL,
        ))

    fused_metadata = tables.get("fused_metadata")
    if fused_metadata is not None:
        present = sorted(metadata_columns & set(fused_metadata.columns))
        checks.append(_check(
            "leakage::metadata_preserved",
            "Metadata columns are preserved separately for joins/stratification",
            len(present), len(metadata_columns),
            PASS if len(present) == len(metadata_columns) else WARN,
        ))
    return checks


def check_values_and_missing(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    id_column = config["input"]["id_column"]
    max_allowed = float(config.get("validation", {}).get("max_missing_fraction", 0.30))
    stats: dict[str, Any] = {}

    for name in ("stylometric", "writeprints", "gltr", "watermark", "fused"):
        table = tables.get(name)
        if table is None:
            continue
        numeric = _numeric_columns(table, id_column)
        missing = table[numeric].isna().mean() if numeric else pd.Series(dtype=float)
        stats[name] = {
            "rows": int(len(table)),
            "feature_columns": len(numeric),
            "overall_missing_fraction": float(missing.mean()) if len(missing) else 0.0,
            "max_missing_fraction": float(missing.max()) if len(missing) else 0.0,
        }

        # The missing-value threshold is a gate on the ML-facing fused matrix
        # only; per-branch missingness is reported as information because
        # undefined features (e.g. MSTTR below 50 words, sentence spread for a
        # single sentence) are legitimately empty and never imputed.
        if name == "fused":
            over = {c: float(missing[c]) for c in numeric if missing[c] > max_allowed}
            checks.append(_check(
                f"missing::threshold::{name}",
                "Feature columns stay within the configured missing fraction",
                {"over_threshold": over, "max": stats[name]["max_missing_fraction"]},
                {"over_threshold": {}, "max_max": max_allowed},
                PASS if not over else FAIL,
            ))

        non_finite: dict[str, int] = {}
        negative: dict[str, int] = {}
        out_of_range: dict[str, int] = {}
        for column in numeric:
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            n_inf = int((~finite & ~np.isnan(values)).sum())
            if n_inf:
                non_finite[column] = n_inf
            valid = values[finite]
            signed = column.endswith(SIGNED_SUFFIXES)
            if valid.size and not signed and float(valid.min()) < -1e-9:
                negative[column] = int((valid < 0).sum())
            is_ratio = column.endswith(RATIO_SUFFIXES) and not column.endswith(UNBOUNDED_RATIO_SUFFIXES)
            if valid.size and is_ratio:
                bad = int(((valid < -1e-9) | (valid > 1 + 1e-9)).sum())
                if bad:
                    out_of_range[column] = bad
        stats[name].update({
            "non_finite": non_finite, "negative": negative, "out_of_range": out_of_range,
        })
        clean = not (non_finite or negative or out_of_range)
        checks.append(_check(
            f"values::{name}",
            "No infinities, negative counts or out-of-range ratios",
            {"non_finite": non_finite, "negative": negative, "out_of_range": out_of_range},
            {"all_empty": True},
            PASS if clean else FAIL,
        ))
    return checks, stats


def check_watermark_semantics(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    table = tables.get("watermark")
    if table is None:
        return [_check("watermark::present", "watermark_results.csv exists", "missing", "present", FAIL)]

    if "watermark_status" not in table.columns:
        return [_check("watermark::status_column", "watermark_status column present", "missing", "present", FAIL)]

    statuses = table["watermark_status"].astype(str)
    if "watermark_detected" in table.columns:
        detected = table["watermark_detected"]
        # An unsupported/inconclusive row must never be encoded as a negative class.
        bad = table.loc[
            statuses.isin(["unsupported", "inconclusive"]) & detected.notna()
        ]
        checks.append(_check(
            "watermark::no_false_negative_class",
            "unsupported/inconclusive rows are not encoded as detected=False",
            int(len(bad)), 0, PASS if bad.empty else FAIL,
            "An unknown scheme must not become an artificial negative class.",
        ))
    checks.append(_check(
        "watermark::statuses_valid",
        "Only the four declared statuses are used",
        sorted(set(statuses)),
        ["detected", "inconclusive", "not_detected", "unsupported"],
        PASS if set(statuses) <= {"detected", "not_detected", "unsupported", "inconclusive"} else FAIL,
    ))
    return checks


def check_gltr(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    report_path = resolve(config["outputs"]["report_json"])
    gltr_available = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        gltr_available = report.get("modules", {}).get("gltr", {}).get("available")

    table = tables.get("gltr")
    if table is not None:
        id_column = config["input"]["id_column"]
        feature_columns = [c for c in table.columns if c != id_column]
        has_values = bool(feature_columns) and table[feature_columns].notna().any().any()
        checks.append(_check(
            "gltr::availability_consistent",
            "GLTR table has values iff the report says it is available",
            {"report_available": gltr_available, "table_has_values": has_values},
            {"consistent": True},
            PASS if bool(gltr_available) == has_values else FAIL,
        ))
    return checks


def summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for check in checks:
        counts[check["status"]] += 1
    status = FAIL if counts[FAIL] else (WARN if counts[WARN] else PASS)
    return {"status": status, "counts": counts, "total": len(checks)}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Fingerprint feature validation report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Feature set version: `{report['fingerprint_version']}`",
        f"- Overall status: **{summary['status'].upper()}**",
        f"- Checks: {summary['counts']['pass']} pass, {summary['counts']['warn']} warn, "
        f"{summary['counts']['fail']} fail (of {summary['total']})",
        "",
        "## Missing-value statistics",
        "",
        "| Branch | Rows | Feature columns | Overall missing | Max column missing |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, info in report["missing"].items():
        lines.append(
            f"| {name} | {info['rows']} | {info['feature_columns']} | "
            f"{info['overall_missing_fraction']:.3f} | {info['max_missing_fraction']:.3f} |"
        )
    lines += ["", "## Checks", "", "| Check | Status | Actual | Expected |", "|---|---|---|---|"]
    for check in report["checks"]:
        lines.append(
            f"| `{check['check']}` | {check['status']} | "
            f"`{json.dumps(check['actual'], default=str)}` | "
            f"`{json.dumps(check['expected'], default=str)}` |"
        )
    lines += [
        "",
        "## Method",
        "",
        "Read-only. The validator never writes to Neo4j, the datasets or the existing "
        "graph artifacts. Undefined features are emitted as NaN and never imputed; the "
        "missing-value check therefore reports fractions rather than filling them.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="fingerprint_config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    artifact_checks, tables, extras = check_artifacts(config)
    checks = list(artifact_checks)
    checks += check_schema(config, tables)
    checks += check_leakage(config, tables)
    value_checks, missing_stats = check_values_and_missing(config, tables)
    checks += value_checks
    checks += check_watermark_semantics(config, tables)
    checks += check_gltr(config, tables)

    summary = summarize(checks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint_version": config["fingerprint_version"],
        "summary": summary,
        "expected_samples": extras["expected_samples"],
        "missing": missing_stats,
        "checks": checks,
        "read_only": True,
    }
    report_dir = resolve(config["outputs"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "fingerprint_validation_report.json"
    md_path = report_dir / "fingerprint_validation_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Overall: {summary['status'].upper()} "
          f"({summary['counts']['pass']} pass, {summary['counts']['warn']} warn, "
          f"{summary['counts']['fail']} fail)")
    print(f"Report: {json_path}")
    print(f"Report: {md_path}")
    return 0 if summary["status"] != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
