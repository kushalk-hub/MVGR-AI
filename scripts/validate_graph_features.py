"""Validate the extracted graph feature tables for the MVGR-AI pilot build.

Phase F of the graph feature pipeline. This script is read-only: it never writes
to Neo4j and never mutates the feature tables. It checks the four artifacts
produced by ``scripts/extract_graph_features.py``:

* ``data/processed/graph_features.csv``            one row per graph view
* ``data/processed/sample_multiview_features.csv`` one row per sample
* ``data/processed/node_features.csv``             one row per token node
* ``data/processed/feature_schema.json``

and, when a database is reachable, cross-checks the feature totals against the
live Neo4j counts (read-only ``MATCH`` statements only).

Checks performed
----------------
1. Sample coverage: every expected sample has features, both views present.
2. Schema conformance: CSV columns match ``feature_schema.json`` exactly.
3. Duplicates: no duplicate ``graph_id``, ``(sample_id, view)`` or
   ``(graph_id, node_id)``.
4. Missing values: per-column NaN counts/fractions (undefined features are
   emitted as NaN, never imputed).
5. Invalid values: no infinities, no negative counts, ratios inside [0, 1],
   binary flags inside {0, 1}.
6. Multi-view association: exactly one row per view per sample, both views for
   every sample.
7. Aggregation consistency: the per-sample table agrees with the per-view table.
8. Leakage: identifiers/target metadata are never predictive features.
9. Database alignment: feature totals match the live Neo4j counts.

Outputs
-------
``outputs/graph_audit/feature_extraction_report.json``
``outputs/graph_audit/feature_extraction_report.md``

Usage
-----
    python scripts/validate_graph_features.py
    python scripts/validate_graph_features.py --skip-db
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow `python scripts/validate_graph_features.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.paths import load_config, resolve  # noqa: E402

PASS = "pass"
WARN = "warn"
FAIL = "fail"

# Columns that legitimately carry a sign (per-view deltas).
SIGNED_COLUMNS = {
    "node_count_diff",
    "edge_count_diff",
    "density_diff",
    "avg_degree_diff",
    "avg_clustering_diff",
    "n_components_diff",
    "largest_cc_ratio_diff",
    "avg_shortest_path_diff",
    "token_count_diff",
    "unique_token_diff",
    "avg_token_length_diff",
    "vocabulary_diversity_diff",
    "stopword_ratio_diff",
    "punctuation_ratio_diff",
    "pos_entropy_diff",
}

# Columns that must lie inside [0, 1].
RATIO_COLUMNS = {
    "density",
    "largest_component_ratio",
    "isolated_ratio",
    "vocabulary_diversity",
    "repeated_token_ratio",
    "stopword_ratio",
    "punctuation_ratio",
    "entity_token_ratio",
    "root_ratio",
    "position_ratio",
    "density_lexical",
    "density_syntactic",
    "largest_cc_ratio_lexical",
    "largest_cc_ratio_syntactic",
    "structural_cosine_similarity",
    "degree_dist_cosine_similarity",
    "degree_dist_js_similarity",
}

# Boolean flags that must be exactly 0 or 1.
BINARY_COLUMNS = {
    "is_directed",
    "is_punct",
    "is_stop",
    "is_root",
    "is_repeated",
    "is_first_token",
    "is_last_token",
}

RATIO_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    coerced = pd.to_numeric(series, errors="coerce")
    # Only treat as numeric if at least one value parsed and all non-null did.
    if coerced.notna().any() and (coerced.isna() == series.isna()).all():
        return coerced
    return None


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    outputs = config["outputs"]
    return {
        "graph_features": pd.read_csv(resolve(outputs["graph_features"])),
        "sample_multiview_features": pd.read_csv(resolve(outputs["sample_multiview_features"])),
        "node_features": pd.read_csv(resolve(outputs["node_features"])),
    }


def load_schema(config: dict[str, Any]) -> dict[str, Any]:
    path = resolve(config["outputs"]["feature_schema"])
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# checks (pure: no database access)
# --------------------------------------------------------------------------- #
def check_schema_conformance(tables: dict[str, pd.DataFrame],
                             schema: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name, table in tables.items():
        declared = [c["name"] for c in schema["tables"][table_name]["columns"]]
        actual = list(table.columns)
        missing = [c for c in declared if c not in actual]
        extra = [c for c in actual if c not in declared]
        ordered = declared == actual
        status = PASS if (not missing and not extra and ordered) else FAIL
        checks.append(_check(
            f"schema_columns::{table_name}",
            "CSV columns match the declared schema exactly (names and order)",
            {"n_columns": len(actual), "ordered": ordered},
            {"n_columns": len(declared), "ordered": True},
            status,
            f"missing={missing} extra={extra}",
        ))
    for key in ("feature_set_version", "split_policy_version", "views", "null_policy",
                "leakage_policy", "tables"):
        present = key in schema
        checks.append(_check(
            f"schema_key::{key}",
            f"feature_schema.json declares '{key}'",
            present, True, PASS if present else FAIL,
        ))
    return checks


def check_coverage(tables: dict[str, pd.DataFrame], config: dict[str, Any]) -> list[dict[str, Any]]:
    graph = tables["graph_features"]
    multiview = tables["sample_multiview_features"]
    node = tables["node_features"]
    g = config["graph"]
    views = g["views"]
    expected_views = {views["lexical"], views["syntactic"]}
    checks: list[dict[str, Any]] = []

    n_graph_samples = int(graph["sample_id"].nunique())
    checks.append(_check(
        "sample_coverage::graph_features",
        "Every expected sample has at least one graph view row",
        n_graph_samples, g["expected_samples"],
        PASS if n_graph_samples == g["expected_samples"] else FAIL,
    ))
    checks.append(_check(
        "row_count::graph_features",
        "Graph view rows equal expected_graphs (one row per view)",
        int(len(graph)), g["expected_graphs"],
        PASS if len(graph) == g["expected_graphs"] else FAIL,
    ))
    checks.append(_check(
        "row_count::sample_multiview_features",
        "Multiview rows equal expected_samples (one row per sample)",
        int(len(multiview)), g["expected_samples"],
        PASS if len(multiview) == g["expected_samples"] else FAIL,
    ))
    expected_nodes = int(g["expected_nodes"]) - int(g["expected_graphs"]) - int(g["expected_samples"])
    checks.append(_check(
        "row_count::node_features",
        "Node rows equal the expected token-node count",
        int(len(node)), expected_nodes,
        PASS if len(node) == expected_nodes else FAIL,
    ))

    # Both views per sample.
    per_sample_views = graph.groupby("sample_id")["view"].nunique()
    missing = int((per_sample_views != 2).sum())
    checks.append(_check(
        "two_views_per_sample",
        "Every sample carries exactly two distinct graph views",
        missing, 0, PASS if missing == 0 else FAIL,
    ))
    unexpected = sorted(set(graph["view"].unique()) - expected_views)
    checks.append(_check(
        "views_present",
        "Only the two configured views are present",
        sorted(set(graph["view"].unique())), sorted(expected_views),
        PASS if not unexpected else FAIL,
    ))

    # Sample sets agree across tables.
    graph_samples = set(graph["sample_id"])
    mv_samples = set(multiview["sample_id"])
    only_graph = len(graph_samples - mv_samples)
    only_mv = len(mv_samples - graph_samples)
    checks.append(_check(
        "sample_sets_aligned",
        "graph_features and sample_multiview_features cover the same samples",
        {"only_in_graph": only_graph, "only_in_multiview": only_mv},
        {"only_in_graph": 0, "only_in_multiview": 0},
        PASS if only_graph == 0 and only_mv == 0 else FAIL,
    ))
    return checks


def check_duplicates(tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    graph = tables["graph_features"]
    multiview = tables["sample_multiview_features"]
    node = tables["node_features"]
    checks: list[dict[str, Any]] = []

    dup_graph_id = int(graph["graph_id"].duplicated().sum())
    checks.append(_check("unique_graph_id", "graph_id is unique in graph_features",
                         dup_graph_id, 0, PASS if dup_graph_id == 0 else FAIL))

    dup_sample_view = int(graph.duplicated(subset=["sample_id", "view"]).sum())
    checks.append(_check("unique_sample_view", "No duplicate (sample_id, view) pair",
                         dup_sample_view, 0, PASS if dup_sample_view == 0 else FAIL))

    dup_mv = int(multiview["sample_id"].duplicated().sum())
    checks.append(_check("unique_multiview_sample", "sample_id is unique in the multiview table",
                         dup_mv, 0, PASS if dup_mv == 0 else FAIL))

    dup_node = int(node.duplicated(subset=["graph_id", "node_id"]).sum())
    checks.append(_check("unique_graph_node", "No duplicate (graph_id, node_id) pair",
                         dup_node, 0, PASS if dup_node == 0 else FAIL))
    return checks


def check_missing(tables: dict[str, pd.DataFrame], schema: dict[str, Any],
                  config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (checks, per-table missing-value statistics).

    Two gates are applied:

    * no *predictive* feature column may be entirely null (a dead feature);
    * on the ML-facing ``sample_multiview_features`` table every predictive
      column must stay within the configured missing fraction.

    The per-view ``graph_features`` table legitimately holds nulls for
    view-specific features on the other view's rows, and metadata columns may be
    null in the source, so those are reported but not gated.
    """
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    max_allowed = float(config.get("validation", {}).get("max_missing_fraction", 0.30))
    for table_name, table in tables.items():
        missing = table.isna().sum()
        fractions = (missing / max(len(table), 1)).round(6)
        offenders = {c: float(fractions[c]) for c in table.columns if missing[c] > 0}
        predictive = {c["name"] for c in schema["tables"][table_name]["columns"]
                      if c.get("predictive")}
        dead = sorted(c for c in predictive if c in table.columns and missing[c] == len(table))
        stats[table_name] = {
            "rows": int(len(table)),
            "columns": int(len(table.columns)),
            "total_missing_cells": int(missing.sum()),
            "columns_with_missing": offenders,
            "predictive_columns_entirely_missing": dead,
        }
        checks.append(_check(
            f"missing_values::dead_features::{table_name}",
            "No predictive feature column is entirely missing",
            dead, [], PASS if not dead else FAIL,
        ))

        if table_name == "sample_multiview_features":
            worst = max((offenders.get(c, 0.0) for c in predictive), default=0.0)
            over = {c: offenders[c] for c in predictive if offenders.get(c, 0.0) > max_allowed}
            checks.append(_check(
                f"missing_values::threshold::{table_name}",
                "Predictive columns stay within the configured missing fraction",
                {"max_fraction": worst, "over_threshold": over},
                {"max_fraction_max": max_allowed},
                PASS if not over else FAIL,
                "undefined features are emitted as NaN and never imputed",
            ))
    return checks, stats


def check_invalid_values(tables: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (checks, per-table invalid-value statistics)."""
    checks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for table_name, table in tables.items():
        non_finite: dict[str, int] = {}
        negative: dict[str, int] = {}
        out_of_range: dict[str, int] = {}
        non_binary: dict[str, int] = {}
        for column in table.columns:
            series = _numeric(table, column)
            if series is None:
                continue
            values = series.to_numpy(dtype=float)
            finite = np.isfinite(values)
            n_inf = int((~finite & ~np.isnan(values)).sum())
            if n_inf:
                non_finite[column] = n_inf
            valid = values[finite]
            if column not in SIGNED_COLUMNS and valid.size and float(valid.min()) < 0:
                negative[column] = int((valid < 0).sum())
            if column in RATIO_COLUMNS and valid.size:
                bad = int(((valid < -RATIO_TOLERANCE) | (valid > 1 + RATIO_TOLERANCE)).sum())
                if bad:
                    out_of_range[column] = bad
            if column in BINARY_COLUMNS and valid.size:
                bad = int((~np.isin(valid, (0.0, 1.0))).sum())
                if bad:
                    non_binary[column] = bad
        stats[table_name] = {
            "non_finite": non_finite,
            "negative": negative,
            "out_of_range": out_of_range,
            "non_binary": non_binary,
        }
        clean = not (non_finite or negative or out_of_range or non_binary)
        checks.append(_check(
            f"invalid_values::{table_name}",
            "No infinities, negative counts, out-of-range ratios or non-binary flags",
            {"non_finite": non_finite, "negative": negative,
             "out_of_range": out_of_range, "non_binary": non_binary},
            {"all_empty": True},
            PASS if clean else FAIL,
        ))
    return checks, stats


def check_aggregation(tables: dict[str, pd.DataFrame], config: dict[str, Any]) -> list[dict[str, Any]]:
    graph = tables["graph_features"]
    multiview = tables["sample_multiview_features"]
    views = config["graph"]["views"]
    checks: list[dict[str, Any]] = []

    lex = graph[graph["view"] == views["lexical"]].set_index("sample_id")
    syn = graph[graph["view"] == views["syntactic"]].set_index("sample_id")
    mv = multiview.set_index("sample_id")
    aligned = mv.index.intersection(lex.index).intersection(syn.index)

    pairs = (
        ("n_nodes_lexical", "n_nodes", lex),
        ("n_nodes_syntactic", "n_nodes", syn),
        ("n_edges_lexical", "n_edges", lex),
        ("n_edges_syntactic", "n_edges", syn),
    )
    for mv_col, graph_col, source in pairs:
        lhs = mv.loc[aligned, mv_col].to_numpy(dtype=float)
        rhs = source.loc[aligned, graph_col].to_numpy(dtype=float)
        mismatches = int(np.sum(~np.isclose(lhs, rhs)))
        checks.append(_check(
            f"aggregation::{mv_col}",
            f"multiview '{mv_col}' equals the per-view '{graph_col}'",
            mismatches, 0, PASS if mismatches == 0 else FAIL,
        ))

    diff = mv.loc[aligned, "n_nodes_lexical"].to_numpy(float) - \
        mv.loc[aligned, "n_nodes_syntactic"].to_numpy(float)
    stated = mv.loc[aligned, "node_count_diff"].to_numpy(float)
    mismatches = int(np.sum(~np.isclose(diff, stated)))
    checks.append(_check(
        "aggregation::node_count_diff",
        "multiview 'node_count_diff' equals lexical minus syntactic node count",
        mismatches, 0, PASS if mismatches == 0 else FAIL,
    ))

    # Cross-view token-content deltas are a known structural property of this
    # build (both views duplicate the identical token set), so they are expected
    # to be exactly zero. Reported as information, not as a hard failure.
    zero_diff_cols = [
        "token_count_diff", "unique_token_diff", "avg_token_length_diff",
        "vocabulary_diversity_diff", "stopword_ratio_diff",
        "punctuation_ratio_diff", "pos_entropy_diff",
    ]
    nonzero = {
        c: int((~np.isclose(mv[c].to_numpy(float), 0.0)).sum())
        for c in zero_diff_cols if c in mv.columns
    }
    all_zero = all(v == 0 for v in nonzero.values())
    checks.append(_check(
        "cross_view_token_deltas",
        "Token-content cross-view deltas are zero (both views share the token set)",
        nonzero, {c: 0 for c in nonzero},
        PASS if all_zero else WARN,
        "Structural property of the pilot build, documented in the final report.",
    ))
    return checks


def check_leakage(tables: dict[str, pd.DataFrame], schema: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name, spec in schema["tables"].items():
        features = {c["name"] for c in spec["columns"] if c.get("predictive")}
        identifiers = {c["name"] for c in spec["columns"] if c.get("leakage_role") == "identifier"}
        metadata = {c["name"] for c in spec["columns"] if c.get("leakage_role") == "metadata"}
        leaked_id = sorted(features & identifiers)
        leaked_md = sorted(features & metadata)
        checks.append(_check(
            f"leakage_identifiers::{table_name}",
            "No identifier column is marked predictive",
            leaked_id, [], PASS if not leaked_id else FAIL,
        ))
        checks.append(_check(
            f"leakage_metadata::{table_name}",
            "No target/metadata column is marked predictive",
            leaked_md, [], PASS if not leaked_md else FAIL,
        ))
    policy = schema.get("leakage_policy", {})
    ok = bool(policy.get("require_approved_manifest_for_formal_split", False))
    checks.append(_check(
        "leakage_policy::formal_split_gate",
        "The schema requires an approved manifest for a formal split",
        ok, True, PASS if ok else FAIL,
    ))
    return checks


def run_file_checks(tables: dict[str, pd.DataFrame], schema: dict[str, Any],
                    config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks += check_schema_conformance(tables, schema)
    checks += check_coverage(tables, config)
    checks += check_duplicates(tables)
    missing_checks, missing_stats = check_missing(tables, schema, config)
    checks += missing_checks
    invalid_checks, invalid_stats = check_invalid_values(tables)
    checks += invalid_checks
    checks += check_aggregation(tables, config)
    checks += check_leakage(tables, schema)
    extras = {"missing_values": missing_stats, "invalid_values": invalid_stats}
    return checks, extras


# --------------------------------------------------------------------------- #
# database cross-check (read-only)
# --------------------------------------------------------------------------- #
def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def collect_db_counts(config: dict[str, Any]) -> dict[str, int]:
    from neo4j import GraphDatabase

    neo = config["neo4j"]
    load_dotenv_file(resolve(neo.get("dotenv_file", ".env")))
    uri = os.environ.get(neo["uri_env"], "")
    user = os.environ.get(neo["user_env"], "")
    password = os.environ.get(neo["password_env"], "")
    database = os.environ.get(neo["database_env"], neo["database_default"])
    if not (uri and user and password):
        raise RuntimeError("missing Neo4j settings")

    g = config["graph"]
    rel = g["relationships"]
    driver = GraphDatabase.driver(uri, auth=(user, password),
                                  connection_timeout=float(neo.get("connection_timeout_seconds", 30)))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            def scalar(query: str, **params) -> int:
                record = session.run(query, **params).single()
                return int(record[0]) if record and record[0] is not None else 0

            counts = {
                "samples": scalar(f"MATCH (s:{g['sample_label']}) RETURN count(s)"),
                "graphs": scalar(f"MATCH (g:{g['graph_label']}) RETURN count(g)"),
                "tokens": scalar(f"MATCH (t:{g['token_label']}) RETURN count(t)"),
                "has_graph": scalar(f"MATCH ()-[r:{rel['sample_to_graph']}]->() RETURN count(r)"),
                "has_token": scalar(f"MATCH ()-[r:{rel['graph_to_token']}]->() RETURN count(r)"),
                "co_occurs": scalar(f"MATCH ()-[r:{rel['lexical_edge']}]->() RETURN count(r)"),
                "depends_on": scalar(f"MATCH ()-[r:{rel['syntactic_edge']}]->() RETURN count(r)"),
            }
    finally:
        driver.close()
    return counts


def check_db_alignment(db: dict[str, int], tables: dict[str, pd.DataFrame],
                       config: dict[str, Any]) -> list[dict[str, Any]]:
    graph = tables["graph_features"]
    node = tables["node_features"]
    g = config["graph"]
    views = g["views"]
    checks: list[dict[str, Any]] = []

    def eq(check_id: str, description: str, actual: Any, expected: Any) -> None:
        checks.append(_check(check_id, description, actual, expected,
                             PASS if actual == expected else FAIL))

    eq("db::samples", "Live Sample count matches expected", db["samples"], g["expected_samples"])
    eq("db::graphs", "Live TextGraph count matches expected", db["graphs"], g["expected_graphs"])
    eq("db::tokens", "Live TokenOccurrence count matches the node feature rows",
       db["tokens"], int(len(node)))
    eq("db::has_graph", "Live HAS_GRAPH count matches the graph count",
       db["has_graph"], g["expected_graphs"])
    eq("db::has_token", "Live HAS_TOKEN count matches the token count",
       db["has_token"], db["tokens"])

    lex_rows = graph[graph["view"] == views["lexical"]]
    syn_rows = graph[graph["view"] == views["syntactic"]]
    eq("db::co_occurs", "Live CO_OCCURS count matches lexical edge features",
       db["co_occurs"], int(lex_rows["n_edges"].sum()))
    eq("db::depends_on", "Live DEPENDS_ON count matches syntactic edge features",
       db["depends_on"], int(syn_rows["n_edges"].sum()))
    eq("db::lexical_edges_expected", "Lexical edge total matches the pilot manifest",
       db["co_occurs"], g["expected_lexical_edges"])
    eq("db::syntactic_edges_expected", "Syntactic edge total matches the pilot manifest",
       db["depends_on"], g["expected_syntactic_edges"])
    eq("db::node_features_per_view", "Token feature rows equal the sum of per-view node counts",
       int(len(node)), int(graph["n_nodes"].sum()))
    return checks


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for check in checks:
        counts[check["status"]] += 1
    status = FAIL if counts[FAIL] else (WARN if counts[WARN] else PASS)
    return {"status": status, "counts": counts, "total": len(checks)}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Graph feature extraction validation report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Build ID: `{report['build_id']}`",
        f"- Feature set version: `{report['feature_set_version']}`",
        f"- Split policy version: `{report['split_policy_version']}`",
        f"- Overall status: **{summary['status'].upper()}**",
        f"- Checks: {summary['counts']['pass']} pass, {summary['counts']['warn']} warn, "
        f"{summary['counts']['fail']} fail (of {summary['total']})",
        f"- Database cross-check: {'performed' if report['database_checked'] else 'skipped'}",
        "",
        "## Output files",
        "",
        "| Artifact | Path | Rows | Columns |",
        "|---|---|---:|---:|",
    ]
    for name, info in report["file_stats"].items():
        lines.append(f"| `{name}` | `{info['path']}` | {info['rows']:,} | {info['columns']} |")

    lines += [
        "",
        "## Missing values",
        "",
        "| Table | Rows | Missing cells | Columns with missing |",
        "|---|---:|---:|---:|",
    ]
    for name, info in report["missing_values"].items():
        lines.append(
            f"| `{name}` | {info['rows']:,} | {info['total_missing_cells']:,} | "
            f"{len(info['columns_with_missing'])} |"
        )

    if report["database_checked"]:
        lines += ["", "## Live Neo4j counts (read-only)", "", "| Metric | Value |", "|---|---:|"]
        for key, value in report["db_counts"].items():
            lines.append(f"| `{key}` | {value:,} |")

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Status | Actual | Expected |",
        "|---|---|---|---|",
    ]
    for check in report["checks"]:
        actual = check["actual"]
        expected = check["expected"]
        lines.append(
            f"| `{check['check']}` | {check['status']} | `{json.dumps(actual, default=str)}` | "
            f"`{json.dumps(expected, default=str)}` |"
        )

    lines += [
        "",
        "## Method",
        "",
        "The validator reads the four feature artifacts and performs no writes. "
        "When a database is reachable it issues only read-only `MATCH`/`RETURN` "
        "statements to cross-check the feature totals against the live graph. "
        "Undefined features are emitted as NaN (never imputed); the missing-value "
        "check therefore reports counts rather than silently filling them.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], config: dict[str, Any]) -> tuple[Path, Path]:
    json_path = resolve(config["outputs"]["feature_report_json"])
    md_path = resolve(config["outputs"]["feature_report_md"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="graph_feature_config")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip the live Neo4j cross-check")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    tables = load_tables(config)
    schema = load_schema(config)

    checks, extras = run_file_checks(tables, schema, config)

    db_counts: dict[str, int] = {}
    database_checked = False
    if not args.skip_db:
        try:
            db_counts = collect_db_counts(config)
            database_checked = True
            checks += check_db_alignment(db_counts, tables, config)
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            checks.append(_check(
                "db::connectivity",
                "Live Neo4j cross-check could not run",
                f"{type(exc).__name__}: {exc}", "reachable", WARN,
            ))

    summary = summarize(checks)
    file_stats = {
        name: {
            "path": config["outputs"][name],
            "rows": int(len(table)),
            "columns": int(len(table.columns)),
        }
        for name, table in tables.items()
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build_id": schema.get("build_id"),
        "feature_set_version": schema.get("feature_set_version"),
        "split_policy_version": schema.get("split_policy_version"),
        "summary": summary,
        "file_stats": file_stats,
        "missing_values": extras["missing_values"],
        "invalid_values": extras["invalid_values"],
        "database_checked": database_checked,
        "db_counts": db_counts,
        "checks": checks,
        "read_only": True,
    }
    json_path, md_path = write_reports(report, config)

    print(f"Overall: {summary['status'].upper()} "
          f"({summary['counts']['pass']} pass, {summary['counts']['warn']} warn, "
          f"{summary['counts']['fail']} fail)")
    for name, info in file_stats.items():
        print(f"  {name}: rows={info['rows']:,} cols={info['columns']}")
    if database_checked:
        print(f"  db counts: {db_counts}")
    print(f"Report: {json_path}")
    print(f"Report: {md_path}")
    return 0 if summary["status"] != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
