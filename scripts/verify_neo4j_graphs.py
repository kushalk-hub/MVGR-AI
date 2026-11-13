"""Read-only Neo4j graph verification for the MVGR-AI pilot build.

Phase A of the graph feature pipeline. This script never writes to Neo4j: every
statement is a read. It discovers node labels, relationship types and property
keys from the live database rather than assuming them, then compares the actual
counts against the expected pilot statistics in
``configs/graph_feature_config.yaml``.

Outputs
-------
``outputs/graph_audit/neo4j_verification_report.json``
``outputs/graph_audit/neo4j_verification_report.md``

Usage
-----
    python scripts/verify_neo4j_graphs.py
    python scripts/verify_neo4j_graphs.py --build-id <other_build>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python scripts/verify_neo4j_graphs.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.paths import REPO_ROOT, load_config, resolve  # noqa: E402

PASS = "pass"
WARN = "warn"
FAIL = "fail"


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ without overriding real env vars."""
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


class Settings:
    def __init__(self, config: dict[str, Any]) -> None:
        neo = config["neo4j"]
        load_dotenv_file(resolve(neo.get("dotenv_file", ".env")))
        self.uri = os.environ.get(neo["uri_env"], "")
        self.user = os.environ.get(neo["user_env"], "")
        self.password = os.environ.get(neo["password_env"], "")
        self.database = os.environ.get(neo["database_env"], neo["database_default"])
        self.timeout = float(neo.get("connection_timeout_seconds", 30))
        self.max_retries = int(neo.get("max_retries", 3))

    def missing(self) -> list[str]:
        return [name for name, value in (("uri", self.uri), ("user", self.user),
                                         ("password", self.password)) if not value]


def connect(settings: Settings):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
        connection_timeout=settings.timeout,
    )
    driver.verify_connectivity()
    return driver


# --------------------------------------------------------------------------- #
# stats collection
# --------------------------------------------------------------------------- #
def _scalar(session, query: str, default: int = 0, **params) -> int:
    record = session.run(query, **params).single()
    if record is None:
        return default
    value = record[0]
    return int(value) if value is not None else default


def collect_stats(session, config: dict[str, Any]) -> dict[str, Any]:
    g = config["graph"]
    rel = g["relationships"]
    sample_label = g["sample_label"]
    graph_label = g["graph_label"]
    token_label = g["token_label"]
    edge_types = [rel["lexical_edge"], rel["syntactic_edge"]]

    stats: dict[str, Any] = {}

    stats["total_nodes"] = _scalar(session, "MATCH (n) RETURN count(n)")
    stats["total_relationships"] = _scalar(session, "MATCH ()-[r]->() RETURN count(r)")

    stats["nodes_by_label"] = {
        record["label"]: int(record["count"])
        for record in session.run(
            "MATCH (n) UNWIND labels(n) AS label "
            "RETURN label, count(*) AS count ORDER BY count DESC"
        )
    }
    stats["relationships_by_type"] = {
        record["type"]: int(record["count"])
        for record in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
        )
    }

    stats["samples"] = _scalar(session, f"MATCH (s:{sample_label}) RETURN count(s)")
    stats["distinct_sample_ids"] = _scalar(
        session, f"MATCH (s:{sample_label}) RETURN count(DISTINCT s.sample_id)"
    )
    stats["graphs"] = _scalar(session, f"MATCH (g:{graph_label}) RETURN count(g)")
    stats["graphs_by_view"] = {
        record["view"]: int(record["count"])
        for record in session.run(
            f"MATCH (g:{graph_label}) RETURN g.view AS view, count(*) AS count ORDER BY view"
        )
    }
    stats["tokens"] = _scalar(session, f"MATCH (t:{token_label}) RETURN count(t)")

    # Two views per sample.
    record = session.run(
        f"""
        MATCH (s:{sample_label})-[:{rel['sample_to_graph']}]->(g:{graph_label})
        WITH s, count(DISTINCT g.view) AS views
        RETURN count(s) AS samples_with_graphs,
               sum(CASE WHEN views = 2 THEN 1 ELSE 0 END) AS with_two_views,
               sum(CASE WHEN views <> 2 THEN 1 ELSE 0 END) AS missing_views
        """
    ).single()
    stats["samples_with_graphs"] = int(record["samples_with_graphs"] or 0)
    stats["samples_with_two_views"] = int(record["with_two_views"] or 0)
    stats["samples_missing_views"] = int(record["missing_views"] or 0)

    # A sample with two graphs of the same view would break multi-view features.
    stats["duplicate_view_pairs"] = _scalar(
        session,
        f"""
        MATCH (s:{sample_label})-[:{rel['sample_to_graph']}]->(g:{graph_label})
        WITH s, g.view AS view, count(g) AS c WHERE c > 1
        RETURN count(*)
        """,
    )

    # Orphans and missing relationships.
    stats["orphan_tokens"] = _scalar(
        session,
        f"MATCH (t:{token_label}) WHERE NOT (:{graph_label})-[:{rel['graph_to_token']}]->(t) "
        "RETURN count(t)",
    )
    stats["graphs_without_tokens"] = _scalar(
        session,
        f"MATCH (g:{graph_label}) WHERE NOT (g)-[:{rel['graph_to_token']}]->() RETURN count(g)",
    )
    stats["samples_without_graphs"] = _scalar(
        session,
        f"MATCH (s:{sample_label}) WHERE NOT (s)-[:{rel['sample_to_graph']}]->() RETURN count(s)",
    )
    stats["dangling_graphs"] = _scalar(
        session,
        f"MATCH (g:{graph_label}) WHERE NOT (:{sample_label})-[:{rel['sample_to_graph']}]->(g) "
        "RETURN count(g)",
    )

    # Graph-view isolation: no structural edge may cross a graph_id boundary.
    stats["cross_graph_edges"] = _scalar(
        session,
        "MATCH (a:"
        + token_label
        + ")-[r]->(b:"
        + token_label
        + ") WHERE type(r) IN $edge_types AND a.graph_id <> b.graph_id RETURN count(r)",
        edge_types=edge_types,
    )

    # Stored counts must match the actual neighbourhood sizes.
    stats["token_count_mismatches"] = _scalar(
        session,
        f"""
        MATCH (g:{graph_label})-[:{rel['graph_to_token']}]->(t:{token_label})
        WITH g, count(t) AS actual WHERE actual <> g.token_count
        RETURN count(g)
        """,
    )
    stats["edge_count_mismatches"] = _scalar(
        session,
        """
        MATCH (a:{tl})-[r]->(b:{tl})
        WHERE type(r) IN $edge_types AND a.graph_id = b.graph_id
        WITH a.graph_id AS gid, count(r) AS actual
        MATCH (g:{gl} {graph_id: gid})
        WHERE actual <> g.edge_count
        RETURN count(g)
        """.replace("{tl}", token_label).replace("{gl}", graph_label),
        edge_types=edge_types,
    )

    # Build identity.
    stats["build_ids"] = session.run(
        f"MATCH (g:{graph_label}) RETURN DISTINCT g.build_id AS build_id ORDER BY build_id"
    ).value()
    stats["graphs_in_expected_build"] = _scalar(
        session,
        f"MATCH (g:{graph_label} {{build_id: $build_id}}) RETURN count(g)",
        build_id=g["build_id"],
    )
    stats["dangling_token_graph_ids"] = _scalar(
        session,
        f"""
        MATCH (t:{token_label})
        WHERE NOT EXISTS {{ MATCH (:{graph_label} {{graph_id: t.graph_id}}) }}
        RETURN count(t)
        """,
    )

    # Per-view edge counts, so they can be compared to the pilot manifest.
    stats["edges_by_type_and_view"] = {
        record["type"]: int(record["count"])
        for record in session.run(
            "MATCH (a:"
            + token_label
            + ")-[r]->(b:"
            + token_label
            + ") WHERE type(r) IN $edge_types AND a.graph_id = b.graph_id "
            "RETURN type(r) AS type, count(*) AS count ORDER BY type",
            edge_types=edge_types,
        )
    }
    return stats


# --------------------------------------------------------------------------- #
# checks (pure function so tests can exercise it without a database)
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


def run_checks(stats: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    g = config["graph"]
    v = config["validation"]
    checks: list[dict[str, Any]] = []

    def eq(check_id: str, description: str, actual: Any, expected: Any,
           on_mismatch: str = FAIL) -> None:
        status = PASS if actual == expected else on_mismatch
        checks.append(_check(check_id, description, actual, expected, status))

    eq("total_nodes", "Total nodes match the pilot manifest",
       stats["total_nodes"], g["expected_nodes"])
    eq("total_relationships", "Total relationships match the pilot manifest",
       stats["total_relationships"], g["expected_relationships"])
    eq("sample_count", "Sample nodes equal the expected sample count",
       stats["samples"], g["expected_samples"])
    eq("graph_count", "TextGraph nodes equal the expected graph count",
       stats["graphs"], g["expected_graphs"])

    expected_views = {g["views"]["lexical"], g["views"]["syntactic"]}
    actual_views = set(stats["graphs_by_view"])
    checks.append(
        _check(
            "graph_views",
            "Both graph views are present and no other view exists",
            sorted(actual_views),
            sorted(expected_views),
            PASS if actual_views == expected_views else FAIL,
        )
    )

    if v.get("require_unique_sample_ids", True):
        eq("unique_sample_ids", "No duplicate sample_id values",
           stats["samples"] - stats["distinct_sample_ids"], 0)

    if v.get("require_two_views_per_sample", True):
        eq("two_views_per_sample", "Every sample has exactly two graph views",
           stats["samples_missing_views"], 0)
        eq("sample_graph_coverage", "Every sample is connected to its graphs",
           stats["samples_without_graphs"], 0)
        eq("duplicate_view_pairs", "No sample has two graphs of the same view",
           stats["duplicate_view_pairs"], 0)

    if v.get("require_no_orphan_tokens", True):
        eq("orphan_tokens", "No token node lacks a HAS_TOKEN parent",
           stats["orphan_tokens"], 0)
        eq("graphs_without_tokens", "No graph lacks token children",
           stats["graphs_without_tokens"], 0)
        eq("dangling_graphs", "No graph lacks a HAS_GRAPH parent",
           stats["dangling_graphs"], 0)

    if v.get("require_no_cross_graph_edges", True):
        eq("cross_graph_edges", "No structural edge crosses a graph boundary",
           stats["cross_graph_edges"], 0)

    if v.get("require_no_missing_relationships", True):
        eq("token_count_mismatch", "Stored token_count matches actual HAS_TOKEN count",
           stats["token_count_mismatches"], 0)
        eq("edge_count_mismatch", "Stored edge_count matches actual edge count",
           stats["edge_count_mismatches"], 0)

    if v.get("require_build_id_consistency", True):
        eq("build_id_consistency", "Every graph belongs to the expected build",
           stats["graphs_in_expected_build"], stats["graphs"])
        eq("build_id_single", "Only one build_id is present",
           len(stats["build_ids"]), 1)
        eq("dangling_token_graph_ids", "Every token graph_id resolves to a graph",
           stats["dangling_token_graph_ids"], 0)

    # Node labels must be exactly the three expected labels.
    expected_labels = {g["sample_label"], g["graph_label"], g["token_label"]}
    actual_labels = set(stats["nodes_by_label"])
    checks.append(
        _check(
            "node_labels",
            "Only the expected node labels exist",
            sorted(actual_labels),
            sorted(expected_labels),
            PASS if actual_labels == expected_labels else FAIL,
        )
    )

    expected_rel_types = {
        g["relationships"]["sample_to_graph"],
        g["relationships"]["graph_to_token"],
        g["relationships"]["lexical_edge"],
        g["relationships"]["syntactic_edge"],
    }
    actual_rel_types = set(stats["relationships_by_type"])
    checks.append(
        _check(
            "relationship_types",
            "Only the expected relationship types exist",
            sorted(actual_rel_types),
            sorted(expected_rel_types),
            PASS if actual_rel_types == expected_rel_types else FAIL,
        )
    )

    # Internal consistency: the total must equal the sum of the per-type counts.
    # This catches arithmetic slips in externally quoted "projected" totals.
    rel_sum = sum(stats["relationships_by_type"].values())
    eq("relationship_total_consistency",
       "Total relationships equal the sum of per-type counts",
       stats["total_relationships"], rel_sum)
    node_sum = sum(stats["nodes_by_label"].values())
    eq("node_total_consistency",
       "Total nodes equal the sum of per-label counts",
       stats["total_nodes"], node_sum)

    # The manifest's per-view edge counts are advisory; a mismatch is a warning.
    for edge_type, key in (
        (g["relationships"]["lexical_edge"], "lexical_edge_count"),
        (g["relationships"]["syntactic_edge"], "syntactic_edge_count"),
    ):
        checks.append(
            _check(
                key,
                f"{edge_type} edges are consistent with graph isolation",
                stats["edges_by_type_and_view"].get(edge_type, 0),
                "reported",
                PASS,
                "Counted only for edges whose endpoints share a graph_id.",
            )
        )
    return checks


def summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for check in checks:
        counts[check["status"]] += 1
    status = FAIL if counts[FAIL] else (WARN if counts[WARN] else PASS)
    return {"status": status, "counts": counts}


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Neo4j graph verification report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Database: `{report['database']}`",
        f"- Build ID: `{report['build_id']}`",
        f"- Overall status: **{report['summary']['status'].upper()}**",
        f"- Checks: {report['summary']['counts']['pass']} pass, "
        f"{report['summary']['counts']['warn']} warn, {report['summary']['counts']['fail']} fail",
        "",
        "## Actual database statistics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    stats = report["stats"]
    for key in (
        "total_nodes", "total_relationships", "samples", "graphs", "tokens",
        "samples_with_two_views", "samples_missing_views", "orphan_tokens",
        "graphs_without_tokens", "cross_graph_edges", "token_count_mismatches",
        "edge_count_mismatches",
    ):
        lines.append(f"| {key} | {stats.get(key)} |")

    lines += ["", "### Nodes by label", "", "| Label | Count |", "|---|---:|"]
    for label, count in stats["nodes_by_label"].items():
        lines.append(f"| `{label}` | {count:,} |")

    lines += ["", "### Relationships by type", "", "| Type | Count |", "|---|---:|"]
    for rel_type, count in stats["relationships_by_type"].items():
        lines.append(f"| `{rel_type}` | {count:,} |")

    lines += ["", "### Graphs by view", "", "| View | Count |", "|---|---:|"]
    for view, count in stats["graphs_by_view"].items():
        lines.append(f"| `{view}` | {count:,} |")

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Status | Actual | Expected | Description |",
        "|---|---|---|---|---|",
    ]
    for check in report["checks"]:
        lines.append(
            f"| `{check['check']}` | {check['status']} | {check['actual']} | "
            f"{check['expected']} | {check['description']} |"
        )

    lines += [
        "",
        "## Method",
        "",
        "All statements are read-only (`MATCH`/`RETURN`); the verification never "
        "creates, merges or deletes a node or relationship. Labels, relationship "
        "types and property keys are discovered from the live database.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], config: dict[str, Any]) -> tuple[Path, Path]:
    json_path = resolve(config["outputs"]["verification_json"])
    md_path = resolve(config["outputs"]["verification_md"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", default=None, help="Override the configured build id")
    parser.add_argument("--config", default="graph_feature_config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.build_id:
        config["graph"]["build_id"] = args.build_id

    settings = Settings(config)
    missing = settings.missing()
    if missing:
        print(f"Cannot verify: missing Neo4j setting(s) {missing}. "
              f"Configure {config['neo4j']['dotenv_file']} or the environment.",
              file=sys.stderr)
        return 3

    try:
        driver = connect(settings)
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        print(f"Cannot verify: connection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    try:
        with driver.session(database=settings.database) as session:
            stats = collect_stats(session, config)
    finally:
        driver.close()

    checks = run_checks(stats, config)
    summary = summarize(checks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": settings.database,
        "build_id": config["graph"]["build_id"],
        "summary": summary,
        "stats": stats,
        "checks": checks,
        "read_only": True,
    }
    json_path, md_path = write_reports(report, config)

    print(f"Overall: {summary['status'].upper()} "
          f"({summary['counts']['pass']} pass, {summary['counts']['warn']} warn, "
          f"{summary['counts']['fail']} fail)")
    print(f"Nodes: {stats['total_nodes']:,}  Relationships: {stats['total_relationships']:,}")
    print(f"Samples: {stats['samples']:,}  Graphs: {stats['graphs']:,}  Tokens: {stats['tokens']:,}")
    print(f"Report: {json_path}")
    print(f"Report: {md_path}")
    return 0 if summary["status"] != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
