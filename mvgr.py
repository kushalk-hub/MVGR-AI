"""MVGR-AI command line interface.

Every command is intentionally thin: all logic lives in ``person1``, ``person2``
or ``backend`` so that tests and notebooks can call the same functions. Imports
are lazy, which keeps ``mvgr --help`` fast and lets the core run without the
optional graph, GNN or API extras installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SPLITS = ("train", "validation", "test")


# --------------------------------------------------------------------------- #
# audit & lineage
# --------------------------------------------------------------------------- #
def cmd_audit(args: argparse.Namespace) -> int:
    from dataset_audit.audit import print_summary, run_full_audit

    result = run_full_audit(
        near_duplicates=args.near_duplicates,
        write_family=not args.no_family,
        write_reports=True,
    )
    print_summary(result)
    return 0 if result["gates"]["status"] == "pass" else 1


def cmd_lineage(args: argparse.Namespace) -> int:
    from dataset_audit.inventory import load_dataset
    from dataset_audit.lineage import (
        build_family_index,
        family_summary,
        read_family_index,
        write_family_index,
    )
    from shared.paths import dataset_config, derived_root

    config = dataset_config()
    path = derived_root(config) / "family_index.jsonl"
    if args.rebuild or not path.exists():
        frame = load_dataset(config=config)
        index = build_family_index(frame, config=config)
        write_family_index(index, path=path)
        print(f"Wrote family index: {path}")
    else:
        frame = load_dataset(config=config)
        index = read_family_index(path)
        print(f"Loaded family index: {path}")

    summary = family_summary(index, frame)
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# split manifest lifecycle
# --------------------------------------------------------------------------- #
def cmd_split_manifest(args: argparse.Namespace) -> int:
    from dataset_audit.inventory import load_dataset
    from dataset_audit.label_mapping import detect_confounding
    from dataset_audit.lineage import build_family_index
    from dataset_audit.split_manifest import build_manifest
    from shared.paths import dataset_config, manifest_root

    config = dataset_config()
    frame = load_dataset(config=config)
    families = build_family_index(frame, config=config)
    manifest = build_manifest(
        frame,
        families,
        config=config,
        mode=args.mode,
        scope_mode=args.scope,
        scope_sources=args.sources,
        created_by=args.created_by,
        confounding=detect_confounding(frame),
        near_duplicates=args.near_duplicates,
    )
    path = manifest_root(config) / "candidate" / manifest["manifest_id"] / "manifest.json"
    print(
        json.dumps(
            {
                "manifest": str(path),
                "manifest_id": manifest["manifest_id"],
                "scope": manifest["scope"],
                "counts": manifest["counts"],
                "gate_status": manifest["gates"]["status"],
                "failing_checks": [
                    check["check"]
                    for check in manifest["gates"]["checks"]
                    if check["status"] == "fail"
                ],
                "warnings": [
                    check["check"]
                    for check in manifest["gates"]["checks"]
                    if check["status"] == "warn"
                ],
            },
            indent=2,
        )
    )
    return 0 if manifest["gates"]["status"] == "pass" else 1


def _latest_candidate() -> Path:
    from shared.paths import manifest_root

    directory = manifest_root() / "candidate"
    candidates = sorted(directory.glob("man_*/manifest.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No candidate manifests under {directory}. Run `mvgr split-manifest` first."
        )
    return candidates[-1]


def cmd_approve_manifest(args: argparse.Namespace) -> int:
    from dataset_audit.inventory import load_dataset
    from dataset_audit.lineage import build_family_index
    from dataset_audit.split_manifest import approve_manifest
    from shared.paths import dataset_config

    config = dataset_config()
    path = Path(args.manifest) if args.manifest else _latest_candidate()
    frame = load_dataset(config=config)
    families = build_family_index(frame, config=config)
    canonical = approve_manifest(
        path,
        approver=args.approver,
        statement=args.statement,
        frame=frame,
        families=families,
        dataset_path=config["dataset"]["canonical"]["parquet"],
    )
    print(f"Approved manifest written to {canonical}")
    return 0


def cmd_verify_manifest(args: argparse.Namespace) -> int:
    from dataset_audit.inventory import load_dataset
    from dataset_audit.lineage import build_family_index
    from dataset_audit.split_manifest import load_manifest, verify_manifest
    from shared.paths import dataset_config, manifest_root

    config = dataset_config()
    path = Path(args.manifest) if args.manifest else manifest_root(config) / "split_manifest.approved.json"
    manifest = load_manifest(path)
    frame = load_dataset(config=config)
    families = build_family_index(frame, config=config)
    report = verify_manifest(
        manifest, frame, families, dataset_path=config["dataset"]["canonical"]["parquet"]
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


# --------------------------------------------------------------------------- #
# graph pipeline
# --------------------------------------------------------------------------- #
def cmd_graph_build(args: argparse.Namespace) -> int:
    import pandas as pd

    from dataset_audit.split_manifest import load_manifest, read_assignments
    from graph_engine.graph_export.export import build_view_export, write_build_manifest
    from shared.paths import dataset_config, graph_config, manifest_root

    dataset_cfg = dataset_config()
    graph_cfg = graph_config()
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else manifest_root(dataset_cfg) / "split_manifest.approved.json"
    )
    manifest = load_manifest(manifest_path)
    if manifest["status"] != "approved" and not args.allow_unapproved:
        raise SystemExit(
            f"Manifest {manifest['manifest_id']} is {manifest['status']}, not approved. "
            "Pass --allow-unapproved for a pre-approval smoke build; never train on it."
        )
    if manifest["status"] != "approved":
        print(
            "WARNING: building from a non-approved manifest. These exports are for "
            "inspection only and must not be used for reported results.",
            file=sys.stderr,
        )

    assignments = read_assignments(manifest)
    if args.split == "all":
        sample_ids = [sample_id for ids in assignments.values() for sample_id in ids]
    else:
        sample_ids = assignments[args.split]
    if args.limit:
        sample_ids = sorted(sample_ids)[: args.limit]

    frame = pd.read_parquet(
        dataset_cfg["dataset"]["canonical"]["parquet"], columns=["sample_id", "text"]
    )
    frame["sample_id"] = frame["sample_id"].astype(str)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(dataset_cfg["paths"]["derived_root"]) / "graph_exports"
    )
    if not output_root.is_absolute():
        from shared.paths import REPO_ROOT

        output_root = REPO_ROOT / output_root

    view_manifest = build_view_export(
        frame=frame,
        sample_ids=sample_ids,
        view=args.view,
        output_root=output_root,
        dataset_version=manifest["dataset_version"],
        split_id=manifest["manifest_id"],
        graph_config=graph_cfg,
        config_snapshot={
            "graph_spec_version": graph_cfg["graph_spec_version"],
            "view": args.view,
            "view_config": graph_cfg["views"][args.view],
            "manifest_id": manifest["manifest_id"],
        },
    )
    write_build_manifest(
        build_id=view_manifest["graph_build_id"],
        output_root=output_root,
        view_manifests=[view_manifest],
        config_snapshot={
            "graph_spec_version": graph_cfg["graph_spec_version"],
            "views": [args.view],
            "manifest_id": manifest["manifest_id"],
        },
        dataset_version=manifest["dataset_version"],
        split_id=manifest["manifest_id"],
    )
    print(json.dumps(view_manifest, indent=2))
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    from deterministic.feature_extractor import extract_export

    result = extract_export(
        args.export,
        output_path=args.output,
        registry_path=args.registry,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from baseline.tfidf_logistic import run_tfidf_logistic

    summary = run_tfidf_logistic(
        manifest_path=args.manifest,
        allow_test=args.allow_test,
        override_test_lock=args.override_test_lock,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_deterministic(args: argparse.Namespace) -> int:
    from deterministic.pipeline import run_deterministic_scoring

    summary = run_deterministic_scoring(
        features_path=args.features,
        view=args.view,
        manifest_path=args.manifest,
        allow_test=args.allow_test,
        override_test_lock=args.override_test_lock,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Neo4j, GNN, API, contracts
# --------------------------------------------------------------------------- #
def cmd_neo4j(args: argparse.Namespace) -> int:
    from graph_engine.graph_export.export import read_export
    from graph_engine.neo4j.connection import Neo4jConnection
    from graph_engine.neo4j.loader import GraphLoader
    from graph_engine.neo4j.queries import build_summary, list_builds
    from graph_engine.neo4j.schema import apply_schema, register_build
    from shared.paths import graph_config

    connection = Neo4jConnection.from_env()
    if args.action == "list":
        print(json.dumps(list_builds(connection), indent=2, default=str))
        return 0
    if args.action == "schema":
        print(json.dumps(apply_schema(connection, graph_config()), indent=2))
        return 0
    if args.action == "verify":
        print(json.dumps(build_summary(connection, args.build_id), indent=2))
        return 0
    if args.action == "load":
        if not args.export or not args.build_id:
            raise SystemExit("--export and --build-id are required for load")
        expected = sum(1 for _ in read_export(args.export))
        apply_schema(connection, graph_config())
        register_build(
            connection,
            build_id=args.build_id,
            graph_spec_version=args.graph_spec_version,
            dataset_version=args.dataset_version,
            split_id=args.split_id,
            config_id=args.config_id,
            sample_count=expected,
        )
        loader = GraphLoader(connection, batch_size=args.batch_size)
        report = loader.load_export(
            args.export,
            build_id=args.build_id,
            graph_view=args.view,
            expected_samples=expected,
            report_path=args.report,
        )
        print(json.dumps(report.as_dict(), indent=2))
        return 0 if report.failed == 0 else 1
    raise SystemExit(f"unknown neo4j action {args.action!r}")


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is not installed. Run `pip install -r requirements/api.txt`."
        ) from exc
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_contracts(args: argparse.Namespace) -> int:
    from shared.contracts.validators import SCHEMA_FILES, load_schema
    from shared.contracts.versions import CONTRACT_VERSIONS

    print(json.dumps(CONTRACT_VERSIONS, indent=2, sort_keys=True))
    if args.validate:
        failures: list[str] = []
        for name, filename in sorted(SCHEMA_FILES.items()):
            try:
                load_schema(name)
            except Exception as exc:  # noqa: BLE001 - aggregated report
                failures.append(f"{filename}: {type(exc).__name__}: {exc}")
        if failures:
            print("\nSchema failures:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1
        print(f"\nAll {len(SCHEMA_FILES)} contract schemas load and parse.")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def _add_manifest_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default=None, help="Explicit manifest path")
    parser.add_argument("--allow-test", action="store_true", help="Permit test-split evaluation")
    parser.add_argument(
        "--override-test-lock",
        action="store_true",
        help="Record a repeat test access for this experiment",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvgr",
        description="MVGR-AI: multi-view graph analysis of human and AI text",
    )
    parser.add_argument("--version", action="version", version="mvgr-ai 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Inventory, lineage, leakage and confounding audit")
    audit.add_argument("--near-duplicates", action="store_true", help="Slow MinHash cross-split scan")
    audit.add_argument("--no-family", action="store_true", help="Do not rewrite the family index")
    audit.set_defaults(func=cmd_audit)

    lineage = sub.add_parser("lineage", help="Show family-lineage summary")
    lineage.add_argument("--rebuild", action="store_true", help="Rebuild data/derived/family_index.jsonl")
    lineage.set_defaults(func=cmd_lineage)

    split = sub.add_parser("split-manifest", help="Build a candidate split manifest")
    split.add_argument("--mode", choices=["existing", "regenerate"], default="regenerate")
    split.add_argument("--scope", choices=["pooled", "source_scoped"], default="pooled")
    split.add_argument("--sources", nargs="*", default=None)
    split.add_argument("--created-by", default="person1")
    split.add_argument("--near-duplicates", action="store_true")
    split.set_defaults(func=cmd_split_manifest)

    approve = sub.add_parser("approve-manifest", help="Approve a candidate manifest")
    approve.add_argument("--manifest", default=None)
    approve.add_argument("--approver", required=True)
    approve.add_argument("--statement", required=True)
    approve.set_defaults(func=cmd_approve_manifest)

    verify = sub.add_parser("verify-manifest", help="Verify the approved manifest")
    verify.add_argument("--manifest", default=None)
    verify.set_defaults(func=cmd_verify_manifest)

    graph = sub.add_parser("graph-build", help="Build and export one graph view")
    graph.add_argument("--view", choices=["lexical", "syntactic"], default="lexical")
    graph.add_argument("--split", choices=[*SPLITS, "all"], default="train")
    graph.add_argument("--limit", type=int, default=0, help="Only build the first N ids")
    graph.add_argument("--manifest", default=None)
    graph.add_argument("--output-root", default=None)
    graph.add_argument("--allow-unapproved", action="store_true")
    graph.set_defaults(func=cmd_graph_build)

    features = sub.add_parser("features", help="Extract features.v1 from a graph export")
    features.add_argument("--export", required=True)
    features.add_argument("--output", default=None)
    features.add_argument("--registry", default=None)
    features.set_defaults(func=cmd_features)

    baseline = sub.add_parser("baseline", help="Run the TF-IDF + logistic baseline")
    _add_manifest_options(baseline)
    baseline.set_defaults(func=cmd_baseline)

    deterministic = sub.add_parser(
        "deterministic", help="Score features.v1 with a frozen validation rule"
    )
    deterministic.add_argument("--features", required=True)
    deterministic.add_argument("--view", choices=["lexical", "syntactic"], default="lexical")
    deterministic.add_argument("--top-k", type=int, default=12)
    _add_manifest_options(deterministic)
    deterministic.set_defaults(func=cmd_deterministic)

    neo4j = sub.add_parser("neo4j", help="Load or inspect a Neo4j graph build")
    neo4j.add_argument("action", choices=["load", "schema", "verify", "list"])
    neo4j.add_argument("--export", default=None)
    neo4j.add_argument("--build-id", default=None)
    neo4j.add_argument("--view", choices=["lexical", "syntactic"], default="lexical")
    neo4j.add_argument("--batch-size", type=int, default=500)
    neo4j.add_argument("--report", default=None)
    neo4j.add_argument("--graph-spec-version", default="1.0.0")
    neo4j.add_argument("--dataset-version", default="mvgr-ai/2026-11-13")
    neo4j.add_argument("--split-id", default="split_unknown")
    neo4j.add_argument("--config-id", default="cfg_unknown")
    neo4j.set_defaults(func=cmd_neo4j)

    serve = sub.add_parser("serve", help="Run the FastAPI backend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    contracts = sub.add_parser("contracts", help="Print or validate contract versions")
    contracts.add_argument("--validate", action="store_true", help="Load and parse every schema")
    contracts.set_defaults(func=cmd_contracts)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
