"""Command line entry points for the dataset-audit module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dataset_audit.audit import print_summary, run_full_audit
from dataset_audit.inventory import load_dataset
from dataset_audit.label_mapping import detect_confounding
from dataset_audit.lineage import build_family_index
from dataset_audit.split_manifest import (
    ManifestError,
    approve_manifest,
    build_manifest,
    load_manifest,
    verify_manifest,
)
from shared.paths import dataset_config, manifest_root


def _latest_candidate() -> Path:
    directory = manifest_root() / "candidate"
    candidates = sorted(directory.glob("man_*/manifest.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No candidate manifests under {directory}. Run `mvgr split-manifest` first.")
    return candidates[-1]


def cmd_audit(args: argparse.Namespace) -> int:
    result = run_full_audit(near_duplicates=args.near_duplicates, write_family=not args.no_family)
    print_summary(result)
    return 0 if result["gates"]["status"] == "pass" else 1


def cmd_split_manifest(args: argparse.Namespace) -> int:
    config = dataset_config()
    frame = load_dataset(config=config)
    families = build_family_index(frame, config=config)
    confounding = detect_confounding(frame)
    manifest = build_manifest(
        frame,
        families,
        config=config,
        mode=args.mode,
        scope_mode=args.scope,
        scope_sources=args.sources,
        created_by=args.created_by,
        confounding=confounding,
        near_duplicates=args.near_duplicates,
    )
    path = manifest_root() / "candidate" / manifest["manifest_id"] / "manifest.json"
    print(json.dumps(
        {
            "manifest": str(path),
            "manifest_id": manifest["manifest_id"],
            "status": manifest["status"],
            "scope": manifest["scope"],
            "counts": manifest["counts"],
            "gate_status": manifest["gates"]["status"],
            "failing_checks": [
                check for check in manifest["gates"]["checks"] if check["status"] == "fail"
            ],
            "warnings": [
                check for check in manifest["gates"]["checks"] if check["status"] == "warn"
            ],
        },
        indent=2,
    ))
    return 0 if manifest["gates"]["status"] == "pass" else 1


def cmd_approve_manifest(args: argparse.Namespace) -> int:
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
    config = dataset_config()
    path = Path(args.manifest) if args.manifest else manifest_root() / "split_manifest.approved.json"
    manifest = load_manifest(path)
    frame = load_dataset(config=config)
    families = build_family_index(frame, config=config)
    report = verify_manifest(
        manifest,
        frame,
        families,
        dataset_path=config["dataset"]["canonical"]["parquet"],
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvgr-audit", description="MVGR-AI dataset audit")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Run inventory, lineage, leakage and confounding audit")
    audit.add_argument("--near-duplicates", action="store_true", help="Run MinHash cross-split scan (slow)")
    audit.add_argument("--no-family", action="store_true", help="Do not rewrite data/derived/family_index.jsonl")
    audit.set_defaults(func=cmd_audit)

    split = sub.add_parser("split-manifest", help="Build a candidate split manifest")
    split.add_argument("--mode", choices=["existing", "regenerate"], default="regenerate")
    split.add_argument("--scope", choices=["pooled", "source_scoped"], default="pooled")
    split.add_argument("--sources", nargs="*", default=None, help="Sources for source_scoped scope")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ManifestError as exc:
        print(f"MANIFEST ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
