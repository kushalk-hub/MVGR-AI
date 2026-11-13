"""Audit orchestration and human-readable report generation (Person 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_audit.inventory import build_inventory, load_dataset
from dataset_audit.label_mapping import (
    detect_confounding,
    observed_task_condition_matrix,
    task_semantics,
    validate_labels,
)
from dataset_audit.leakage_audit import run_leakage_audit, summarize
from dataset_audit.lineage import build_family_index, family_summary, write_family_index
from shared.contracts.validators import write_json
from shared.paths import REPO_ROOT, dataset_config, resolve


def _report_dir(config: dict[str, Any]) -> Path:
    return resolve(config["paths"]["report_root"])


def _markdown_report(
    *,
    inventory: dict[str, Any],
    families: dict[str, Any],
    gates: dict[str, Any],
    confounding: list[dict[str, Any]],
    labels: dict[str, Any],
) -> str:
    lines = [
        "# MVGR-AI dataset audit",
        "",
        f"Dataset: `{inventory['dataset']['path']}`",
        f"SHA-256: `{inventory['dataset']['sha256']}`",
        f"Rows: **{inventory['dataset']['rows']:,}**",
        "",
        "## Label semantics",
        "",
        f"- Observed: {labels['observed']}",
        f"- Counts: {labels['counts']}",
        "- Labels describe authorship only. Paraphrase condition is not a label.",
        "",
        "## Sources and splits",
        "",
        "| Source | Rows |",
        "|---|---:|",
    ]
    for source, count in inventory["sources"]["counts"].items():
        lines.append(f"| {source} | {count:,} |")
    lines += ["", "| Split | Rows |", "|---|---:|"]
    for split, count in inventory["splits"]["counts"].items():
        lines.append(f"| {split} | {count:,} |")

    lines += [
        "",
        "## Lineage",
        "",
        f"- Families: {families['families']:,}",
        f"- Resolution: {families['resolution_status']}",
        f"- Families with both labels: {families['families_with_both_labels']}",
        f"- Families spanning splits: {families['families_spanning_splits']}",
        "",
        "> BiScope is grouped by exact normalized text only. Paraphrase families",
        "> cannot be reconstructed from the current export, so pooled paraphrase",
        "> claims require a recorded waiver or a source-scoped design.",
        "",
        "## Gates",
        "",
        f"Overall: **{gates['status'].upper()}**",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for check in gates["checks"]:
        lines.append(f"| {check['check']} | {check['status']} | {check['details']} |")

    lines += [
        "",
        "## Confounding",
        "",
        f"{len(confounding)} condition/label confounding findings.",
        "",
    ]
    for finding in confounding[:20]:
        lines.append(
            f"- `{finding.get('source_dataset')}/{finding.get('source_task')}` "
            f"text_type=`{finding.get('text_type') or finding.get('only_text_type')}`: "
            f"{finding['kind']} ({finding['rows']:,} rows). {finding['consequence']}"
        )
    if len(confounding) > 20:
        lines.append(f"- ... and {len(confounding) - 20} more (see confounding.json).")

    lines += [
        "",
        "## Text statistics",
        "",
        f"- Empty: {inventory['text']['empty_rows']}",
        f"- Shorter than 40 chars: {inventory['text']['shorter_than_40_chars']}",
        f"- Duplicate normalized texts: {inventory['text']['duplicate_normalized_rows']}",
        f"- Median length: {inventory['text']['length']['median']:.0f} characters",
        "",
        "## Warnings",
        "",
    ]
    for warning in inventory["warnings"]:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def run_full_audit(
    *,
    config: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    near_duplicates: bool = False,
    write_family: bool = True,
    write_reports: bool = True,
) -> dict[str, Any]:
    config = config or dataset_config()
    frame = load_dataset(config=config, path=dataset_path)
    dataset_path = dataset_path or config["dataset"]["canonical"]["parquet"]

    inventory = build_inventory(frame, config=config, dataset_path=dataset_path)
    labels = validate_labels(frame)
    families = build_family_index(frame, config=config)
    family_stats = family_summary(families, frame)
    confounding = detect_confounding(frame)
    checks = run_leakage_audit(
        frame,
        config=config,
        families=families,
        scope_mode="pooled",
        near_duplicates=near_duplicates,
        confounding=confounding,
    )
    gates = summarize(checks)

    if write_family:
        write_family_index(families, config=config)

    result = {
        "inventory": inventory,
        "labels": labels,
        "families": family_stats,
        "gates": gates,
        "confounding": confounding,
    }

    if write_reports:
        report_dir = _report_dir(config)
        report_dir.mkdir(parents=True, exist_ok=True)
        write_json(report_dir / "inventory.json", inventory)
        write_json(report_dir / "leakage_audit.json", gates)
        write_json(report_dir / "family_summary.json", family_stats)
        write_json(report_dir / "confounding.json", confounding)
        task_semantics(config).to_csv(report_dir / "task_reconciliation.csv", index=False)
        observed_task_condition_matrix(frame).to_csv(
            report_dir / "task_condition_matrix.csv", index=False
        )
        (report_dir / "audit_report.md").write_text(
            _markdown_report(
                inventory=inventory,
                families=family_stats,
                gates=gates,
                confounding=confounding,
                labels=labels,
            ),
            encoding="utf-8",
        )
        write_json(Path(REPO_ROOT) / "reports" / "dataset_audit" / "audit_summary.json", {
            "dataset_sha256": inventory["dataset"]["sha256"],
            "rows": inventory["dataset"]["rows"],
            "gates": gates["status"],
            "failing_checks": [c["check"] for c in gates["checks"] if c["status"] == "fail"],
        })
    return result


def print_summary(result: dict[str, Any]) -> None:
    inventory = result["inventory"]
    gates = result["gates"]
    print(f"Dataset rows: {inventory['dataset']['rows']:,}")
    print(f"SHA-256: {inventory['dataset']['sha256']}")
    print(f"Labels: {result['labels']['counts']}")
    print(f"Families: {result['families']['families']:,}")
    print(f"Gate status: {gates['status'].upper()}")
    for check in gates["checks"]:
        print(f"  [{check['status']:4}] {check['check']}: {check['details']}")
    print(f"Confounding findings: {len(result['confounding'])}")
    print(f"Reports: {_report_dir(dataset_config())}")
