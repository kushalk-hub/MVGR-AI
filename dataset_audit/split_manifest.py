"""Canonical split manifest construction, approval and verification (Person 1).

No model may train against a split that is not an approved manifest. The
manifest stores checksummed, sorted ID files rather than a 128k-row JSON blob,
so approvals are reviewable in git.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_audit.inventory import file_sha256
from dataset_audit.leakage_audit import (
    run_leakage_audit,
    summarize,
)
from shared.contracts.ids import digest
from shared.contracts.validators import (
    utc_now,
    validate_manifest,
    write_json,
)
from shared.paths import REPO_ROOT, dataset_config, manifest_root

SPLITS = ("train", "validation", "test")


class ManifestError(ValueError):
    pass


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sorted_ids(values: list[str]) -> list[str]:
    return sorted({str(value) for value in values})


def assign_from_existing(frame: pd.DataFrame) -> pd.DataFrame:
    """Adopt the split column already present in the dataset as a candidate."""
    if "split" not in frame:
        raise ManifestError("dataset has no split column")
    observed = set(str(value) for value in frame["split"].unique())
    unexpected = observed - set(SPLITS)
    if unexpected:
        raise ManifestError(f"unexpected split values {sorted(unexpected)}")
    return frame[["sample_id", "split"]].copy()


def assign_by_family(
    frame: pd.DataFrame,
    families: pd.DataFrame,
    *,
    config: dict[str, Any],
    seed: int | None = None,
) -> pd.DataFrame:
    """Deterministic, family-disjoint proportional assignment.

    Families are the assignment unit, so a paraphrase pair can never be split.
    Within each (source, family-label-composition) stratum, families are
    ordered by a seeded hash and greedily placed in the split with the largest
    relative deficit.
    """
    seed = int(seed if seed is not None else config["splitting"]["seed"])
    ratios: dict[str, float] = {
        "train": float(config["splitting"]["ratios"]["train"]),
        "validation": float(config["splitting"]["ratios"]["validation"]),
        "test": float(config["splitting"]["ratios"]["test"]),
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ManifestError(f"split ratios must sum to 1.0, got {ratios}")

    merged = frame[["sample_id", "source_dataset", "label"]].merge(
        families[["sample_id", "family_id"]], on="sample_id", how="left", validate="one_to_one"
    )
    if merged["family_id"].isna().any():
        raise ManifestError("some samples have no family_id")

    family_stats = (
        merged.groupby("family_id")
        .agg(
            size=("sample_id", "size"),
            sources=("source_dataset", "nunique"),
            labels=("label", "nunique"),
            source=("source_dataset", "first"),
        )
        .reset_index()
    )
    family_stats["stratum"] = family_stats["source"] + "|" + family_stats["labels"].map(
        {1: "pure", 2: "mixed"}
    )

    assignment: dict[str, str] = {}
    for _, stratum_frame in family_stats.groupby("stratum", sort=True):
        ordered = sorted(
            stratum_frame.itertuples(index=False),
            key=lambda row: hashlib.sha256(f"{seed}:{row.family_id}".encode()).hexdigest(),
        )
        current = {split: 0 for split in SPLITS}
        for family in ordered:
            split = min(SPLITS, key=lambda name: (current[name] + int(family.size)) / ratios[name])
            current[split] += int(family.size)
            for sample_id in merged.loc[merged["family_id"] == family.family_id, "sample_id"]:
                assignment[str(sample_id)] = split

    result = pd.DataFrame(
        {"sample_id": list(assignment.keys()), "split": list(assignment.values())}
    )
    if len(result) != len(frame):
        raise ManifestError("family assignment dropped or duplicated samples")
    return result


def build_manifest(
    frame: pd.DataFrame,
    families: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    mode: str = "regenerate",
    scope_mode: str = "pooled",
    scope_sources: list[str] | None = None,
    created_by: str = "person1",
    dataset_path: str | Path | None = None,
    confounding: list[dict[str, Any]] | None = None,
    near_duplicates: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = config or dataset_config()
    dataset_path = Path(dataset_path) if dataset_path else Path(config["dataset"]["canonical"]["parquet"])

    scope_sources = sorted(scope_sources or [])
    scoped = frame
    if scope_mode == "source_scoped":
        if not scope_sources:
            raise ManifestError("source_scoped manifests require at least one source")
        scoped = frame[frame["source_dataset"].isin(scope_sources)].copy()
        if scoped.empty:
            raise ManifestError(f"no rows for scope sources {scope_sources}")
    elif scope_mode != "pooled":
        raise ManifestError(f"unknown scope mode {scope_mode!r}")

    if mode == "existing":
        assignments = assign_from_existing(scoped)
    elif mode == "regenerate":
        assignments = assign_by_family(scoped, families, config=config)
    else:
        raise ManifestError(f"unknown manifest mode {mode!r}")

    merged = scoped.drop(columns=["split"], errors="ignore").merge(
        assignments, on="sample_id", how="left", validate="one_to_one"
    )
    if merged["split"].isna().any():
        raise ManifestError("some scoped samples were not assigned to a split")
    checks = run_leakage_audit(
        merged,
        config=config,
        families=families,
        scope_mode=scope_mode,
        near_duplicates=near_duplicates,
        confounding=confounding,
    )
    gates = summarize(checks)
    if gates["status"] == "fail":
        # Keep the candidate for inspection, but it can never be approved
        # without a recorded waiver.
        pass

    counts = {split: int((assignments["split"] == split).sum()) for split in SPLITS}
    if any(count == 0 for count in counts.values()):
        gates["status"] = "fail"
        gates["checks"].append(
            {
                "check": "non_empty_splits",
                "status": "fail",
                "details": f"empty splits: {[k for k, v in counts.items() if v == 0]}",
            }
        )

    manifest_id = "man_" + digest(
        {
            "dataset_sha256": file_sha256(dataset_path),
            "dataset_version": config["dataset"]["version"],
            "split_policy_version": config["splitting"]["policy_version"],
            "group_key": "family_id",
            "seed": int(config["splitting"]["seed"]),
            "scope": {"mode": scope_mode, "sources": scope_sources},
            "mode": mode,
            "counts": counts,
            "ids": {
                split: hashlib.sha256(
                    "\n".join(_sorted_ids(assignments.loc[assignments["split"] == split, "sample_id"].tolist())).encode()
                ).hexdigest()
                for split in SPLITS
            },
        },
        length=20,
    )

    root = Path(output_dir) if output_dir else manifest_root(config) / "candidate" / manifest_id
    root.mkdir(parents=True, exist_ok=True)
    assignment_block: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        ids = _sorted_ids(assignments.loc[assignments["split"] == split, "sample_id"].tolist())
        path = root / f"{split}.ids.txt"
        payload = "\n".join(ids) + ("\n" if ids else "")
        path.write_text(payload, encoding="utf-8", newline="\n")
        assignment_block[split] = {
            "path": _repo_relative(path),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "count": len(ids),
        }

    manifest = {
        "manifest_id": manifest_id,
        "manifest_version": "1.0.0",
        "dataset_version": config["dataset"]["version"],
        "dataset_sha256": file_sha256(dataset_path),
        "split_policy_version": config["splitting"]["policy_version"],
        "group_key": "family_id",
        "seed": int(config["splitting"]["seed"]),
        "created_at": utc_now(),
        "created_by": created_by,
        "status": "candidate",
        "scope": {
            "mode": scope_mode,
            "sources": scope_sources,
            "notes": (
                "Pooled scope. Exact-text-only families in BiScope block pooled "
                "paraphrase-family claims until a waiver is recorded."
                if scope_mode == "pooled"
                else "Source-scoped evaluation; do not generalise across sources."
            ),
        },
        "counts": counts,
        "assignments": assignment_block,
        "gates": gates,
        "approvals": [],
    }
    validate_manifest(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def _resolve_assignment_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def read_assignments(manifest: dict[str, Any], *, verify_checksums: bool = True) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in SPLITS:
        entry = manifest["assignments"][split]
        path = _resolve_assignment_path(entry["path"])
        if not path.exists():
            raise ManifestError(f"assignment file missing: {path}")
        payload = path.read_text(encoding="utf-8")
        ids = [line.strip() for line in payload.splitlines() if line.strip()]
        if verify_checksums:
            actual = hashlib.sha256(payload.encode()).hexdigest()
            if actual != entry["sha256"]:
                raise ManifestError(f"assignment checksum mismatch for {split}: {path}")
        if len(ids) != int(entry["count"]):
            raise ManifestError(f"assignment count mismatch for {split}: {path}")
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise ManifestError(f"assignment file must be sorted and unique: {path}")
        result[split] = ids
    return result


def assignment_map(manifest: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split, ids in read_assignments(manifest).items():
        for sample_id in ids:
            mapping[sample_id] = split
    return mapping


def verify_manifest(
    manifest: dict[str, Any],
    frame: pd.DataFrame,
    families: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    validate_manifest(manifest)
    assignments = read_assignments(manifest)
    all_ids = [sample_id for ids in assignments.values() for sample_id in ids]
    problems: list[str] = []
    warnings: list[str] = []

    if dataset_path is not None:
        actual = file_sha256(dataset_path)
        if actual != manifest["dataset_sha256"]:
            problems.append("dataset digest does not match the manifest")

    dataset_ids = set(frame["sample_id"].astype(str))
    assigned_ids = set(all_ids)
    if len(all_ids) != len(assigned_ids):
        problems.append("a sample is assigned to more than one split")
    if manifest["scope"]["mode"] == "pooled" and assigned_ids != dataset_ids:
        problems.append(
            f"pooled manifest does not cover the dataset "
            f"(assigned={len(assigned_ids)}, dataset={len(dataset_ids)})"
        )
    unknown = assigned_ids - dataset_ids
    if unknown:
        problems.append(f"{len(unknown)} assignment ids do not exist in the dataset")

    for split in SPLITS:
        if len(assignments[split]) != int(manifest["counts"][split]):
            problems.append(f"count mismatch for {split}")

    merged = frame.merge(
        pd.DataFrame({"sample_id": all_ids, "split": [s for s, ids in assignments.items() for _ in ids]}),
        on="sample_id",
        how="inner",
    )
    merged = merged.merge(families[["sample_id", "family_id"]], on="sample_id", how="left")
    spanning_families = int((merged.groupby("family_id")["split"].nunique() > 1).sum())
    if spanning_families:
        problems.append(f"{spanning_families} families span splits")

    total = len(merged)
    for split, target in {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }.items():
        observed = len(merged[merged["split"] == split]) / max(total, 1)
        if abs(observed - target) > 0.05:
            warnings.append(f"{split} share {observed:.3f} deviates from {target:.2f}")

    return {
        "manifest_id": manifest["manifest_id"],
        "status": manifest["status"],
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "counts": {split: len(assignments[split]) for split in SPLITS},
    }


def approve_manifest(
    manifest_path: str | Path,
    *,
    approver: str,
    statement: str,
    frame: pd.DataFrame | None = None,
    families: pd.DataFrame | None = None,
    dataset_path: str | Path | None = None,
) -> Path:
    """Approve a candidate manifest and publish it as the canonical manifest.

    A failing gate requires a statement beginning with ``WAIVER:``; the waiver
    is stored verbatim with the approval.
    """
    if not approver.strip():
        raise ManifestError("an approver name is required")
    if not statement.strip():
        raise ManifestError("an approval statement is required")

    manifest = load_manifest(manifest_path)
    if manifest["status"] != "candidate":
        raise ManifestError(f"manifest status is {manifest['status']}, expected candidate")

    waived = statement.strip().upper().startswith("WAIVER:")
    if manifest["gates"]["status"] != "pass" and not waived:
        failing = [
            check["check"] for check in manifest["gates"]["checks"] if check["status"] == "fail"
        ]
        raise ManifestError(
            "manifest gates failed: "
            f"{failing}. Fix the data or record an explicit statement beginning with WAIVER:"
        )

    assignments = read_assignments(manifest)
    manifest_id = manifest["manifest_id"]
    approved_root = manifest_root() / "approved" / manifest_id
    approved_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        source = _resolve_assignment_path(manifest["assignments"][split]["path"])
        target = approved_root / f"{split}.ids.txt"
        shutil.copyfile(source, target)

    manifest["status"] = "approved"
    manifest["assignments"] = {
        split: {
            **manifest["assignments"][split],
            "path": _repo_relative(approved_root / f"{split}.ids.txt"),
        }
        for split in SPLITS
    }
    manifest.setdefault("approvals", []).append(
        {
            "approver": approver.strip(),
            "approved_at": utc_now(),
            "statement": statement.strip(),
        }
    )
    validate_manifest(manifest)

    canonical = manifest_root() / "split_manifest.approved.json"
    write_json(canonical, manifest)

    # Fail closed: the published manifest must still verify on disk.
    if frame is not None and families is not None:
        report = verify_manifest(manifest, frame, families, dataset_path=dataset_path)
        if not report["ok"]:
            raise ManifestError(f"approved manifest failed verification: {report['problems']}")
    return canonical


def require_approved_manifest(
    path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or dataset_config()
    path = Path(path) if path else manifest_root(config) / "split_manifest.approved.json"
    if not path.exists():
        raise ManifestError(
            f"No approved split manifest at {path}. Run `mvgr split-manifest` and `mvgr approve-manifest` first."
        )
    manifest = load_manifest(path)
    if manifest["status"] != "approved":
        raise ManifestError(f"manifest {manifest['manifest_id']} is not approved")
    if not manifest.get("approvals"):
        raise ManifestError("approved manifest has no approval record")
    read_assignments(manifest)
    return manifest


__all__ = [
    "ManifestError",
    "SPLITS",
    "approve_manifest",
    "assign_by_family",
    "assign_from_existing",
    "assignment_map",
    "build_manifest",
    "load_manifest",
    "read_assignments",
    "require_approved_manifest",
    "verify_manifest",
]
