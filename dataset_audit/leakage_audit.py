"""Leakage and integrity audit (Person 1).

Every check returns pass / warn / fail. A manifest gate is the union of these
checks; failure blocks approval unless an explicit, recorded waiver is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from shared.preprocessing.text import normalize_for_hashing, sha256_text


@dataclass(frozen=True)
class CheckResult:
    check: str
    status: str  # pass | warn | fail
    details: str

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "status": self.status, "details": self.details}


def _fail(check: str, details: str) -> CheckResult:
    return CheckResult(check, "fail", details)


def _warn(check: str, details: str) -> CheckResult:
    return CheckResult(check, "warn", details)


def _pass(check: str, details: str) -> CheckResult:
    return CheckResult(check, "pass", details)


def check_unique_ids(frame: pd.DataFrame) -> CheckResult:
    duplicates = int(frame["sample_id"].duplicated().sum())
    if duplicates:
        return _fail("unique_sample_ids", f"{duplicates} duplicate sample_id rows")
    return _pass("unique_sample_ids", f"{len(frame)} unique sample ids")


def check_text_present(frame: pd.DataFrame) -> CheckResult:
    lengths = frame["text"].fillna("").str.len()
    empty = int((lengths == 0).sum())
    short = int((lengths < 40).sum())
    if empty:
        return _fail("text_present", f"{empty} empty texts")
    if short:
        return _warn("text_present", f"{short} texts shorter than 40 characters")
    return _pass("text_present", "no empty texts")


def check_label_domain(frame: pd.DataFrame) -> CheckResult:
    observed = sorted(int(value) for value in frame["label"].dropna().unique())
    if frame["label"].isna().any():
        return _fail("label_domain", "null labels present")
    if observed != [0, 1]:
        return _fail("label_domain", f"observed labels {observed}, expected [0, 1]")
    return _pass("label_domain", "binary labels 0/1 only")


def check_family_disjoint(frame: pd.DataFrame) -> CheckResult:
    required = {"family_id", "split"}
    if not required.issubset(frame.columns):
        return _fail("family_disjoint", f"missing columns {sorted(required - set(frame.columns))}")
    per_family = frame.groupby("family_id")["split"].nunique()
    spanning = int((per_family > 1).sum())
    if spanning:
        return _fail("family_disjoint", f"{spanning} families span more than one split")
    return _pass(
        "family_disjoint",
        f"{len(per_family)} families, none spanning splits",
    )


def check_text_hash_disjoint(frame: pd.DataFrame) -> CheckResult:
    hashes = frame["text"].map(normalize_for_hashing).map(sha256_text)
    per_hash = frame.assign(_hash=hashes).groupby("_hash")["split"].nunique()
    spanning = int((per_hash > 1).sum())
    if spanning:
        return _fail("text_hash_disjoint", f"{spanning} normalized texts appear in more than one split")
    return _pass("text_hash_disjoint", "no normalized text crosses a split")


def check_label_coverage(frame: pd.DataFrame, min_per_cell: int) -> CheckResult:
    test = frame[frame["split"] == "test"]
    if test.empty:
        return _fail("label_coverage", "test split is empty")
    if set(int(value) for value in test["label"].unique()) != {0, 1}:
        return _fail("label_coverage", "test split does not contain both classes")
    cells = test.groupby(["source_task", "label"]).size()
    small = {
        f"{task}/label={label}": int(count)
        for (task, label), count in cells.items()
        if count < min_per_cell
    }
    if small:
        return _warn("label_coverage", f"test cells below {min_per_cell}: {small}")
    return _pass("label_coverage", f"all {len(cells)} task/label test cells have >= {min_per_cell} rows")


def check_split_ratios(frame: pd.DataFrame, ratios: dict[str, float], tolerance: float = 0.03) -> CheckResult:
    total = len(frame)
    if total == 0:
        return _fail("split_ratios", "empty dataset")
    observed = frame["split"].value_counts(normalize=True)
    deviations = {
        split: abs(float(observed.get(split, 0.0)) - target)
        for split, target in ratios.items()
    }
    rendered = ", ".join(f"{split}={observed.get(split, 0.0):.3f} (target {target:.3f})" for split, target in ratios.items())
    if any(value > tolerance for value in deviations.values()):
        return _warn("split_ratios", f"deviation > {tolerance}: {rendered}")
    return _pass("split_ratios", rendered)


def check_class_imbalance(frame: pd.DataFrame, max_ratio: float) -> CheckResult:
    counts = frame["label"].value_counts()
    if 0 not in counts or 1 not in counts:
        return _fail("class_imbalance", "one class is absent")
    ratio = float(counts.max() / counts.min())
    if ratio > max_ratio:
        return _fail(
            "class_imbalance",
            f"AI:human ratio {ratio:.2f} exceeds {max_ratio}; report balanced metrics, never accuracy alone",
        )
    return _pass("class_imbalance", f"AI:human ratio {ratio:.2f}")


def check_lineage_resolution(
    frame: pd.DataFrame,
    *,
    allow_unresolved: bool,
    scope_mode: str,
) -> CheckResult:
    if "resolution_status" not in frame.columns:
        return _fail("lineage_resolution", "family resolution status was not joined into the audit frame")
    unresolved = frame[frame["resolution_status"].isin(["exact_text_only", "unresolved"])]
    count = int(len(unresolved))
    if count == 0:
        return _pass("lineage_resolution", "all families resolved at source level")
    sources = sorted(str(value) for value in unresolved["source_dataset"].unique())
    details = (
        f"{count} rows rely on exact-text-only or unresolved lineage in {sources}; "
        "paraphrase families may cross splits. Source-scoped claims are possible; "
        "pooled claims require an explicit waiver."
    )
    if allow_unresolved:
        return _warn("lineage_resolution", details)
    if scope_mode == "pooled":
        return _fail("lineage_resolution", details)
    return _warn("lineage_resolution", details)


def check_condition_confounding(findings: Iterable[dict[str, Any]]) -> CheckResult:
    findings = list(findings)
    if not findings:
        return _pass("condition_confounding", "no perfect condition/label confounding detected")
    cells = sum(int(item.get("rows", 0)) for item in findings)
    return _warn(
        "condition_confounding",
        f"{len(findings)} confounded cells covering {cells} rows; condition effects cannot be estimated there",
    )


def check_near_duplicates(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.85,
    permutations: int = 128,
) -> CheckResult:
    """Cross-split near-duplicate detection with MinHash + band LSH.

    This is intentionally opt-in because it is CPU-heavy on the full corpus.
    """
    from fingerprinting.minhash import MinHash

    hasher = MinHash(permutations=permutations, seed=42, shingle_size=5)
    hashes = frame["text"].map(normalize_for_hashing)
    signatures = [hasher.signature(text) for text in hashes]
    splits = frame["split"].tolist()
    sample_ids = frame["sample_id"].tolist()

    bands = hasher.bands
    index: dict[tuple[Any, ...], list[int]] = {}
    conflicts: set[tuple[str, str]] = set()
    for position, signature in enumerate(signatures):
        for band_index, key in enumerate(bands(signature)):
            bucket = (band_index, key)
            for other in index.get(bucket, []):
                if splits[other] == splits[position]:
                    continue
                estimate = hasher.similarity_from_signature(signature, signatures[other])
                if estimate >= threshold:
                    pair = tuple(sorted((sample_ids[other], sample_ids[position])))
                    conflicts.add(pair)
            index.setdefault(bucket, []).append(position)

    if conflicts:
        sample = sorted(conflicts)[:5]
        return _fail(
            "near_duplicates",
            f"{len(conflicts)} cross-split near-duplicate pairs at >= {threshold}, e.g. {sample}",
        )
    return _pass("near_duplicates", f"no cross-split pairs at >= {threshold}")


def run_leakage_audit(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    families: pd.DataFrame,
    scope_mode: str = "pooled",
    near_duplicates: bool = False,
    confounding: list[dict[str, Any]] | None = None,
) -> list[CheckResult]:
    merged = frame.merge(families[["sample_id", "family_id", "resolution_status"]], on="sample_id", how="left")
    gates = config["splitting"]["gates"]
    checks = [
        check_unique_ids(merged),
        check_text_present(merged),
        check_label_domain(merged),
        check_family_disjoint(merged),
        check_text_hash_disjoint(merged),
        check_split_ratios(merged, config["splitting"]["ratios"]),
        check_label_coverage(merged, int(gates["require_min_test_per_cell"])),
        check_class_imbalance(merged, float(gates["max_label_imbalance_ratio"])),
        check_lineage_resolution(
            merged,
            allow_unresolved=bool(gates["allow_unresolved_lineage"]),
            scope_mode=scope_mode,
        ),
        check_condition_confounding(confounding or []),
    ]
    if near_duplicates:
        checks.append(
            check_near_duplicates(
                merged,
                threshold=float(config["duplicates"]["near_duplicate_threshold"]),
            )
        )
    return checks


def summarize(checks: Iterable[CheckResult]) -> dict[str, Any]:
    checks = list(checks)
    statuses = [check.status for check in checks]
    status = "fail" if "fail" in statuses else "pass"
    return {
        "status": status,
        "counts": {
            "pass": statuses.count("pass"),
            "warn": statuses.count("warn"),
            "fail": statuses.count("fail"),
        },
        "checks": [check.as_dict() for check in checks],
    }
