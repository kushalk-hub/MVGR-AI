"""Person 1: dataset contract, deterministic analysis and evaluation."""

from dataset_audit.audit import run_full_audit
from dataset_audit.inventory import build_inventory, load_dataset
from dataset_audit.leakage_audit import run_leakage_audit, summarize
from dataset_audit.lineage import build_family_index, family_summary
from dataset_audit.split_manifest import (
    approve_manifest,
    assignment_map,
    build_manifest,
    load_manifest,
    require_approved_manifest,
    verify_manifest,
)

__all__ = [
    "approve_manifest",
    "assignment_map",
    "build_family_index",
    "build_inventory",
    "build_manifest",
    "family_summary",
    "load_dataset",
    "load_manifest",
    "require_approved_manifest",
    "run_full_audit",
    "run_leakage_audit",
    "summarize",
    "verify_manifest",
]
