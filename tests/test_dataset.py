"""Dataset contract, provenance, lineage and split-gate tests.

Provenance is exercised on temporary files so the suite does not need the
multi-hundred-megabyte corpus. Nothing here writes to the canonical dataset.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from dataset_audit.inventory import verify_provenance
from dataset_audit.leakage_audit import (
    check_family_disjoint,
    check_label_domain,
    check_lineage_resolution,
    check_text_hash_disjoint,
    summarize,
)
from dataset_audit.lineage import build_family_index, read_family_index, write_family_index
from dataset_audit.split_manifest import (
    ManifestError,
    assign_by_family,
    assign_from_existing,
)
from shared.contracts.validators import load_schema
from shared.paths import dataset_config


@pytest.fixture(scope="module")
def config() -> dict:
    return dataset_config()


def _lineage_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "p1", "text": "shared pair text one", "source_dataset": "PADBen", "group_id": "g1"},
            {"sample_id": "p2", "text": "shared pair text two", "source_dataset": "PADBen", "group_id": "g1"},
            {"sample_id": "b1", "text": "Exact   Text", "source_dataset": "BiScope", "group_id": "x"},
            {"sample_id": "b2", "text": "exact text", "source_dataset": "BiScope", "group_id": "y"},
        ]
    )


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def test_verify_provenance_accepts_identical_copies(tmp_path):
    payload = b"identical bytes"
    source = tmp_path / "source.parquet"
    canonical = tmp_path / "canonical.parquet"
    source.write_bytes(payload)
    canonical.write_bytes(payload)

    result = verify_provenance(dataset_path=canonical, source_path=source)

    assert result["match"] is True
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["source_sha256"] == result["sha256"]


def test_verify_provenance_fails_closed_on_divergence(tmp_path):
    source = tmp_path / "source.parquet"
    canonical = tmp_path / "canonical.parquet"
    source.write_bytes(b"upstream")
    canonical.write_bytes(b"modified")

    with pytest.raises(ValueError, match="Provenance mismatch"):
        verify_provenance(dataset_path=canonical, source_path=source)


def test_verify_provenance_requires_the_upstream_file(tmp_path):
    canonical = tmp_path / "canonical.parquet"
    canonical.write_bytes(b"only one copy")

    with pytest.raises(FileNotFoundError):
        verify_provenance(dataset_path=canonical, source_path=tmp_path / "missing.parquet")


# --------------------------------------------------------------------------- #
# lineage
# --------------------------------------------------------------------------- #
def test_family_index_groups_pairs_and_exact_text(config):
    index = build_family_index(_lineage_frame(), config=config).set_index("sample_id")

    assert index.loc["p1", "family_id"] == index.loc["p2", "family_id"]
    assert index.loc["p1", "resolution_status"] == "resolved"

    # BiScope has no paraphrase-family identifier, so only exact normalized text
    # groups rows; case and whitespace differences collapse.
    assert index.loc["b1", "family_id"] == index.loc["b2", "family_id"]
    assert index.loc["b1", "resolution_status"] == "exact_text_only"
    assert index.loc["p1", "family_id"] != index.loc["b1", "family_id"]


def test_family_index_is_order_independent(config):
    frame = _lineage_frame()
    ordered = build_family_index(frame, config=config)
    shuffled = build_family_index(frame.sample(frac=1.0, random_state=0), config=config)
    pd.testing.assert_frame_equal(ordered, shuffled)


def test_family_index_round_trips_through_the_contract(config, tmp_path):
    index = build_family_index(_lineage_frame(), config=config)
    path = write_family_index(index, path=tmp_path / "family_index.jsonl", config=config)
    loaded = read_family_index(path)
    assert len(loaded) == len(index)
    assert set(loaded["sample_id"]) == set(index["sample_id"])


# --------------------------------------------------------------------------- #
# leakage gates
# --------------------------------------------------------------------------- #
def test_family_disjoint_gate_detects_a_spanning_family():
    frame = pd.DataFrame(
        [
            {"family_id": "f1", "split": "train"},
            {"family_id": "f1", "split": "test"},
            {"family_id": "f2", "split": "train"},
        ]
    )
    assert check_family_disjoint(frame).status == "fail"


def test_text_hash_gate_is_case_and_whitespace_insensitive():
    frame = pd.DataFrame(
        [
            {"text": "Hello World", "split": "train"},
            {"text": "hello    world", "split": "test"},
        ]
    )
    assert check_text_hash_disjoint(frame).status == "fail"


def test_label_domain_rejects_non_binary_labels():
    frame = pd.DataFrame({"label": [0, 1, 2]})
    assert check_label_domain(frame).status == "fail"


def test_lineage_gate_blocks_pooled_but_warns_source_scoped():
    frame = pd.DataFrame(
        [{"resolution_status": "exact_text_only", "source_dataset": "BiScope"}]
    )
    pooled = check_lineage_resolution(frame, allow_unresolved=False, scope_mode="pooled")
    scoped = check_lineage_resolution(frame, allow_unresolved=False, scope_mode="source_scoped")
    assert pooled.status == "fail"
    assert scoped.status == "warn"


def test_summarize_reports_fail_when_any_check_fails():
    summary = summarize(
        [
            check_label_domain(pd.DataFrame({"label": [0, 1]})),
            check_family_disjoint(pd.DataFrame({"family_id": ["f"], "split": ["train"]})),
            check_label_domain(pd.DataFrame({"label": [0, 2]})),
        ]
    )
    assert summary["status"] == "fail"
    assert summary["counts"]["fail"] == 1
    assert summary["counts"]["pass"] == 2


# --------------------------------------------------------------------------- #
# split assignment
# --------------------------------------------------------------------------- #
def _split_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(30)],
            "source_dataset": ["PADBen"] * 30,
            "label": [i % 2 for i in range(30)],
        }
    )
    families = pd.DataFrame(
        {"sample_id": frame["sample_id"], "family_id": [f"fam{i // 2}" for i in range(30)]}
    )
    return frame, families


def test_assign_by_family_is_deterministic_and_family_disjoint(config):
    frame, families = _split_frame()
    first = assign_by_family(frame, families, config=config)
    second = assign_by_family(frame, families, config=config)
    pd.testing.assert_frame_equal(first, second)

    merged = frame.merge(first, on="sample_id").merge(families, on="sample_id")
    assert (merged.groupby("family_id")["split"].nunique() == 1).all()
    assert set(merged["split"]) <= {"train", "validation", "test"}


def test_assign_from_existing_rejects_unknown_split_values():
    frame = pd.DataFrame({"sample_id": ["a"], "split": ["bogus"]})
    with pytest.raises(ManifestError):
        assign_from_existing(frame)


def test_manifest_gates_schema_allows_counts_and_rejects_extras():
    gates_schema = load_schema("manifest")["properties"]["gates"]
    validator = Draft202012Validator(gates_schema)
    validator.validate(
        {"status": "pass", "counts": {"pass": 1, "warn": 0, "fail": 0}, "checks": []}
    )
    with pytest.raises(Exception):
        validator.validate({"status": "pass", "checks": [], "unexpected": 1})
