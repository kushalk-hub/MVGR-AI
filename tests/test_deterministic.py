"""Deterministic feature, scoring, fingerprint, watermark and metric tests.

These tests must never read the test split and must never call the test lock.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from deterministic.feature_extractor import META_COLUMNS, extract_export, extract_graph_features, extract_records
from deterministic.feature_spec import FEATURE_NAMES, assert_registry_complete, feature_registry
from deterministic.scoring import (
    ScoringError,
    apply_rule,
    fit_scoring_rule,
    load_rule,
    save_rule,
)
from evaluation.metrics import (
    EvaluationError,
    binary_metrics,
    evaluate_predictions,
)
from fingerprinting.index import IndexError_, ReferenceIndex, build_index
from fingerprinting.minhash import MinHash
from fingerprinting.ngram import jaccard
from graph_engine.graph_builder import build_graph
from shared.paths import graph_config
from watermark.verifier import verify_text

LONG_TEXT = "the quick brown fox jumps over the lazy dog while the dog sleeps soundly"
OTHER_TEXT = "quantum entanglement describes correlated quantum states across distance"


@pytest.fixture(scope="module")
def gcfg() -> dict:
    return graph_config()


def _record(text: str, gcfg: dict, sample_id: str = "s1") -> dict:
    view_config = dict(gcfg["views"]["lexical"])
    view_config["shared_max_nodes"] = int(gcfg["shared"]["max_nodes"])
    return build_graph(
        "lexical",
        sample_id=sample_id,
        text=text,
        graph_build_id="gb_" + "c" * 20,
        config_id="cfg_" + "d" * 16,
        view_config=view_config,
        graph_spec_version=str(gcfg["graph_spec_version"]),
    )


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_feature_registry_is_complete_and_excludes_extras():
    registry = feature_registry()
    assert registry["feature_set_version"] == "features.v1"
    assert len(registry["features"]) == len(FEATURE_NAMES)

    assert_registry_complete({name: 0.0 for name in FEATURE_NAMES})
    with pytest.raises(AssertionError):
        assert_registry_complete({"n_nodes": 1.0})
    with pytest.raises(AssertionError):
        assert_registry_complete({**{name: 0.0 for name in FEATURE_NAMES}, "bogus": 1.0})


def test_extract_graph_features_covers_the_registry_and_is_deterministic(gcfg):
    record = _record(LONG_TEXT, gcfg)
    first = extract_graph_features(record)
    second = extract_graph_features(record)

    assert set(first) == set(FEATURE_NAMES)
    assert first == second
    assert first["n_nodes"] > 0
    assert first["n_edges"] > 0
    assert first["is_directed"] == 0.0


def test_extract_records_uses_the_declared_column_order(gcfg):
    frame = extract_records([_record(LONG_TEXT, gcfg, "s1"), _record(OTHER_TEXT, gcfg, "s2")])
    assert list(frame.columns) == list(META_COLUMNS) + list(FEATURE_NAMES)
    assert len(frame) == 2
    assert frame["feature_set_version"].eq("features.v1").all()


def test_extract_export_writes_parquet_and_registry(gcfg, tmp_path):
    from graph_engine.graph_export.export import write_view_export

    write_view_export(
        [_record(LONG_TEXT, gcfg, "s1"), _record(OTHER_TEXT, gcfg, "s2")],
        output_dir=tmp_path,
        build_meta={"graph_build_id": "gb_" + "c" * 20, "graph_view": "lexical"},
    )
    summary = extract_export(tmp_path / "lexical.jsonl", output_path=tmp_path / "lexical.features.parquet")

    assert summary["rows"] == 2
    assert (tmp_path / "lexical.features.parquet").exists()
    registry = json.loads((tmp_path / "lexical.features.registry.json").read_text())
    assert registry["feature_set_version"] == "features.v1"


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _scoring_data(n: int = 80) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * (n // 2))
    train = pd.DataFrame(
        {"signal": labels + rng.normal(0, 0.4, n), "noise": rng.normal(0, 1.0, n)}
    )
    validation = pd.DataFrame(
        {"signal": labels + rng.normal(0, 0.4, n), "noise": rng.normal(0, 1.0, n)}
    )
    return train, validation, labels


def test_fit_scoring_rule_selects_on_validation_and_freezes():
    train, validation, labels = _scoring_data()
    rule = fit_scoring_rule(
        train, labels, validation, labels, top_k=2, config_id="cfg_test"
    )

    assert rule["selection_split"] == "validation"
    assert rule["threshold_split"] == "validation"
    assert rule["validation_optimism"] is True
    assert "signal" in {feature["name"] for feature in rule["features"]}
    assert rule["rule_hash"]

    scores, predicted = apply_rule(validation, rule)
    assert scores.shape == (len(validation),)
    assert set(np.unique(predicted)) <= {0, 1}


def test_fit_scoring_rule_requires_both_classes_in_validation():
    train, validation, labels = _scoring_data()
    with pytest.raises(ScoringError):
        fit_scoring_rule(train, labels, validation, np.zeros(len(validation), dtype=int), top_k=2)


def test_rule_round_trip_and_tamper_detection(tmp_path):
    train, validation, labels = _scoring_data()
    rule = fit_scoring_rule(train, labels, validation, labels, top_k=2)
    path = save_rule(rule, tmp_path / "scoring_rule.json")

    assert load_rule(path)["rule_hash"] == rule["rule_hash"]

    tampered = json.loads(path.read_text())
    tampered["threshold"] = float(tampered["threshold"]) + 0.05
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ScoringError, match="hash does not match"):
        load_rule(path)


def test_apply_rule_reports_missing_feature_columns():
    train, validation, labels = _scoring_data()
    rule = fit_scoring_rule(train, labels, validation, labels, top_k=2)
    required = {feature["name"] for feature in rule["features"]}
    present = pd.DataFrame({name: [0.1] for name in required})
    dropped = present.drop(columns=[next(iter(required))])
    with pytest.raises(ScoringError, match="missing"):
        apply_rule(dropped, rule)


# --------------------------------------------------------------------------- #
# fingerprinting
# --------------------------------------------------------------------------- #
def test_minhash_similarity_is_bounded_and_handles_empty_text():
    hasher = MinHash(permutations=64, seed=42, shingle_size=5)
    assert hasher.similarity(LONG_TEXT, LONG_TEXT) == 1.0
    assert hasher.similarity("", LONG_TEXT) == 0.0
    assert 0.0 <= hasher.similarity(LONG_TEXT, OTHER_TEXT) <= 1.0


def test_ngram_jaccard_bounds():
    assert jaccard(LONG_TEXT, LONG_TEXT, size=5) == 1.0
    assert jaccard(LONG_TEXT, OTHER_TEXT, size=5) == 0.0


def test_reference_index_restricts_scope_and_finds_exact_match(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["r1", "r2", "r3"],
            "text": [LONG_TEXT, LONG_TEXT + " again", OTHER_TEXT],
        }
    )
    index = build_index(frame=frame, reference_ids=["r1", "r2", "r3"], permutations=64)
    assert len(index) == 3

    results = index.query(LONG_TEXT, top_k=3, threshold=0.0)
    assert results[0]["reference_id"] == "r1"
    assert results[0]["normalized_exact"] is True
    assert results[0]["score"] == 1.0

    path = index.save(tmp_path / "reference_index.npz")
    reloaded = ReferenceIndex.load(path)
    assert reloaded.sample_ids == index.sample_ids
    assert np.array_equal(reloaded.signatures, index.signatures)


def test_reference_index_rejects_unknown_ids():
    frame = pd.DataFrame({"sample_id": ["r1"], "text": [LONG_TEXT]})
    with pytest.raises(IndexError_):
        build_index(frame=frame, reference_ids=["r1", "missing"])


# --------------------------------------------------------------------------- #
# watermark
# --------------------------------------------------------------------------- #
def test_watermark_defaults_to_unsupported_and_never_implies_human():
    result = verify_text(LONG_TEXT)
    assert result.status.value == "unsupported"
    assert result.supports_human_authorship is False


def test_empty_text_watermark_is_inconclusive():
    assert verify_text("   ").status.value == "inconclusive"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_binary_metrics_confusion_matrix_and_auc():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_evaluate_predictions_rejects_leaked_metadata():
    predictions = pd.DataFrame(
        {"sample_id": ["a"], "predicted_score": [0.5], "label": [1]}
    )
    labels = pd.DataFrame({"sample_id": ["a"], "label": [1]})
    with pytest.raises(EvaluationError, match="target metadata"):
        evaluate_predictions(predictions, labels)


def test_evaluate_predictions_reports_overall_and_strata():
    predictions = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(4)],
            "predicted_score": [0.1, 0.2, 0.8, 0.9],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(4)],
            "label": [0, 0, 1, 1],
            "source_dataset": ["X"] * 4,
            "source_task": ["t"] * 4,
            "text_type": ["original"] * 4,
        }
    )
    result = evaluate_predictions(predictions, labels, strata=["source_dataset"], min_cell_size=2)
    assert result["overall"]["accuracy"] == 1.0
    assert result["strata"]["source_dataset"]["X"]["n"] == 4
