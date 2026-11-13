"""Cross-module integration tests for the online and offline pipelines.

These tests stitch graph building, feature extraction, scoring, evaluation, the
backend report contract and the test-set lock together. They never touch the
real test split and never require an approved manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evaluation.test_lock as test_lock
from backend.services.analysis import analyze
from backend.services.model_bundle import load_bundle
from deterministic.feature_extractor import META_COLUMNS, extract_records
from deterministic.feature_spec import FEATURE_NAMES
from deterministic.pipeline import _predictions
from deterministic.scoring import fit_scoring_rule, save_rule
from evaluation.metrics import evaluate_predictions
from graph_engine.graph_builder import build_graph
from shared.contracts.validators import ContractError, validate, validate_analysis_report
from shared.paths import graph_config

ONLINE_TEXT = (
    "Retrieval augmented generation grounds a language model in external "
    "documents so that answers cite verifiable sources instead of guessing."
)


@pytest.fixture(scope="module")
def gcfg() -> dict:
    return graph_config()


def _record(text: str, gcfg: dict, sample_id: str) -> dict:
    view_config = dict(gcfg["views"]["lexical"])
    view_config["shared_max_nodes"] = int(gcfg["shared"]["max_nodes"])
    return build_graph(
        "lexical",
        sample_id=sample_id,
        text=text,
        graph_build_id="gb_" + "e" * 20,
        config_id="cfg_" + "f" * 16,
        view_config=view_config,
        graph_spec_version=str(gcfg["graph_spec_version"]),
    )


# --------------------------------------------------------------------------- #
# online pipeline: backend report contract
# --------------------------------------------------------------------------- #
def test_analyze_without_a_bundle_stays_contract_valid(tmp_path):
    report = analyze(ONLINE_TEXT, bundle=load_bundle(tmp_path / "no-such-bundle"))

    validate_analysis_report(report)
    assert report["classification"]["status"] == "unavailable"
    assert report["watermark"]["status"] == "unsupported"
    assert report["fingerprint"]["status"] == "no_reference_index"
    assert report["graph_analysis"]["status"] == "available"
    assert any("human authorship" in item for item in report["limitations"])


def test_analyze_with_a_frozen_bundle_produces_a_score(tmp_path):
    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 40)
    columns = ["n_nodes", "n_edges", "degree_mean", "avg_clustering"]
    train = pd.DataFrame({name: labels + rng.normal(0, 0.3, 80) for name in columns})
    validation = pd.DataFrame({name: labels + rng.normal(0, 0.3, 80) for name in columns})
    rule = fit_scoring_rule(train, labels, validation, labels, top_k=4)
    save_rule(rule, tmp_path / "bundle" / "scoring_rule.json")

    report = analyze(ONLINE_TEXT, bundle=load_bundle(tmp_path / "bundle"))
    validate_analysis_report(report)
    assert report["classification"]["status"] == "available"
    assert report["classification"]["label"] in {"human", "ai"}
    assert 0.0 <= report["classification"]["p_ai"] <= 1.0
    assert report["classification"]["calibrated"] is False


def test_report_contract_forbids_watermark_absence_as_human():
    report = analyze(ONLINE_TEXT, bundle=load_bundle("definitely-missing"))
    report["watermark"]["status"] = "not_detected"
    report["classification"] = {"status": "available", "label": "human", "p_ai": 0.1}
    with pytest.raises(ContractError):
        validate_analysis_report(report)


def test_backend_app_exposes_the_documented_routes():
    pytest.importorskip("fastapi")
    from backend.main import app

    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/v1/analyze" in paths
    assert "/v1/contracts/versions" in paths


# --------------------------------------------------------------------------- #
# test-set lock
# --------------------------------------------------------------------------- #
def test_test_lock_is_single_use_and_records_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(test_lock, "_lock_path", lambda: tmp_path / "test_access_log.jsonl")
    experiment = "exp_" + "a" * 20

    test_lock.request_test_access(experiment_id=experiment, purpose="first evaluation")
    with pytest.raises(test_lock.TestLockError):
        test_lock.request_test_access(experiment_id=experiment, purpose="silent repeat")
    test_lock.request_test_access(
        experiment_id=experiment, purpose="documented repeat", override=True
    )

    entries = test_lock.test_access_log()
    assert len(entries) == 2
    assert entries[-1]["override"] is True


# --------------------------------------------------------------------------- #
# offline pipeline: graph -> features -> scoring -> evaluation
# --------------------------------------------------------------------------- #
def _synthetic_corpus() -> pd.DataFrame:
    base = ["alpha", "beta", "gamma", "delta"]
    rows: list[dict[str, object]] = []
    for index in range(20):
        order = base[index % 4 :] + base[: index % 4]
        rows.append({"sample_id": f"h{index}", "text": " ".join(order), "label": 0})
    for index in range(20):
        text = " ".join(f"token{index}x{position}" for position in range(30))
        rows.append({"sample_id": f"a{index}", "text": text, "label": 1})
    return pd.DataFrame(rows)


def test_graph_to_features_to_scoring_to_evaluation(gcfg):
    corpus = _synthetic_corpus()
    records = [
        _record(str(row.text), gcfg, str(row.sample_id)) for row in corpus.itertuples(index=False)
    ]
    features = extract_records(records)

    # No target metadata may reach the feature table.
    assert list(features.columns) == list(META_COLUMNS) + list(FEATURE_NAMES)
    assert not {"label", "split", "source_dataset", "family_id"} & set(features.columns)

    labels = corpus.set_index("sample_id")["label"]
    features = features.set_index("sample_id")
    is_train = labels.index.str[1:].astype(int) < 10
    train_ids, validation_ids = labels.index[is_train], labels.index[~is_train]

    rule = fit_scoring_rule(
        features.loc[train_ids, list(FEATURE_NAMES)],
        labels.loc[train_ids].to_numpy(),
        features.loc[validation_ids, list(FEATURE_NAMES)],
        labels.loc[validation_ids].to_numpy(),
        top_k=5,
        config_id="cfg_integration",
    )
    assert rule["rule_hash"]

    from deterministic.scoring import apply_rule

    validation_features = features.loc[validation_ids, list(FEATURE_NAMES)].reset_index()
    scores, predicted = apply_rule(validation_features[list(FEATURE_NAMES)], rule)

    records_out = _predictions(
        validation_features,
        scores,
        predicted,
        experiment="exp_" + "b" * 20,
        split_identity="split_" + "c" * 20,
        config_identity="cfg_" + "d" * 16,
        build_id="gb_" + "e" * 20,
        threshold=rule["threshold"],
    )
    for record in records_out:
        validate("prediction", record)
        assert "label" not in record

    predictions = pd.DataFrame(records_out)
    truth = pd.DataFrame(
        {
            "sample_id": validation_ids,
            "label": labels.loc[validation_ids].to_numpy(),
            "source_dataset": "SYN",
            "source_task": "synthetic",
            "text_type": "original",
        }
    )
    result = evaluate_predictions(
        predictions, truth, strata=["source_dataset"], min_cell_size=5
    )
    assert result["overall"]["accuracy"] >= 0.9
    assert result["strata"]["source_dataset"]["SYN"]["n"] == len(validation_ids)
