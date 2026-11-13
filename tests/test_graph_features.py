"""Graph feature extraction and ML-dataset preparation tests (Phases B-E).

The pure feature, selection and leakage functions are exercised on small
synthetic graphs so they run without a database. The live-database test is
skipped automatically when Neo4j credentials are not configured, and it only
ever reads.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.paths import load_config  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


egf = _load("extract_graph_features", "extract_graph_features.py")
pml = _load("prepare_ml_dataset", "prepare_ml_dataset.py")
vfy = _load("verify_neo4j_graphs", "verify_neo4j_graphs.py")

POS_VOCAB = ["ADP", "DET", "NOUN", "VERB"]
DEP_VOCAB = ["det", "nsubj", "obl"]


# --------------------------------------------------------------------------- #
# synthetic graphs
# --------------------------------------------------------------------------- #
def _node(index, surface, lemma, pos, tag, *, is_stop=False, is_punct=False, ent=""):
    return {
        "node_id": f"g::{index}", "position": index, "surface": surface,
        "normalized": surface.lower(), "lemma": lemma, "pos": pos, "tag": tag,
        "ent_type": ent, "is_punct": is_punct, "is_stop": is_stop,
    }


def _lexical_nodes():
    return [
        _node(0, "The", "the", "DET", "DT", is_stop=True),
        _node(1, "cat", "cat", "NOUN", "NN"),
        _node(2, "sat", "sit", "VERB", "VBD"),
        _node(3, "down", "down", "ADP", "IN", is_stop=True),
    ]


def _lexical_edges():
    return [
        {"source": "g::0", "target": "g::1", "rel_type": "CO_OCCURS", "distance": 1, "dependency_label": None},
        {"source": "g::1", "target": "g::2", "rel_type": "CO_OCCURS", "distance": 1, "dependency_label": None},
        {"source": "g::2", "target": "g::3", "rel_type": "CO_OCCURS", "distance": 1, "dependency_label": None},
        {"source": "g::0", "target": "g::2", "rel_type": "CO_OCCURS", "distance": 2, "dependency_label": None},
    ]


def _syntactic_edges():
    # root is g::2 ("sat"): sat -nsubj-> cat, cat -det-> The, sat -obl-> down
    return [
        {"source": "g::2", "target": "g::1", "rel_type": "DEPENDS_ON", "distance": 1, "dependency_label": "nsubj"},
        {"source": "g::2", "target": "g::3", "rel_type": "DEPENDS_ON", "distance": 1, "dependency_label": "obl"},
        {"source": "g::1", "target": "g::0", "rel_type": "DEPENDS_ON", "distance": 1, "dependency_label": "det"},
    ]


def _features(nodes, edges, view):
    return egf.graph_view_features(
        nodes, edges, view=view, pos_vocab=POS_VOCAB, dep_vocab=DEP_VOCAB,
        guards={"max_nodes_for_shortest_path": 5000},
    )


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def test_sanitize_makes_safe_column_names():
    assert egf.sanitize("ADP") == "adp"
    assert egf.sanitize("obj") == "obj"
    assert egf.sanitize("nsubj-pass") == "nsubj_pass"
    assert egf.sanitize(None) == "none"


def test_entropy_and_histogram():
    assert math.isclose(egf.entropy([1, 1]), math.log(2), rel_tol=1e-9)
    assert egf.entropy([]) is None
    hist = egf.degree_histogram(np.asarray([0, 1, 1, 5]))
    assert hist.shape == (egf.N_DEGREE_BINS,)
    assert math.isclose(float(hist.sum()), 1.0, rel_tol=1e-9)


def test_cosine_and_jensen_shannon():
    assert math.isclose(egf.cosine_similarity([1, 0], [1, 0]), 1.0, rel_tol=1e-9)
    assert egf.cosine_similarity([0, 0], [1, 0]) is None
    assert math.isclose(egf.jensen_shannon_similarity([1, 0], [1, 0]), 1.0, rel_tol=1e-9)
    assert egf.jensen_shannon_similarity([0, 0], [1, 0]) is None


# --------------------------------------------------------------------------- #
# per-view feature computation
# --------------------------------------------------------------------------- #
def test_lexical_view_features():
    features = _features(_lexical_nodes(), _lexical_edges(), "lexical_cooccurrence")
    assert features["n_nodes"] == 4
    assert features["n_edges"] == 4
    assert features["n_unique_edges"] == 4
    assert features["self_loop_count"] == 0
    assert features["token_count"] == 4
    assert features["unique_token_count"] == 4
    assert math.isclose(features["vocabulary_diversity"], 1.0)
    assert math.isclose(features["stopword_ratio"], 0.5)
    assert math.isclose(features["punctuation_ratio"], 0.0)
    assert features["n_pos_tags"] == 4
    assert features["is_directed"] is False
    # A lexical view carries no dependency edges: syntactic features are undefined.
    assert features["dep_edge_count"] is None
    assert features["dep_det_count"] is None


def test_syntactic_view_features():
    features = _features(_lexical_nodes(), _syntactic_edges(), "syntactic_dependency")
    assert features["dep_edge_count"] == 3
    assert features["dep_relation_diversity"] == 3
    assert features["root_count"] == 1
    assert features["dep_det_count"] == 1
    assert features["dep_nsubj_count"] == 1
    assert features["dep_other_count"] == 0
    assert features["tree_depth"] == 2
    assert features["is_directed"] is True


def test_self_loops_and_parallel_edges_are_counted():
    nodes = [_node(0, "a", "a", "NOUN", "NN"), _node(1, "b", "b", "NOUN", "NN")]
    edges = [
        {"source": "g::0", "target": "g::0", "rel_type": "CO_OCCURS", "distance": 0, "dependency_label": None},
        {"source": "g::0", "target": "g::1", "rel_type": "CO_OCCURS", "distance": 1, "dependency_label": None},
        {"source": "g::1", "target": "g::0", "rel_type": "CO_OCCURS", "distance": 1, "dependency_label": None},
    ]
    features = _features(nodes, edges, "lexical_cooccurrence")
    assert features["self_loop_count"] == 1
    assert features["n_unique_edges"] == 1
    assert features["parallel_edge_count"] == 1


def test_path_stats_match_networkx():
    """The fast SciPy path statistics must equal NetworkX's exact values."""
    import networkx as nx

    for seed in range(30):
        graph = nx.gnp_random_graph(10, 0.4, seed=seed)
        component = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(component)
        if subgraph.number_of_nodes() < 2:
            continue
        stats = egf._path_stats(subgraph)
        assert stats is not None
        avg, diameter, pairs = stats
        assert math.isclose(avg, nx.average_shortest_path_length(subgraph), rel_tol=1e-9)
        assert diameter == nx.diameter(subgraph)
        assert pairs == subgraph.number_of_nodes() * (subgraph.number_of_nodes() - 1) // 2


# --------------------------------------------------------------------------- #
# node-level features
# --------------------------------------------------------------------------- #
def test_node_feature_rows():
    nodes = _lexical_nodes()
    edges = _lexical_edges()
    graph_row = {"sample_id": "s1", "graph_id": "g", "view": "lexical_cooccurrence"}
    rows = egf.node_feature_rows(graph_row, nodes, edges)
    assert len(rows) == 4
    by_id = {row["node_id"]: row for row in rows}
    assert by_id["g::1"]["lexical_degree"] == 2
    assert by_id["g::2"]["lexical_degree"] == 3
    assert by_id["g::0"]["is_first_token"] is True
    assert by_id["g::3"]["is_last_token"] is True
    assert by_id["g::1"]["is_repeated"] is False
    assert by_id["g::0"]["sample_id"] == "s1"


# --------------------------------------------------------------------------- #
# multi-view aggregation
# --------------------------------------------------------------------------- #
def test_multiview_features():
    lexical = _features(_lexical_nodes(), _lexical_edges(), "lexical_cooccurrence")
    syntactic = _features(_lexical_nodes(), _syntactic_edges(), "syntactic_dependency")
    multi = egf.multiview_features(lexical, syntactic)
    assert multi["n_nodes_lexical"] == 4
    assert multi["n_nodes_syntactic"] == 4
    assert multi["node_count_diff"] == 0
    assert multi["edge_count_diff"] == 3 - 4
    assert multi["density_diff"] is not None
    assert 0.0 <= multi["degree_dist_js_similarity"] <= 1.0
    assert -1.0 <= multi["structural_cosine_similarity"] <= 1.0


# --------------------------------------------------------------------------- #
# feature selection gates
# --------------------------------------------------------------------------- #
def test_select_features_drops_missing_constant_and_correlated():
    frame = pd.DataFrame({
        "f_const": [1.0, 1.0, 1.0, 1.0, 1.0],
        "f_missing": [np.nan, np.nan, np.nan, np.nan, 1.0],
        "f_corr1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f_corr2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "f_good": [5.0, 3.0, 9.0, 1.0, 7.0],
    })
    result = pml.select_features(
        frame, ["f_const", "f_missing", "f_corr1", "f_corr2", "f_good"],
        max_missing_fraction=0.30, constant_tolerance=1e-12, correlation_threshold=0.98,
    )
    assert "f_missing" in result["dropped_missing"]
    assert "f_const" in result["dropped_constant"]
    assert any(dropped == "f_corr2" for _, dropped, _ in result["dropped_correlated"])
    assert result["selected"] == ["f_corr1", "f_good"]


def test_missing_summary():
    frame = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
    summary = pml.missing_summary(frame, ["a"])
    assert math.isclose(summary["a"], 0.5)


# --------------------------------------------------------------------------- #
# leakage checks
# --------------------------------------------------------------------------- #
def test_check_no_identifier_features():
    assert pml.check_no_identifier_features(["a", "b"], ["a"])["passed"] is False
    assert pml.check_no_identifier_features(["b", "c"], ["a"])["passed"] is True


def test_check_label_not_a_feature():
    assert pml.check_label_not_a_feature(["a", "b"], "label")["passed"] is True
    assert pml.check_label_not_a_feature(["a", "label"], "label")["passed"] is False


def test_check_graph_view_split_consistency():
    good = pd.DataFrame({"sample_id": ["s1", "s1", "s2", "s2"], "split": ["train"] * 4})
    assert pml.check_graph_view_split_consistency(good)["passed"] is True
    bad = pd.DataFrame({"sample_id": ["s1", "s1", "s2", "s2"],
                        "split": ["train", "test", "train", "train"]})
    assert pml.check_graph_view_split_consistency(bad)["passed"] is False


def test_check_group_split_purity():
    metadata = pd.DataFrame({
        "group_id": ["g1", "g1", "g2", "g2"],
        "split": ["train", "test", "train", "train"],
    })
    result = pml.check_group_split_purity(metadata)
    assert result["passed"] is False
    assert result["groups_with_multiple_splits"] == 1


# --------------------------------------------------------------------------- #
# verification checks on synthetic stats
# --------------------------------------------------------------------------- #
def _consistent_stats(config):
    g = config["graph"]
    return {
        "total_nodes": g["expected_nodes"],
        "total_relationships": g["expected_relationships"],
        "nodes_by_label": {"Sample": 500, "TextGraph": 1000, "TokenOccurrence": 102822},
        "relationships_by_type": {"HAS_GRAPH": 1000, "HAS_TOKEN": 102822,
                                  "CO_OCCURS": 101322, "DEPENDS_ON": 48884},
        "samples": 500,
        "distinct_sample_ids": 500,
        "graphs": 1000,
        "graphs_by_view": {"lexical_cooccurrence": 500, "syntactic_dependency": 500},
        "tokens": 102822,
        "samples_with_graphs": 500,
        "samples_with_two_views": 500,
        "samples_missing_views": 0,
        "duplicate_view_pairs": 0,
        "orphan_tokens": 0,
        "graphs_without_tokens": 0,
        "samples_without_graphs": 0,
        "dangling_graphs": 0,
        "cross_graph_edges": 0,
        "token_count_mismatches": 0,
        "edge_count_mismatches": 0,
        "build_ids": [g["build_id"]],
        "graphs_in_expected_build": 1000,
        "dangling_token_graph_ids": 0,
        "edges_by_type_and_view": {"CO_OCCURS": 101322, "DEPENDS_ON": 48884},
    }


def _statuses(checks):
    return {check["check"]: check["status"] for check in checks}


def test_verification_passes_on_consistent_stats():
    config = load_config("graph_feature_config")
    checks = vfy.run_checks(_consistent_stats(config), config)
    assert all(check["status"] == vfy.PASS for check in checks)


def test_verification_detects_missing_view():
    config = load_config("graph_feature_config")
    stats = _consistent_stats(config)
    stats["samples_missing_views"] = 1
    statuses = _statuses(vfy.run_checks(stats, config))
    assert statuses["two_views_per_sample"] == vfy.FAIL


def test_verification_detects_cross_graph_edge():
    config = load_config("graph_feature_config")
    stats = _consistent_stats(config)
    stats["cross_graph_edges"] = 7
    statuses = _statuses(vfy.run_checks(stats, config))
    assert statuses["cross_graph_edges"] == vfy.FAIL


def test_verification_detects_relationship_total_slip():
    config = load_config("graph_feature_config")
    stats = _consistent_stats(config)
    stats["total_relationships"] = 253528  # the master prompt's arithmetic slip
    statuses = _statuses(vfy.run_checks(stats, config))
    assert statuses["total_relationships"] == vfy.FAIL
    assert statuses["relationship_total_consistency"] == vfy.FAIL


# --------------------------------------------------------------------------- #
# formal-split guard
# --------------------------------------------------------------------------- #
def test_approved_manifest_absent_blocks_formal_split():
    config = load_config("graph_feature_config")
    status = pml.approved_manifest_status(config)
    # The repository currently ships only candidate manifests, no approved one.
    assert status["formal_split_generated"] is False
    if not status["approved_manifest_found"]:
        assert "No approved split manifest" in status["reason"]


# --------------------------------------------------------------------------- #
# end-to-end preparation on synthetic tables
# --------------------------------------------------------------------------- #
def _synthetic_prepare_inputs(tmp_path):
    schema = {
        "build_id": "test_build",
        "tables": {
            "sample_multiview_features": {
                "columns": [
                    {"name": "sample_id", "predictive": False, "leakage_role": "identifier"},
                    {"name": "group_id", "predictive": False, "leakage_role": "identifier"},
                    {"name": "label", "predictive": False, "leakage_role": "metadata"},
                    {"name": "source_dataset", "predictive": False, "leakage_role": "metadata"},
                    {"name": "split", "predictive": False, "leakage_role": "metadata"},
                    {"name": "f_a", "predictive": True, "leakage_role": "feature"},
                    {"name": "f_b", "predictive": True, "leakage_role": "feature"},
                    {"name": "f_const", "predictive": True, "leakage_role": "feature"},
                ]
            }
        },
    }
    samples = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4"],
        "group_id": ["g1", "g2", "g3", "g4"],
        "label": ["ai", "human", "ai", "human"],
        "source_dataset": ["PADBen", "PADBen", "BiScope", "BiScope"],
        "split": ["train", "train", "test", "test"],
        "f_a": [1.0, 2.0, 3.0, 4.0],
        "f_b": [4.0, 1.0, 2.0, 7.0],
        "f_const": [1.0, 1.0, 1.0, 1.0],
    })
    graph_features = pd.DataFrame({
        "sample_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        "view": ["lexical_cooccurrence", "syntactic_dependency"] * 4,
        "split": ["train", "train", "train", "train", "test", "test", "test", "test"],
    })
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    samples.to_csv(tmp_path / "sample_multiview_features.csv", index=False)
    graph_features.to_csv(tmp_path / "graph_features.csv", index=False)
    config = {
        "outputs": {
            "graph_features": str(tmp_path / "graph_features.csv"),
            "sample_multiview_features": str(tmp_path / "sample_multiview_features.csv"),
            "feature_schema": str(tmp_path / "feature_schema.json"),
            "ml_features": str(tmp_path / "ml_ready_features.csv"),
            "ml_metadata": str(tmp_path / "ml_ready_metadata.csv"),
            "ml_report": str(tmp_path / "ml_dataset_report.json"),
        },
        "validation": {
            "max_missing_fraction": 0.30,
            "constant_feature_tolerance": 1e-12,
            "high_correlation_threshold": 0.98,
        },
        "leakage": {
            "approved_manifest_pointer": "data/manifests/does_not_exist.json",
            "identifier_columns": ["sample_id", "group_id"],
            "metadata_columns": ["label", "source_dataset", "split"],
        },
    }
    return config


def test_prepare_end_to_end_separates_features_and_metadata(tmp_path):
    config = _synthetic_prepare_inputs(tmp_path)
    report = pml.prepare(config)

    features = pd.read_csv(tmp_path / "ml_ready_features.csv")
    metadata = pd.read_csv(tmp_path / "ml_ready_metadata.csv")

    # No identifier or metadata column may appear in the feature matrix.
    assert not ({"sample_id", "group_id", "label", "source_dataset", "split"}
                & set(features.columns))
    # The constant feature is dropped; the two informative features remain.
    assert set(features.columns) == {"f_a", "f_b"}
    # The target is preserved, encoded, only in the metadata table.
    assert "label" in metadata.columns
    assert "label_encoded" in metadata.columns
    assert set(metadata["label_encoded"]) == {0, 1}
    # Row alignment and leakage checks.
    assert report["leakage_passed"] is True
    assert report["formal_split"]["formal_split_generated"] is False
    assert report["training_performed"] is False
    assert len(features) == len(metadata) == 4


# --------------------------------------------------------------------------- #
# live database (read-only; skipped without credentials)
# --------------------------------------------------------------------------- #
def test_live_database_connectivity_and_counts():
    config = load_config("graph_feature_config")
    settings = egf.Settings(config)
    if settings.missing():
        pytest.skip("Neo4j credentials are not configured")

    driver = egf.connect(settings)
    try:
        with driver.session(database=settings.database) as session:
            stats = vfy.collect_stats(session, config)
    finally:
        driver.close()

    assert stats["samples"] == config["graph"]["expected_samples"]
    assert stats["graphs"] == config["graph"]["expected_graphs"]
    checks = vfy.run_checks(stats, config)
    failures = [check["check"] for check in checks if check["status"] == vfy.FAIL]
    assert not failures, f"live verification failures: {failures}"
