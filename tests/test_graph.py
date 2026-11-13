"""Graph builder, contract and export tests.

The lexical view must be deterministic, contract-valid, and free of every
target-metadata field. The syntactic view must fail loudly when spaCy is absent
rather than fall back to the lexical view.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from graph_engine.graph_builder import build_graph
from graph_engine.graph_builder.syntactic_graph import ParserUnavailable, load_pipeline
from graph_engine.graph_export.export import (
    ExportError,
    build_view_export,
    read_export,
    write_view_export,
)
from shared.contracts.ids import graph_build_id
from shared.contracts.validators import validate_graph_record
from shared.paths import graph_config

FORBIDDEN_KEYS = {
    "label",
    "source_dataset",
    "source_task",
    "family_id",
    "text_type",
    "condition",
    "split",
}
BUILD_ID = "gb_" + "a" * 20
CONFIG_ID = "cfg_" + "b" * 16


@pytest.fixture(scope="module")
def gcfg() -> dict:
    return graph_config()


def _view_config(gcfg: dict, view: str = "lexical") -> dict:
    config = dict(gcfg["views"][view])
    config["shared_max_nodes"] = int(gcfg["shared"]["max_nodes"])
    return config


def _lexical(text: str, gcfg: dict, sample_id: str = "s1") -> dict:
    return build_graph(
        "lexical",
        sample_id=sample_id,
        text=text,
        graph_build_id=BUILD_ID,
        config_id=CONFIG_ID,
        view_config=_view_config(gcfg),
        graph_spec_version=str(gcfg["graph_spec_version"]),
    )


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def test_lexical_graph_is_contract_valid_and_metadata_free(gcfg):
    record = _lexical(
        "The quick brown fox jumps over the lazy dog. The dog sleeps.", gcfg
    )
    validate_graph_record(record)

    assert not FORBIDDEN_KEYS & set(record)
    assert record["graph_view"] == "lexical"
    assert record["empty"] is False
    assert record["nodes"]
    assert [node["node_id"] for node in record["nodes"]] == list(range(len(record["nodes"])))
    assert all(edge["edge_type"] == "cooccurrence" for edge in record["edges"])
    assert all(edge["directed"] is False for edge in record["edges"])
    assert all(edge["weight"] >= 1 for edge in record["edges"])
    assert all(edge["distance"] >= 1 for edge in record["edges"])


def test_lexical_graph_is_deterministic(gcfg):
    text = "Deterministic graphs matter for reproducible research."
    assert _lexical(text, gcfg) == _lexical(text, gcfg)


def test_empty_text_emits_an_empty_graph_record(gcfg):
    record = _lexical("   \n\t  ", gcfg)
    validate_graph_record(record)
    assert record["empty"] is True
    assert record["nodes"] == []
    assert record["edges"] == []


def test_single_token_produces_one_node_and_no_edges(gcfg):
    record = _lexical("solitary", gcfg)
    validate_graph_record(record)
    assert len(record["nodes"]) == 1
    assert record["edges"] == []


def test_unknown_view_is_rejected(gcfg):
    with pytest.raises(ValueError, match="unknown graph view"):
        build_graph(
            "not-a-view",
            sample_id="s1",
            text="hello world",
            graph_build_id=BUILD_ID,
            config_id=CONFIG_ID,
            view_config={},
            graph_spec_version=str(gcfg["graph_spec_version"]),
        )


def test_syntactic_view_requires_a_parser_or_fails(gcfg):
    """If spaCy is present the record validates; otherwise the build fails."""
    try:
        load_pipeline(str(gcfg["views"]["syntactic"].get("model", "en_core_web_sm")))
    except ParserUnavailable:
        with pytest.raises(ParserUnavailable):
            build_graph(
                "syntactic",
                sample_id="s1",
                text="Dependency parsing needs a model.",
                graph_build_id=BUILD_ID,
                config_id=CONFIG_ID,
                view_config=_view_config(gcfg, "syntactic"),
                graph_spec_version=str(gcfg["graph_spec_version"]),
            )
        return

    record = build_graph(
        "syntactic",
        sample_id="s1",
        text="Dependency parsing needs a model.",
        graph_build_id=BUILD_ID,
        config_id=CONFIG_ID,
        view_config=_view_config(gcfg, "syntactic"),
        graph_spec_version=str(gcfg["graph_spec_version"]),
    )
    validate_graph_record(record)
    assert all(edge["edge_type"] == "dependency" for edge in record["edges"])


# --------------------------------------------------------------------------- #
# build identity
# --------------------------------------------------------------------------- #
def test_graph_build_id_ignores_sample_order():
    kwargs = dict(
        dataset_version="v",
        split_id_value="split_x",
        graph_spec_version="1.0.0",
        config={"view": "lexical"},
    )
    assert graph_build_id(sample_ids=["a", "b"], **kwargs) == graph_build_id(
        sample_ids=["b", "a"], **kwargs
    )


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_write_view_export_round_trips_and_hashes(gcfg, tmp_path):
    records = [
        _lexical("alpha beta gamma delta", gcfg, sample_id="s1"),
        _lexical("epsilon zeta eta theta", gcfg, sample_id="s2"),
    ]
    manifest = write_view_export(
        records,
        output_dir=tmp_path,
        build_meta={"graph_build_id": BUILD_ID, "graph_view": "lexical"},
    )

    assert manifest["counts"]["samples"] == 2
    assert manifest["counts"]["nodes"] > 0
    exported = list(read_export(tmp_path / "lexical.jsonl"))
    assert [record["sample_id"] for record in exported] == ["s1", "s2"]
    for record in exported:
        validate_graph_record(record)
        assert not FORBIDDEN_KEYS & set(record)


def test_write_view_export_rejects_duplicate_sample_ids(gcfg, tmp_path):
    records = [
        _lexical("alpha beta gamma", gcfg, sample_id="dup"),
        _lexical("delta epsilon zeta", gcfg, sample_id="dup"),
    ]
    with pytest.raises(ExportError, match="duplicate sample_id"):
        write_view_export(
            records,
            output_dir=tmp_path,
            build_meta={"graph_build_id": BUILD_ID, "graph_view": "lexical"},
        )


def test_build_view_export_refuses_target_metadata(gcfg, tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "text": ["alpha beta gamma delta"],
            "label": [1],
        }
    )
    with pytest.raises(ExportError, match="target metadata"):
        build_view_export(
            frame=frame,
            sample_ids=["s1"],
            view="lexical",
            output_root=tmp_path,
            dataset_version="v",
            split_id="split_x",
            graph_config=gcfg,
        )


def test_build_view_export_writes_manifest_and_pointer(gcfg, tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "text": ["alpha beta gamma delta", "epsilon zeta eta theta"],
        }
    )
    manifest = build_view_export(
        frame=frame,
        sample_ids=["s1", "s2"],
        view="lexical",
        output_root=tmp_path,
        dataset_version="v",
        split_id="split_x",
        graph_config=gcfg,
    )
    assert manifest["target_metadata_in_export"] is False
    assert manifest["counts"]["samples"] == 2
    pointer = json.loads((tmp_path / "graph_builds" / "LATEST.json").read_text())
    assert pointer["graph_build_id"] == manifest["graph_build_id"]
