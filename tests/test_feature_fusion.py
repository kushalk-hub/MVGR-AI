"""Module E tests — multi-view feature fusion.

Synthetic frames only. The graph branch is exercised with a hand-built
``graph_features``-shaped frame so no database or file access is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion.multiview_feature_fusion import (  # noqa: E402
    FusionError,
    fuse_multiview,
    normalize_features,
    select_features,
    split_graph_branches,
)

IDS = ["s1", "s2", "s3", "s4"]


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": IDS,
            "label": [0, 1, 0, 1],
            "source_dataset": ["A", "A", "B", "B"],
            "source_task": ["t"] * 4,
            "text_type": ["original"] * 4,
            "graph_category": ["original"] * 4,
            "paraphrase_status": ["False"] * 4,
            "domain": ["Unknown"] * 4,
            "ai_model": ["Unknown"] * 4,
            "padben_split": ["train"] * 4,
            "split": ["train"] * 4,
        }
    )


def _stylometric() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": IDS,
            "styl_a": [1.0, 2.0, 3.0, 4.0],
            "styl_b": [0.4, 0.1, 0.9, 0.2],
            "styl_constant": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _writeprints() -> pd.DataFrame:
    return pd.DataFrame({"sample_id": IDS, "wp_a": [0.1, 0.5, 0.2, 0.7]})


def _graph_features() -> pd.DataFrame:
    rows = []
    for sample_id in IDS:
        for view in ("lexical_cooccurrence", "syntactic_dependency"):
            rows.append(
                {
                    "sample_id": sample_id,
                    "view": view,
                    "build_id": "b1",
                    "graph_spec_version": "1.0.0",
                    "spacy_version": "3.8.18",
                    "model_name": "en_core_web_sm",
                    "parser_status": "ok",
                    "label": 0,
                    "source_dataset": "A",
                    "source_task": "t",
                    "text_type": "original",
                    "graph_category": "original",
                    "paraphrase_status": "False",
                    "domain": "Unknown",
                    "ai_model": "Unknown",
                    "padben_split": "train",
                    "split": "train",
                    "group_id": "g",
                    "source_row_id": "1",
                    "n_nodes": 5.0,
                    "n_edges": 4.0,
                    "density": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_split_graph_branches_maps_views_and_prefixes():
    branches = split_graph_branches(_graph_features())
    assert set(branches) == {"lexical_graph", "syntactic_graph"}
    assert "glex_n_nodes" in branches["lexical_graph"].columns
    assert "gsyn_n_nodes" in branches["syntactic_graph"].columns
    # identifiers / metadata are never carried into the branch
    assert "label" not in branches["lexical_graph"].columns
    assert "graph_id" not in branches["lexical_graph"].columns


def test_fuse_keeps_metadata_separate():
    result = fuse_multiview(
        branch_frames={"stylometric": _stylometric(), "writeprints": _writeprints()},
        metadata=_metadata(),
    )
    assert list(result.features["sample_id"]) == IDS
    assert "label" not in result.features.columns
    assert "label" in result.metadata.columns
    assert result.features.shape[1] == 1 + result.report["features"]
    # A constant feature is dropped by the default selection.
    assert "stylometric__styl_constant" not in result.features.columns


def test_identifier_columns_never_become_features():
    result = fuse_multiview(
        branch_frames={"stylometric": _stylometric()},
        metadata=_metadata(),
        config={"identifier_columns": ["sample_id", "graph_id", "build_id"]},
    )
    for column in result.features.columns:
        assert column not in {"graph_id", "build_id", "label", "split"}


def test_duplicate_sample_ids_are_rejected():
    metadata = pd.concat([_metadata(), _metadata().head(1)], ignore_index=True)
    with pytest.raises(FusionError):
        fuse_multiview(
            branch_frames={"stylometric": _stylometric()}, metadata=metadata
        )


def test_missing_branch_is_recorded_not_imputed():
    result = fuse_multiview(
        branch_frames={"stylometric": _stylometric(), "gltr": None},
        metadata=_metadata(),
    )
    assert result.report["availability"]["gltr"] is False
    assert result.report["availability"]["stylometric"] is True
    assert not any(column.startswith("gltr__") for column in result.features.columns)
    assert (result.metadata["has_gltr"] == 0).all()


def test_watermark_is_metadata_by_default():
    watermark = pd.DataFrame(
        {
            "sample_id": IDS,
            "watermark_status": ["unsupported"] * 4,
            "watermark_z_score": [None] * 4,
        }
    )
    result = fuse_multiview(
        branch_frames={"stylometric": _stylometric()},
        metadata=_metadata(),
        watermark_frame=watermark,
    )
    assert not any(column.startswith("watermark__") for column in result.features.columns)
    assert "watermark_status" in result.metadata.columns


def test_watermark_as_feature_is_opt_in():
    watermark = pd.DataFrame(
        {
            "sample_id": IDS,
            "watermark_status": ["detected", "not_detected", "detected", "unsupported"],
            "watermark_z_score": [5.0, 1.0, 6.0, 2.0],
            "watermark_detected": [True, False, True, None],
        }
    )
    # Disable selection so the test isolates the opt-in wiring, not the filter.
    config = {
        "watermark_as_feature": True,
        "feature_selection": {"drop_constant": False, "drop_high_correlation": False},
    }
    result = fuse_multiview(
        branch_frames={"stylometric": _stylometric()},
        metadata=_metadata(),
        watermark_frame=watermark,
        config=config,
    )
    assert any(column.startswith("watermark__") for column in result.features.columns)
    assert "watermark__watermark_z_score" in result.features.columns


def test_select_features_drops_missing_constant_and_correlated():
    frame = pd.DataFrame(
        {
            "sample_id": IDS,
            "keep": [1.0, 2.0, 3.0, 4.0],
            "mostly_missing": [None, None, None, 1.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "correlated": [2.0, 4.0, 6.0, 8.0],  # perfectly correlated with 'keep'
        }
    )
    selected, report = select_features(
        frame, max_missing_fraction=0.30, high_correlation_threshold=0.98
    )
    assert "mostly_missing" in report["dropped_missing"]
    assert "constant" in report["dropped_constant"]
    assert "correlated" in report["dropped_correlated"]
    assert list(selected.columns) == ["sample_id", "keep"]


def test_normalize_features_zscore():
    frame = pd.DataFrame({"sample_id": IDS, "x": [1.0, 2.0, 3.0, 4.0]})
    normalized, report = normalize_features(frame, "zscore")
    assert abs(float(normalized["x"].mean())) < 1e-9
    assert report["method"] == "zscore"


def test_normalize_unknown_method_raises():
    with pytest.raises(FusionError):
        normalize_features(pd.DataFrame({"sample_id": IDS, "x": [1.0] * 4}), "quantum")


def test_schema_records_experiment_branch_columns():
    config = {
        "experiments": {
            "E3": {"branches": ["stylometric"]},
            "E6": {"branches": ["lexical_graph", "syntactic_graph"]},
        }
    }
    result = fuse_multiview(
        branch_frames={
            "stylometric": _stylometric(),
            "lexical_graph": split_graph_branches(_graph_features())["lexical_graph"],
            "syntactic_graph": split_graph_branches(_graph_features())["syntactic_graph"],
        },
        metadata=_metadata(),
        config=config,
    )
    schema = result.schema
    assert schema["feature_set_version"].startswith("fingerprint")
    assert schema["experiments"]["E3"]  # non-empty list of columns
    assert any(c.startswith("lexical_graph__") for c in schema["experiments"]["E6"])
    assert "watermark" in schema["branches"]


def test_fused_features_have_no_infinities():
    result = fuse_multiview(
        branch_frames={"stylometric": _stylometric(), "writeprints": _writeprints()},
        metadata=_metadata(),
    )
    values = result.features.drop(columns=["sample_id"]).to_numpy(dtype=float)
    assert not np.isinf(values).any()
