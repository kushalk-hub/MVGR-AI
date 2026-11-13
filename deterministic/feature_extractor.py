"""Deterministic graph feature extraction (features.v1).

Input is a versioned graph export produced by the graph engine (see
``shared/contracts/graph_schema.json``). This module never reads labels, source,
task, family, text_type, condition or split. It has no classifier and no
thresholds; scoring is a separate stage (``scoring.py``).

The individual feature families live in sibling modules so that each one can be
tested and reviewed independently:

* ``graph_density``    -- size, density, edge weights
* ``degree_distribution`` -- degree statistics
* ``clustering``       -- clustering and triangles
* ``centrality``       -- degree/betweenness/closeness/eigenvector
* ``connectivity``     -- components, articulation points, bridges
* ``shortest_path``    -- path lengths with an explicit size guard
* ``community``        -- seeded Louvain partition
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .centrality import centrality_features
from .clustering import clustering_features
from .community import community_features
from .connectivity import component_features, connected_components, connectivity_features
from .degree_distribution import degree_features, entropy
from .feature_spec import (
    FEATURE_NAMES,
    SPECS,
    assert_registry_complete,
    feature_registry,
)
from .graph_density import size_features, undirected_projection, weight_features
from .shortest_path import shortest_path_features
from shared.contracts.validators import validate_graph_record
from shared.contracts.versions import FEATURE_SET_VERSION

META_COLUMNS = (
    "sample_id",
    "graph_view",
    "graph_spec_version",
    "graph_build_id",
    "config_id",
    "empty",
    "feature_set_version",
)


def _safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _safe_std(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.std(values)) if values else None


def _dependency_depth(edges: list[dict[str, Any]], n_nodes: int) -> tuple[int | None, int]:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))
    for edge in edges:
        source, target = int(edge["source"]), int(edge["target"])
        if source == target:
            continue
        graph.add_edge(source, target)
    if graph.number_of_edges() == 0:
        return None, 0
    roots = [node for node in sorted(graph.nodes) if graph.in_degree(node) == 0]
    if not roots:
        return None, 0
    try:
        order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        return None, len(roots)
    depth = {node: 0 for node in roots}
    for node in order:
        if node not in depth:
            depth[node] = 0
        for child in sorted(graph.successors(node)):
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
    return int(max(depth.values())), len(roots)


def _view_features(record: dict[str, Any], edges: list[dict[str, Any]], n_nodes: int) -> dict[str, Any]:
    node_types = [str(node.get("node_type", "token")) for node in record["nodes"]]
    edge_types = [str(edge.get("edge_type", "")) for edge in edges]
    tokens = [
        str(node.get("token", ""))
        for node in record["nodes"]
        if node.get("node_type", "token") == "token"
    ]
    stopwords = sum(1 for node in record["nodes"] if node.get("is_stopword") is True)

    features: dict[str, Any] = {
        "node_type_token": float(node_types.count("token")),
        "node_type_punctuation": float(node_types.count("punctuation")),
        "node_type_sentence_id": float(node_types.count("sentence_id")),
        "edge_type_cooccurrence": float(edge_types.count("cooccurrence")),
        "edge_type_dependency": float(edge_types.count("dependency")),
        "stopword_fraction": float(stopwords / n_nodes) if n_nodes else None,
        "token_char_len_mean": _safe_mean(len(token) for token in tokens),
        "token_char_len_std": _safe_std(len(token) for token in tokens),
        "unique_token_ratio": float(len(set(tokens)) / n_nodes) if n_nodes else None,
        "token_repetition_rate": (
            float(1.0 - len(set(tokens)) / n_nodes) if n_nodes else None
        ),
    }

    dependency_labels = [
        str(edge.get("dependency", ""))
        for edge in edges
        if edge.get("edge_type") == "dependency"
    ]
    if dependency_labels:
        label_counts = np.asarray(
            [dependency_labels.count(label) for label in sorted(set(dependency_labels))],
            dtype=float,
        )
        features["dependency_label_count"] = float(len(set(dependency_labels)))
        features["dependency_entropy"] = entropy(label_counts)
    else:
        features["dependency_label_count"] = None
        features["dependency_entropy"] = None

    distances = [
        float(edge.get("distance", 0))
        for edge in edges
        if edge.get("edge_type") == "dependency" and edge.get("distance") is not None
    ]
    features["dependency_distance_mean"] = _safe_mean(distances)
    features["dependency_distance_max"] = max(distances) if distances else None

    depth, roots = _dependency_depth(edges, n_nodes)
    features["tree_depth"] = float(depth) if depth is not None else None
    features["root_count"] = float(roots) if edges else None
    return features


def extract_graph_features(record: dict[str, Any]) -> dict[str, object]:
    """Compute the full features.v1 vector for one graph export record."""
    n_nodes = len(record["nodes"])
    edges = sorted(
        record["edges"],
        key=lambda edge: (
            int(edge["source"]),
            int(edge["target"]),
            str(edge.get("edge_type", "")),
        ),
    )
    weights = np.asarray([float(edge["weight"]) for edge in edges], dtype=float)

    graph, self_loops, unique_pairs = undirected_projection(n_nodes, edges)
    degrees = np.asarray([degree for _, degree in sorted(graph.degree())], dtype=float)
    components = connected_components(graph) if n_nodes else []
    largest = max(components, key=len) if components else set()

    features: dict[str, Any] = {}
    features.update(
        size_features(
            n_nodes=n_nodes, edges=edges, self_loops=self_loops, unique_pairs=unique_pairs
        )
    )
    features.update(degree_features(degrees))
    features.update(clustering_features(graph, n_nodes=n_nodes))
    features.update(centrality_features(graph, n_nodes=n_nodes, largest=largest))
    features.update(
        component_features(graph, n_nodes=n_nodes, components=components)
    )
    features.update(
        connectivity_features(
            graph, n_nodes=n_nodes, components=components, unique_pairs=unique_pairs
        )
    )
    features.update(
        shortest_path_features(graph, n_nodes=n_nodes, components=components)
    )
    features.update(community_features(graph, n_nodes=n_nodes))
    features.update(weight_features(weights))
    features.update(_view_features(record, edges, n_nodes))

    assert_registry_complete(features)
    for name in FEATURE_NAMES:
        value = features[name]
        if isinstance(value, (int, float, np.floating)) and not np.isfinite(float(value)):
            features[name] = None
    return features


def extract_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        validate_graph_record(record)
        features = extract_graph_features(record)
        row: dict[str, object] = {
            "sample_id": record["sample_id"],
            "graph_view": record["graph_view"],
            "graph_spec_version": record["graph_spec_version"],
            "graph_build_id": record["graph_build_id"],
            "config_id": record["config_id"],
            "empty": bool(record["empty"]),
            "feature_set_version": FEATURE_SET_VERSION,
        }
        row.update(features)
        rows.append(row)
    order = list(META_COLUMNS) + list(FEATURE_NAMES)
    return pd.DataFrame(rows).reindex(columns=order)


def extract_export(
    export_path: str | Path,
    *,
    output_path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract features from a JSONL graph export and persist a Parquet table."""
    export_path = Path(export_path)
    records = []
    with export_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{export_path}:{line_number}: invalid JSON: {exc}") from exc

    frame = extract_records(records)
    output_path = Path(output_path) if output_path else export_path.with_name(
        f"{export_path.stem}.features.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    registry = feature_registry()
    registry_path = Path(registry_path) if registry_path else output_path.with_suffix(".registry.json")
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    feature_columns = list(FEATURE_NAMES)
    return {
        "export": str(export_path),
        "output": str(output_path),
        "registry": str(registry_path),
        "rows": int(len(frame)),
        "empty_graphs": int(frame["empty"].sum()),
        "undefined_rates": {name: float(frame[name].isna().mean()) for name in feature_columns},
        "feature_set_version": FEATURE_SET_VERSION,
        "checksum": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "spec_count": len(SPECS),
    }
