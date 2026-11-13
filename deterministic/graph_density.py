"""Graph size, density and weight features."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

SIZE_FEATURES = (
    "n_nodes",
    "n_edges",
    "n_unique_edges",
    "density",
    "is_directed",
    "self_loop_count",
    "parallel_edge_count",
)

WEIGHT_FEATURES = (
    "edge_weight_mean",
    "edge_weight_std",
    "edge_weight_max",
    "edge_weight_min",
)


def undirected_projection(
    n_nodes: int, edges: list[dict[str, Any]]
) -> tuple[nx.Graph, int, int]:
    """Undirected simple projection where parallel edges are summed."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    self_loops = 0
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        weight = float(edge["weight"])
        if source == target:
            self_loops += 1
            continue
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight)
    return graph, self_loops, graph.number_of_edges()


def size_features(
    *,
    n_nodes: int,
    edges: list[dict[str, Any]],
    self_loops: int,
    unique_pairs: int,
) -> dict[str, float | None]:
    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(len(edges)),
        "n_unique_edges": float(unique_pairs),
        "density": (
            float(2.0 * unique_pairs / (n_nodes * (n_nodes - 1))) if n_nodes >= 2 else None
        ),
        "is_directed": float(any(bool(edge["directed"]) for edge in edges)),
        "self_loop_count": float(self_loops),
        "parallel_edge_count": float(len(edges) - self_loops - unique_pairs),
    }


def weight_features(weights: np.ndarray) -> dict[str, float | None]:
    if weights.size == 0:
        return {name: None for name in WEIGHT_FEATURES}
    return {
        "edge_weight_mean": float(weights.mean()),
        "edge_weight_std": float(weights.std()),
        "edge_weight_max": float(weights.max()),
        "edge_weight_min": float(weights.min()),
    }


__all__ = [
    "SIZE_FEATURES",
    "WEIGHT_FEATURES",
    "size_features",
    "undirected_projection",
    "weight_features",
]
