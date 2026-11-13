"""Component and connectivity features.

Articulation points and bridges are exact only up to
``MAX_GRAPH_FOR_EXACT_CONNECTIVITY``; beyond that limit they are reported as
undefined instead of approximated.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .feature_spec import MAX_GRAPH_FOR_EXACT_CONNECTIVITY

COMPONENT_FEATURES = (
    "n_components",
    "largest_component_size",
    "largest_component_fraction",
    "isolated_nodes",
    "isolated_fraction",
    "component_size_mean",
    "component_size_max",
)

CONNECTIVITY_FEATURES = (
    "is_connected",
    "articulation_points",
    "bridges",
    "articulation_point_fraction",
    "bridge_fraction",
)


def connected_components(graph: nx.Graph) -> list[set[int]]:
    return [set(component) for component in nx.connected_components(graph)]


def component_features(
    graph: nx.Graph, *, n_nodes: int, components: list[set[int]]
) -> dict[str, float | None]:
    sizes = sorted((len(component) for component in components), reverse=True)
    isolated = int(sum(1 for _, degree in graph.degree() if degree == 0))
    return {
        "n_components": float(len(components)),
        "largest_component_size": float(sizes[0]) if sizes else None,
        "largest_component_fraction": (
            float(sizes[0] / n_nodes) if sizes and n_nodes else None
        ),
        "isolated_nodes": float(isolated),
        "isolated_fraction": float(isolated / n_nodes) if n_nodes else None,
        "component_size_mean": float(np.mean(sizes)) if sizes else None,
        "component_size_max": float(sizes[0]) if sizes else None,
    }


def connectivity_features(
    graph: nx.Graph,
    *,
    n_nodes: int,
    components: list[set[int]],
    unique_pairs: int,
) -> dict[str, float | None]:
    features: dict[str, float | None] = {
        "is_connected": float(len(components) == 1) if n_nodes else None,
        "articulation_points": None,
        "bridges": None,
        "articulation_point_fraction": None,
        "bridge_fraction": None,
    }
    if n_nodes and n_nodes <= MAX_GRAPH_FOR_EXACT_CONNECTIVITY:
        articulation_points = list(nx.articulation_points(graph))
        bridges = list(nx.bridges(graph))
        features["articulation_points"] = float(len(articulation_points))
        features["bridges"] = float(len(bridges))
        features["articulation_point_fraction"] = float(len(articulation_points) / n_nodes)
        features["bridge_fraction"] = (
            float(len(bridges) / unique_pairs) if unique_pairs else None
        )
    return features


__all__ = [
    "COMPONENT_FEATURES",
    "CONNECTIVITY_FEATURES",
    "component_features",
    "connected_components",
    "connectivity_features",
]
