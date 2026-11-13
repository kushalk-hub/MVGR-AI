"""Shortest-path features with an explicit computational guard."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .feature_spec import MAX_GRAPH_FOR_SHORTEST_PATHS

PATH_FEATURES = (
    "avg_shortest_path_all",
    "avg_shortest_path_largest",
    "diameter_largest",
    "path_length_median",
    "path_length_std",
    "n_connected_pairs",
)


def component_shortest_paths(
    graph: nx.Graph, components: list[set[int]]
) -> tuple[list[int], dict[str, float | None]]:
    distances: list[int] = []
    largest_distances: list[int] = []
    largest = max(components, key=len) if components else set()

    for component in components:
        if len(component) < 2:
            continue
        subgraph = graph.subgraph(component)
        component_distances: list[int] = []
        for source in sorted(component):
            lengths = nx.single_source_shortest_path_length(subgraph, source)
            for target, distance in sorted(lengths.items()):
                if target > source:
                    component_distances.append(int(distance))
        distances.extend(component_distances)
        if component == largest:
            largest_distances = component_distances

    def summarize(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"mean": None, "median": None, "std": None, "max": None, "pairs": 0}
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std()),
            "max": float(array.max()),
            "pairs": int(array.size),
        }

    overall = summarize(distances)
    inside_largest = summarize(largest_distances)
    return distances, {
        "avg_shortest_path_all": overall["mean"],
        "avg_shortest_path_largest": inside_largest["mean"],
        "diameter_largest": inside_largest["max"],
        "path_length_median": overall["median"],
        "path_length_std": overall["std"],
        "n_connected_pairs": float(overall["pairs"]),
    }


def shortest_path_features(
    graph: nx.Graph, *, n_nodes: int, components: list[set[int]]
) -> dict[str, float | None]:
    if n_nodes and n_nodes <= MAX_GRAPH_FOR_SHORTEST_PATHS:
        _, features = component_shortest_paths(graph, components)
        return features
    return {name: None for name in PATH_FEATURES}


__all__ = ["PATH_FEATURES", "component_shortest_paths", "shortest_path_features"]
