"""Centrality features.

Expensive centralities are computed only on the largest connected component,
which is recorded in the feature specification.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

CENTRALITY_FEATURES = (
    "degree_centrality_mean",
    "degree_centrality_max",
    "degree_centrality_std",
    "betweenness_mean",
    "betweenness_max",
    "betweenness_std",
    "closeness_mean",
    "closeness_max",
    "closeness_std",
    "eigenvector_mean",
    "eigenvector_max",
    "eigenvector_std",
)


def _describe(values: np.ndarray) -> tuple[float, float, float]:
    return float(values.mean()), float(values.max()), float(values.std())


def centrality_features(
    graph: nx.Graph, *, n_nodes: int, largest: set[int]
) -> dict[str, float | None]:
    features: dict[str, float | None] = {
        "degree_centrality_mean": None,
        "degree_centrality_max": None,
        "degree_centrality_std": None,
        "betweenness_mean": None,
        "betweenness_max": None,
        "betweenness_std": None,
        "closeness_mean": None,
        "closeness_max": None,
        "closeness_std": None,
        "eigenvector_mean": None,
        "eigenvector_max": None,
        "eigenvector_std": None,
    }

    if n_nodes >= 2:
        degree = np.asarray(list(nx.degree_centrality(graph).values()), dtype=float)
        (
            features["degree_centrality_mean"],
            features["degree_centrality_max"],
            features["degree_centrality_std"],
        ) = _describe(degree)

    if n_nodes >= 3:
        betweenness = np.asarray(
            list(nx.betweenness_centrality(graph, normalized=True).values()), dtype=float
        )
        (
            features["betweenness_mean"],
            features["betweenness_max"],
            features["betweenness_std"],
        ) = _describe(betweenness)

    largest_subgraph = graph.subgraph(largest).copy() if largest else nx.Graph()
    if len(largest) >= 2:
        closeness = np.asarray(
            list(nx.closeness_centrality(largest_subgraph).values()), dtype=float
        )
        (
            features["closeness_mean"],
            features["closeness_max"],
            features["closeness_std"],
        ) = _describe(closeness)

    if len(largest) >= 3:
        try:
            centrality = nx.eigenvector_centrality(largest_subgraph, max_iter=1000, tol=1e-06)
            eigenvector = np.asarray([centrality[node] for node in sorted(largest)], dtype=float)
        except (nx.NetworkXException, ArithmeticError):
            eigenvector = None
        if eigenvector is not None:
            (
                features["eigenvector_mean"],
                features["eigenvector_max"],
                features["eigenvector_std"],
            ) = _describe(eigenvector)
    return features


__all__ = ["CENTRALITY_FEATURES", "centrality_features"]
