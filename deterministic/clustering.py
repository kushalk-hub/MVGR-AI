"""Clustering features: local clustering, transitivity and triangles."""

from __future__ import annotations

import networkx as nx

CLUSTERING_FEATURES = ("avg_clustering", "transitivity", "triangle_count")


def clustering_features(graph: nx.Graph, *, n_nodes: int) -> dict[str, float | None]:
    if not n_nodes:
        return {
            "avg_clustering": None,
            "transitivity": None,
            "triangle_count": 0.0,
        }
    return {
        "avg_clustering": float(nx.average_clustering(graph)),
        "transitivity": float(nx.transitivity(graph)),
        "triangle_count": float(sum(nx.triangles(graph).values()) // 3),
    }


__all__ = ["CLUSTERING_FEATURES", "clustering_features"]
