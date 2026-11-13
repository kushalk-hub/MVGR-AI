"""Community-structure features (seeded Louvain)."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .feature_spec import COMMUNITY_SEED, MAX_GRAPH_FOR_COMMUNITIES

COMMUNITY_FEATURES = (
    "n_communities",
    "modularity",
    "community_size_mean",
    "community_size_max",
    "community_size_std",
    "community_count_ratio",
)


def community_features(graph: nx.Graph, *, n_nodes: int) -> dict[str, float | None]:
    if not n_nodes or n_nodes > MAX_GRAPH_FOR_COMMUNITIES:
        return {name: None for name in COMMUNITY_FEATURES}

    communities = nx.community.louvain_communities(
        graph, weight="weight", seed=COMMUNITY_SEED, resolution=1.0
    )
    sizes = np.asarray([len(community) for community in communities], dtype=float)
    return {
        "n_communities": float(len(communities)),
        "modularity": float(nx.community.modularity(graph, communities, weight="weight")),
        "community_size_mean": float(sizes.mean()),
        "community_size_max": float(sizes.max()),
        "community_size_std": float(sizes.std()),
        "community_count_ratio": float(len(communities) / n_nodes),
    }


__all__ = ["COMMUNITY_FEATURES", "community_features"]
