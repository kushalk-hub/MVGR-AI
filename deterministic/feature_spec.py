"""Deterministic feature specification (features.v1).

Every feature is defined here with its formula and the condition under which it
is undefined. The registry is the authoritative list; the extractor may not
emit a feature that is absent from it.

Undefined values are emitted as ``None`` in JSON and ``NaN`` in Parquet. They
are never silently imputed or replaced with zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from shared.contracts.versions import FEATURE_SET_VERSION

# Computational guards. They are part of features.v1: changing a guard changes
# which values are defined and therefore requires a new feature-set version.
MAX_GRAPH_FOR_EXACT_CONNECTIVITY = 2_000
MAX_GRAPH_FOR_SHORTEST_PATHS = 5_000
MAX_GRAPH_FOR_COMMUNITIES = 20_000

COMMUNITY_SEED = 42


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    description: str
    formula: str
    undefined_when: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


SPECS: tuple[FeatureSpec, ...] = (
    # --- graph size -------------------------------------------------------
    FeatureSpec("n_nodes", "size", "Number of nodes", "|V|", "never"),
    FeatureSpec("n_edges", "size", "Number of directed edge records", "|E|", "never"),
    FeatureSpec("n_unique_edges", "size", "Number of distinct unordered node pairs with an edge", "|unique pairs|", "never"),
    FeatureSpec("density", "size", "Edge density of the undirected simple projection", "2|E| / (|V|(|V|-1))", "|V| < 2"),
    FeatureSpec("is_directed", "size", "Whether any edge is directed", "any(edge.directed)", "never"),
    FeatureSpec("self_loop_count", "size", "Edges whose endpoints are identical", "count(u == v)", "never"),
    FeatureSpec("parallel_edge_count", "size", "Edge records minus distinct unordered pairs", "|E| - |unique pairs|", "never"),
    # --- degree distribution ---------------------------------------------
    FeatureSpec("degree_mean", "degree", "Mean degree of the undirected simple projection", "sum(deg) / |V|", "|V| = 0"),
    FeatureSpec("degree_std", "degree", "Population standard deviation of degree", "sqrt(mean((deg - mean)^2))", "|V| = 0"),
    FeatureSpec("degree_min", "degree", "Minimum degree", "min(deg)", "|V| = 0"),
    FeatureSpec("degree_max", "degree", "Maximum degree", "max(deg)", "|V| = 0"),
    FeatureSpec("degree_median", "degree", "Median degree", "median(deg)", "|V| = 0"),
    FeatureSpec("degree_q25", "degree", "25th percentile of degree", "quantile(deg, 0.25)", "|V| = 0"),
    FeatureSpec("degree_q75", "degree", "75th percentile of degree", "quantile(deg, 0.75)", "|V| = 0"),
    FeatureSpec("degree_cv", "degree", "Coefficient of variation of degree", "std / mean", "mean = 0"),
    FeatureSpec("degree_gini", "degree", "Gini coefficient of the degree sequence", "sum((2i - n - 1) x_i) / (n sum(x))", "|V| = 0 or sum(deg) = 0"),
    FeatureSpec("degree_entropy", "degree", "Shannon entropy of degree distribution, normalized by log |V|", "H(p) / log|V|", "|V| < 2"),
    FeatureSpec("degree_max_ratio", "degree", "Hub concentration", "max(deg) / sum(deg)", "sum(deg) = 0"),
    # --- clustering -------------------------------------------------------
    FeatureSpec("avg_clustering", "clustering", "Mean local clustering coefficient", "mean(C_i)", "|V| = 0"),
    FeatureSpec("transitivity", "clustering", "Global clustering coefficient", "3 * triangles / connected triples", "no connected triples"),
    FeatureSpec("triangle_count", "clustering", "Number of triangles", "|{triangle}|", "never"),
    # --- centrality -------------------------------------------------------
    FeatureSpec("degree_centrality_mean", "centrality", "Mean degree centrality", "mean(deg(v)/(|V|-1))", "|V| < 2"),
    FeatureSpec("degree_centrality_max", "centrality", "Maximum degree centrality", "max(deg(v)/(|V|-1))", "|V| < 2"),
    FeatureSpec("degree_centrality_std", "centrality", "Standard deviation of degree centrality", "std(deg(v)/(|V|-1))", "|V| < 2"),
    FeatureSpec("betweenness_mean", "centrality", "Mean normalized betweenness centrality", "mean(b(v))", "|V| < 3"),
    FeatureSpec("betweenness_max", "centrality", "Maximum normalized betweenness centrality", "max(b(v))", "|V| < 3"),
    FeatureSpec("betweenness_std", "centrality", "Standard deviation of betweenness centrality", "std(b(v))", "|V| < 3"),
    FeatureSpec("closeness_mean", "centrality", "Mean closeness centrality on the largest component", "mean(c(v))", "largest component < 2"),
    FeatureSpec("closeness_max", "centrality", "Maximum closeness centrality on the largest component", "max(c(v))", "largest component < 2"),
    FeatureSpec("closeness_std", "centrality", "Standard deviation of closeness centrality", "std(c(v))", "largest component < 2"),
    FeatureSpec("eigenvector_mean", "centrality", "Mean eigenvector centrality on the largest component", "mean(e(v))", "largest component < 3 or no convergence"),
    FeatureSpec("eigenvector_max", "centrality", "Maximum eigenvector centrality on the largest component", "max(e(v))", "largest component < 3 or no convergence"),
    FeatureSpec("eigenvector_std", "centrality", "Standard deviation of eigenvector centrality", "std(e(v))", "largest component < 3 or no convergence"),
    # --- components -------------------------------------------------------
    FeatureSpec("n_components", "components", "Number of connected components (weak for directed input)", "count(cc)", "never"),
    FeatureSpec("largest_component_size", "components", "Nodes in the largest connected component", "max(|cc|)", "|V| = 0"),
    FeatureSpec("largest_component_fraction", "components", "Largest component share of nodes", "max(|cc|) / |V|", "|V| = 0"),
    FeatureSpec("isolated_nodes", "components", "Nodes with degree zero", "|{v: deg(v) = 0}|", "never"),
    FeatureSpec("isolated_fraction", "components", "Isolated node share", "isolated / |V|", "|V| = 0"),
    FeatureSpec("component_size_mean", "components", "Mean component size", "mean(|cc|)", "|V| = 0"),
    FeatureSpec("component_size_max", "components", "Maximum component size", "max(|cc|)", "|V| = 0"),
    # --- connectivity -----------------------------------------------------
    FeatureSpec("is_connected", "connectivity", "Whether the undirected projection is connected", "n_components == 1", "|V| = 0"),
    FeatureSpec("articulation_points", "connectivity", "Count of articulation points", "|AP|", f"|V| > {MAX_GRAPH_FOR_EXACT_CONNECTIVITY}"),
    FeatureSpec("bridges", "connectivity", "Count of bridges", "|bridges|", f"|V| > {MAX_GRAPH_FOR_EXACT_CONNECTIVITY}"),
    FeatureSpec("articulation_point_fraction", "connectivity", "Articulation points per node", "|AP| / |V|", "AP undefined or |V| = 0"),
    FeatureSpec("bridge_fraction", "connectivity", "Bridges per undirected edge", "|bridges| / |E_undirected|", "bridges undefined or |E| = 0"),
    # --- shortest paths ---------------------------------------------------
    FeatureSpec("avg_shortest_path_all", "paths", "Mean shortest path over all connected node pairs", "sum(d) / n_connected_pairs", "no connected pair or graph too large"),
    FeatureSpec("avg_shortest_path_largest", "paths", "Mean shortest path inside the largest component", "mean(d) in largest cc", "largest component < 2 or graph too large"),
    FeatureSpec("diameter_largest", "paths", "Diameter of the largest component", "max(d) in largest cc", "largest component < 2 or graph too large"),
    FeatureSpec("path_length_median", "paths", "Median connected-pair distance", "median(d)", "no connected pair or graph too large"),
    FeatureSpec("path_length_std", "paths", "Standard deviation of connected-pair distance", "std(d)", "no connected pair or graph too large"),
    FeatureSpec("n_connected_pairs", "paths", "Number of connected node pairs", "sum(c(c-1)/2)", "graph too large"),
    # --- community structure ---------------------------------------------
    FeatureSpec("n_communities", "community", "Louvain community count", "|communities|", f"|V| = 0 or |V| > {MAX_GRAPH_FOR_COMMUNITIES}"),
    FeatureSpec("modularity", "community", "Modularity of the Louvain partition", "Q", f"|V| = 0 or |V| > {MAX_GRAPH_FOR_COMMUNITIES}"),
    FeatureSpec("community_size_mean", "community", "Mean community size", "mean(|c|)", "no communities"),
    FeatureSpec("community_size_max", "community", "Maximum community size", "max(|c|)", "no communities"),
    FeatureSpec("community_size_std", "community", "Standard deviation of community size", "std(|c|)", "no communities"),
    FeatureSpec("community_count_ratio", "community", "Communities per node", "|communities| / |V|", "|V| = 0"),
    # --- weights ----------------------------------------------------------
    FeatureSpec("edge_weight_mean", "weights", "Mean edge weight", "mean(w)", "|E| = 0"),
    FeatureSpec("edge_weight_std", "weights", "Standard deviation of edge weight", "std(w)", "|E| = 0"),
    FeatureSpec("edge_weight_max", "weights", "Maximum edge weight", "max(w)", "|E| = 0"),
    FeatureSpec("edge_weight_min", "weights", "Minimum edge weight", "min(w)", "|E| = 0"),
    # --- view-specific ----------------------------------------------------
    FeatureSpec("node_type_token", "view", "Token-type node count", "count(node_type == token)", "never"),
    FeatureSpec("node_type_punctuation", "view", "Punctuation-type node count", "count(node_type == punctuation)", "never"),
    FeatureSpec("node_type_sentence_id", "view", "Sentence-id-type node count", "count(node_type == sentence_id)", "never"),
    FeatureSpec("edge_type_cooccurrence", "view", "Co-occurrence edge count", "count(edge_type == cooccurrence)", "never"),
    FeatureSpec("edge_type_dependency", "view", "Dependency edge count", "count(edge_type == dependency)", "never"),
    FeatureSpec("stopword_fraction", "view", "Share of nodes flagged as stopwords", "stopword nodes / |V|", "|V| = 0"),
    FeatureSpec("token_char_len_mean", "view", "Mean token length in characters", "mean(len(token))", "no token nodes"),
    FeatureSpec("token_char_len_std", "view", "Standard deviation of token length", "std(len(token))", "no token nodes"),
    FeatureSpec("unique_token_ratio", "view", "Distinct token strings per node", "|unique tokens| / |V|", "|V| = 0"),
    FeatureSpec("token_repetition_rate", "view", "Repeated token occurrences per node", "1 - unique_token_ratio", "|V| = 0"),
    FeatureSpec("dependency_label_count", "view", "Distinct dependency labels", "|labels|", "no dependency edges"),
    FeatureSpec("dependency_entropy", "view", "Entropy of dependency-label distribution", "H(p)", "no dependency edges"),
    FeatureSpec("dependency_distance_mean", "view", "Mean absolute head-child token distance", "mean(|pos_head - pos_child|)", "no dependency distance"),
    FeatureSpec("dependency_distance_max", "view", "Maximum absolute head-child token distance", "max(|pos_head - pos_child|)", "no dependency distance"),
    FeatureSpec("tree_depth", "view", "Longest dependency path from a root", "longest path", "syntactic view with no edges or a cycle"),
    FeatureSpec("root_count", "view", "Nodes with no incoming dependency edge", "count(indegree == 0)", "no nodes"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in SPECS)
SPEC_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in SPECS}


def feature_registry() -> dict[str, object]:
    return {
        "feature_set_version": FEATURE_SET_VERSION,
        "undefined_policy": "null",
        "limits": {
            "max_graph_for_exact_connectivity": MAX_GRAPH_FOR_EXACT_CONNECTIVITY,
            "max_graph_for_shortest_paths": MAX_GRAPH_FOR_SHORTEST_PATHS,
            "max_graph_for_communities": MAX_GRAPH_FOR_COMMUNITIES,
            "community_seed": COMMUNITY_SEED,
        },
        "features": [spec.as_dict() for spec in SPECS],
    }


def assert_registry_complete(values: dict[str, object]) -> None:
    missing = [name for name in FEATURE_NAMES if name not in values]
    extra = [name for name in values if name not in SPEC_BY_NAME]
    if missing or extra:
        raise AssertionError(f"feature registry mismatch: missing={missing} extra={extra}")
