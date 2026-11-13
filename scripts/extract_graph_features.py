"""Graph feature extraction for the MVGR-AI pilot build (Phases B, C).

Reads the *live pilot schema* directly from Neo4j Aura:

    (:Sample)-[:HAS_GRAPH]->(:TextGraph)-[:HAS_TOKEN]->(:TokenOccurrence)
    (:TokenOccurrence)-[:CO_OCCURS]->(:TokenOccurrence)     # lexical view
    (:TokenOccurrence)-[:DEPENDS_ON]->(:TokenOccurrence)    # syntactic view

and emits three tables plus a schema description:

* ``data/processed/graph_features.csv``            one row per graph view
* ``data/processed/sample_multiview_features.csv`` one row per sample
* ``data/processed/node_features.csv``             one row per token node
* ``data/processed/feature_schema.json``

Nothing is written back to Neo4j. Labels, relationship types, property keys and
categorical vocabularies are discovered from the database, never assumed.

Performance: samples are processed in batches; each batch issues four bounded
queries (metadata, graphs, tokens, edges) so there is no N+1 access and the full
102,822-node set is never resident in memory at once. Per-batch results are
cached to JSONL and an interrupted run resumes from the cache.

Usage
-----
    python scripts/extract_graph_features.py
    python scripts/extract_graph_features.py --limit 20 --no-resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

# Allow `python scripts/extract_graph_features.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.paths import load_config, resolve  # noqa: E402
from shared.contracts.versions import FEATURE_SET_VERSION, SPLIT_POLICY_VERSION  # noqa: E402

# Fixed degree-histogram bin edges (degree 0 is its own bin; the last bin is open).
DEGREE_BIN_EDGES: tuple[float, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 20, math.inf)
N_DEGREE_BINS = len(DEGREE_BIN_EDGES) - 1

SAMPLE_PROPERTIES = (
    "sample_id", "label", "source_dataset", "source_task", "text_type",
    "graph_category", "paraphrase_status", "domain", "ai_model",
    "padben_split", "split", "group_id", "source_row_id",
)
GRAPH_PROPERTIES = (
    "graph_id", "view", "build_id", "graph_spec_version", "spacy_version",
    "model_name", "token_count", "edge_count", "parser_status",
)


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE pairs into os.environ without overriding real env vars."""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class Settings:
    def __init__(self, config: dict[str, Any]) -> None:
        neo = config["neo4j"]
        load_dotenv_file(resolve(neo.get("dotenv_file", ".env")))
        self.uri = os.environ.get(neo["uri_env"], "")
        self.user = os.environ.get(neo["user_env"], "")
        self.password = os.environ.get(neo["password_env"], "")
        self.database = os.environ.get(neo["database_env"], neo["database_default"])
        self.timeout = float(neo.get("connection_timeout_seconds", 30))
        self.max_retries = int(neo.get("max_retries", 3))
        self.backoff = float(neo.get("retry_backoff_seconds", 2))

    def missing(self) -> list[str]:
        return [name for name, value in (("uri", self.uri), ("user", self.user),
                                         ("password", self.password)) if not value]


def connect(settings: Settings):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
        connection_timeout=settings.timeout,
        # Recycle sockets and validate them so a dropped Aura connection is
        # detected on acquisition instead of surfacing mid-result.
        max_connection_lifetime=300,
        liveness_check_timeout=30,
        keep_alive=True,
    )
    driver.verify_connectivity()
    return driver


def retryable_exceptions() -> tuple[type[BaseException], ...]:
    """Neo4j transient/driver errors that are safe to retry after reconnecting."""
    try:
        from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

        return (ServiceUnavailable, SessionExpired, TransientError, OSError)
    except Exception:  # driver not installed; the connect call will fail first
        return (OSError,)


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _safe_mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else None


def _safe_std(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    return float(np.std(values)) if values else None


def _finite(value: Any) -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def sanitize(name: Any) -> str:
    """Make a categorical value safe for use inside a column name."""
    text = str(name) if name is not None else "none"
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned.lower() or "none"


def entropy(counts: Iterable[float]) -> float | None:
    values = np.asarray([float(c) for c in counts if c], dtype=float)
    if values.size == 0:
        return None
    p = values / values.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def cosine_similarity(a: list[float], b: list[float]) -> float | None:
    va, vb = np.asarray(a, float), np.asarray(b, float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def jensen_shannon_similarity(p: list[float], q: list[float]) -> float | None:
    """1 - Jensen-Shannon divergence (natural log), in [1 - ln2, 1]."""
    vp, vq = np.asarray(p, float), np.asarray(q, float)
    if vp.sum() <= 0 or vq.sum() <= 0:
        return None
    vp, vq = vp / vp.sum(), vq / vq.sum()
    m = 0.5 * (vp + vq)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = (x > 0) & (y > 0)
        return float(np.sum(x[mask] * np.log(x[mask] / y[mask])))

    divergence = 0.5 * _kl(vp, m) + 0.5 * _kl(vq, m)
    return float(1.0 - divergence)


def degree_histogram(degrees: np.ndarray) -> np.ndarray:
    if degrees.size == 0:
        return np.zeros(N_DEGREE_BINS, dtype=float)
    counts, _ = np.histogram(degrees, bins=list(DEGREE_BIN_EDGES))
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def _path_stats(subgraph: nx.Graph) -> tuple[float, int, int] | None:
    """Mean shortest-path length, diameter and connected pairs for a component.

    Uses SciPy's C shortest-path when available (exact, unweighted) and falls
    back to a single NetworkX all-pairs pass otherwise. Returns ``None`` for a
    component with fewer than two nodes.
    """
    nodes = list(subgraph.nodes())
    n = len(nodes)
    if n < 2:
        return None
    index = {node: i for i, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    for u, v in subgraph.edges():
        i, j = index[u], index[v]
        rows.append(i)
        cols.append(j)
        rows.append(j)
        cols.append(i)
    pairs = n * (n - 1) // 2
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        distances = shortest_path(matrix, method="D", unweighted=True, directed=False)
        total = float(distances.sum())
        return total / (n * (n - 1)), int(distances.max()), pairs
    except ImportError:
        distance_total = 0
        max_distance = 0
        for source, lengths in nx.all_pairs_shortest_path_length(subgraph):
            for target, dist in lengths.items():
                if target != source:
                    distance_total += dist
                    if dist > max_distance:
                        max_distance = dist
        return distance_total / (n * (n - 1)), max_distance, pairs


# --------------------------------------------------------------------------- #
# feature vocabulary (built once, from the live database)
# --------------------------------------------------------------------------- #
def discover_vocabularies(conn, config: dict[str, Any]) -> dict[str, list[str]]:
    top_k = int(config["extraction"].get("top_dependency_relations", 15))
    token_label = config["graph"]["token_label"]
    pos_values = [
        value for value in conn.value(
            f"MATCH (t:{token_label}) RETURN DISTINCT t.pos AS pos ORDER BY pos",
            label="vocab:pos",
        ) if value
    ]
    dep_rows = conn.data(
        "MATCH ()-[r:DEPENDS_ON]->() RETURN r.dependency_label AS label, count(*) AS c "
        "ORDER BY c DESC, label",
        label="vocab:dependency",
    )
    dep_top = [row["label"] for row in dep_rows if row["label"]][:top_k]
    return {"pos": pos_values, "dependency_top": dep_top}


def discover_schema(conn, config: dict[str, Any]) -> dict[str, Any]:
    g = config["graph"]
    labels = conn.value("CALL db.labels() YIELD label RETURN label ORDER BY label",
                        label="schema:labels")
    rel_types = conn.value(
        "CALL db.relationshipTypes() YIELD relationshipType AS t RETURN t ORDER BY t",
        label="schema:relationships",
    )
    keys: dict[str, list[str]] = {}
    for label in (g["sample_label"], g["graph_label"], g["token_label"]):
        keys[label] = conn.value(
            f"MATCH (n:{label}) UNWIND keys(n) AS k RETURN DISTINCT k ORDER BY k",
            label=f"schema:keys:{label}",
        )
    return {"node_labels": labels, "relationship_types": rel_types, "property_keys": keys}


# --------------------------------------------------------------------------- #
# core per-graph feature computation (pure; unit-testable)
# --------------------------------------------------------------------------- #
def graph_view_features(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    view: str,
    pos_vocab: list[str],
    dep_vocab: list[str],
    guards: dict[str, int],
) -> dict[str, Any]:
    """Compute every per-view feature for one graph.

    ``nodes`` items carry ``node_id, position, surface, normalized, lemma, pos,
    tag, ent_type, is_punct, is_stop``. ``edges`` items carry ``source, target,
    rel_type, dependency_label, distance``. The pilot build stores ``distance``
    on ``CO_OCCURS`` edges only; for ``DEPENDS_ON`` edges the head-child distance
    is recovered from the endpoint token positions.
    """
    node_ids = [n["node_id"] for n in nodes]
    n_nodes = len(node_ids)

    cooc = [e for e in edges if e["rel_type"] == "CO_OCCURS"]
    dep = [e for e in edges if e["rel_type"] == "DEPENDS_ON"]

    # ---- structural (undirected simple projection of all edges) -----------
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    self_loops = 0
    for edge in edges:
        if edge["source"] == edge["target"]:
            self_loops += 1
            continue
        graph.add_edge(edge["source"], edge["target"])

    n_unique_edges = graph.number_of_edges()
    parallel_edges = len(edges) - self_loops - n_unique_edges
    degrees = np.asarray([graph.degree(n) for n in node_ids], dtype=float)
    components = list(nx.connected_components(graph)) if n_nodes else []
    largest = max(components, key=len) if components else set()
    isolated = int(np.sum(degrees == 0)) if n_nodes else 0

    if n_nodes > 1:
        density = float(2 * n_unique_edges / (n_nodes * (n_nodes - 1)))
    else:
        density = None

    avg_clustering = float(nx.average_clustering(graph)) if n_nodes else None
    transitivity = nx.transitivity(graph) if n_nodes else None
    if transitivity is not None and not math.isfinite(float(transitivity)):
        transitivity = None
    triangles = int(sum(nx.triangles(graph).values()) // 3) if n_nodes else 0

    # ---- shortest paths, guarded by component size ------------------------
    # All-pairs distances on the largest component. NetworkX BFS is pure Python
    # and takes ~18 s on a 1,000-node graph, so SciPy's C implementation is used
    # when available (identical unweighted semantics), with a NetworkX fallback.
    avg_shortest_path = diameter = n_connected_pairs = None
    if 2 <= len(largest) <= int(guards["max_nodes_for_shortest_path"]):
        stats = _path_stats(graph.subgraph(largest))
        if stats is not None:
            avg_shortest_path, diameter, n_connected_pairs = stats

    hist = degree_histogram(degrees)

    # ---- lexical (token properties; identical node set across views) ------
    surfaces = [str(n.get("surface") or "") for n in nodes]
    normalized = [str(n.get("normalized") or n.get("surface") or "") for n in nodes]
    lemmas = [str(n.get("lemma") or "") for n in nodes]
    unique_tokens = len({t for t in normalized if t}) if n_nodes else 0
    pos_counts: dict[str, int] = {}
    for node in nodes:
        tag = str(node.get("pos") or "")
        pos_counts[tag] = pos_counts.get(tag, 0) + 1
    entity_tokens = sum(1 for node in nodes if str(node.get("ent_type") or ""))

    lexical: dict[str, Any] = {
        "token_count": n_nodes,
        "unique_token_count": unique_tokens,
        "vocabulary_diversity": (unique_tokens / n_nodes) if n_nodes else None,
        "repeated_token_ratio": (1.0 - unique_tokens / n_nodes) if n_nodes else None,
        "avg_token_length": _safe_mean(len(s) for s in surfaces),
        "std_token_length": _safe_std(len(s) for s in surfaces),
        "avg_lemma_length": _safe_mean(len(l) for l in lemmas),
        "stopword_ratio": _safe_mean(1.0 if n.get("is_stop") else 0.0 for n in nodes),
        "punctuation_ratio": _safe_mean(1.0 if n.get("is_punct") else 0.0 for n in nodes),
        "n_pos_tags": len(pos_counts),
        "pos_entropy": entropy(pos_counts.values()),
        "n_entity_types": len({str(n.get("ent_type")) for n in nodes if str(n.get("ent_type") or "")}),
        "entity_token_ratio": (entity_tokens / n_nodes) if n_nodes else None,
    }
    for tag in pos_vocab:
        lexical[f"pos_{sanitize(tag)}_ratio"] = (
            pos_counts.get(tag, 0) / n_nodes if n_nodes else None
        )

    # ---- syntactic (dependency edges; only defined for the syntactic view) -
    syntactic: dict[str, Any] = {}
    syntactic_names = _syntactic_feature_names(dep_vocab)
    if view != "syntactic_dependency" and not dep:
        for name in syntactic_names:
            syntactic[name] = None
    else:
        labels = [str(e.get("dependency_label") or "") for e in dep]
        label_counts: dict[str, int] = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        position_of = {n["node_id"]: n.get("position") for n in nodes}

        def _dep_distance(edge: dict[str, Any]) -> float | None:
            """Absolute head-child token distance for one dependency edge.

            ``CO_OCCURS`` edges carry ``distance`` directly. The pilot build does
            not set it on ``DEPENDS_ON`` edges, so the distance is recovered from
            the positions of the head and child tokens. Only returns None when
            the edge carries neither a stored distance nor resolvable endpoints.
            """
            raw = edge.get("distance")
            if raw is not None:
                return float(raw)
            head = position_of.get(edge["source"])
            child = position_of.get(edge["target"])
            if head is None or child is None:
                return None
            return float(abs(int(child) - int(head)))

        distances = [d for d in (_dep_distance(e) for e in dep) if d is not None]

        directed = nx.DiGraph()
        directed.add_nodes_from(node_ids)
        for edge in dep:
            if edge["source"] != edge["target"]:
                directed.add_edge(edge["source"], edge["target"])
        in_deg = dict(directed.in_degree())
        out_deg = dict(directed.out_degree())
        dep_active = [n for n in node_ids if in_deg[n] + out_deg[n] > 0]
        roots = [n for n in dep_active if in_deg[n] == 0]

        depth = _dependency_depth(directed)
        syntactic = {
            "dep_edge_count": len(dep),
            "dep_relation_diversity": len(label_counts) if dep else None,
            "dep_relation_entropy": entropy(label_counts.values()) if dep else None,
            "dep_avg_out_degree": (len(dep) / n_nodes) if n_nodes else None,
            "dep_max_in_degree": max(in_deg.values()) if dep_active else None,
            "dep_max_out_degree": max(out_deg.values()) if dep_active else None,
            "root_count": len(roots) if dep else None,
            "root_ratio": (len(roots) / len(dep_active)) if dep_active else None,
            "dependency_distance_mean": _safe_mean(distances),
            "dependency_distance_max": max(distances) if distances else None,
            "tree_depth": depth,
        }
        for label in dep_vocab:
            syntactic[f"dep_{sanitize(label)}_count"] = label_counts.get(label, 0)
        syntactic["dep_other_count"] = sum(
            count for label, count in label_counts.items() if label not in set(dep_vocab)
        )

    features: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "n_unique_edges": n_unique_edges,
        "parallel_edge_count": parallel_edges,
        "self_loop_count": self_loops,
        "is_directed": bool(dep),
        "degree_mean": _safe_mean(degrees),
        "degree_std": _safe_std(degrees),
        "degree_min": int(degrees.min()) if n_nodes else None,
        "degree_max": int(degrees.max()) if n_nodes else None,
        "degree_median": float(np.median(degrees)) if n_nodes else None,
        "degree_p90": float(np.percentile(degrees, 90)) if n_nodes else None,
        "degree_cv": None,
        "density": density,
        "avg_clustering": avg_clustering,
        "transitivity": float(transitivity) if transitivity is not None else None,
        "triangle_count": triangles,
        "n_components": len(components) if n_nodes else None,
        "largest_component_size": len(largest) if n_nodes else None,
        "largest_component_ratio": (len(largest) / n_nodes) if n_nodes else None,
        "isolated_nodes": isolated if n_nodes else None,
        "isolated_ratio": (isolated / n_nodes) if n_nodes else None,
        "component_size_mean": _safe_mean(len(c) for c in components),
        "component_size_max": max((len(c) for c in components), default=None),
        "avg_shortest_path": avg_shortest_path,
        "diameter": diameter,
        "n_connected_pairs": n_connected_pairs,
        "degree_hist_entropy": entropy(hist),
    }
    if features["degree_mean"]:
        features["degree_cv"] = float(features["degree_std"] / features["degree_mean"])
    for index, value in enumerate(hist):
        features[f"deg_bin_{index}"] = float(value)

    features.update(lexical)
    features.update(syntactic)
    return {key: _finite(value) for key, value in features.items()}


def _dependency_depth(directed: nx.DiGraph) -> int | None:
    """Longest dependency path from any root; None when acyclic order is absent."""
    if directed.number_of_edges() == 0:
        return None
    roots = [n for n in directed.nodes if directed.in_degree(n) == 0]
    if not roots:
        return None
    try:
        order = list(nx.topological_sort(directed))
    except nx.NetworkXUnfeasible:
        return None
    depth = {node: 0 for node in roots}
    for node in order:
        depth.setdefault(node, 0)
        for child in directed.successors(node):
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
    return int(max(depth.values()))


def _syntactic_feature_names(dep_vocab: list[str]) -> list[str]:
    base = [
        "dep_edge_count", "dep_relation_diversity", "dep_relation_entropy",
        "dep_avg_out_degree", "dep_max_in_degree", "dep_max_out_degree",
        "root_count", "root_ratio", "dependency_distance_mean",
        "dependency_distance_max", "tree_depth",
    ]
    return base + [f"dep_{sanitize(label)}_count" for label in dep_vocab] + ["dep_other_count"]


def node_feature_rows(
    graph_row: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per token node with graph/sample identifiers and node-level features."""
    node_ids = [n["node_id"] for n in nodes]
    n_nodes = len(node_ids)
    position_of = {n["node_id"]: n.get("position") for n in nodes}

    undirected = nx.Graph()
    undirected.add_nodes_from(node_ids)
    directed = nx.DiGraph()
    directed.add_nodes_from(node_ids)
    for edge in edges:
        if edge["source"] == edge["target"]:
            continue
        undirected.add_edge(edge["source"], edge["target"])
        if edge["rel_type"] == "DEPENDS_ON":
            directed.add_edge(edge["source"], edge["target"])

    freq: dict[str, int] = {}
    for node in nodes:
        key = str(node.get("normalized") or node.get("surface") or "")
        freq[key] = freq.get(key, 0) + 1

    dep_active = {n for n in node_ids if directed.in_degree(n) + directed.out_degree(n) > 0}
    max_position = max((p for p in position_of.values() if p is not None), default=None)

    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node["node_id"]
        key = str(node.get("normalized") or node.get("surface") or "")
        position = node.get("position")
        rows.append({
            "sample_id": graph_row["sample_id"],
            "graph_id": graph_row["graph_id"],
            "view": graph_row["view"],
            "node_id": node_id,
            "position": position,
            "surface_length": len(str(node.get("surface") or "")),
            "normalized_length": len(str(node.get("normalized") or "")),
            "lemma_length": len(str(node.get("lemma") or "")),
            "is_punct": bool(node.get("is_punct")),
            "is_stop": bool(node.get("is_stop")),
            "pos": node.get("pos"),
            "tag": node.get("tag"),
            "ent_type": node.get("ent_type") or "",
            "lexical_degree": int(undirected.degree(node_id)),
            "in_degree": int(directed.in_degree(node_id)),
            "out_degree": int(directed.out_degree(node_id)),
            "is_root": bool(node_id in dep_active and directed.in_degree(node_id) == 0),
            "token_frequency": freq.get(key, 0),
            "is_repeated": freq.get(key, 0) > 1,
            "position_ratio": (
                float(position) / max_position
                if position is not None and max_position not in (None, 0) else None
            ),
            "is_first_token": bool(position == 0),
            "is_last_token": bool(position is not None and position == max_position),
        })
    return rows


# --------------------------------------------------------------------------- #
# multi-view aggregation (pure; unit-testable)
# --------------------------------------------------------------------------- #
STRUCTURAL_SIM_VECTOR = (
    "n_nodes", "n_edges", "degree_mean", "density", "avg_clustering",
    "n_components", "largest_component_ratio",
)


def _vector(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for name in STRUCTURAL_SIM_VECTOR:
        value = row.get(name)
        if name.startswith("n_") and value is not None:
            value = math.log1p(value)
        values.append(float(value) if value is not None else 0.0)
    return values


def _diff(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _ratio(a: Any, b: Any) -> float | None:
    if a in (None, 0) or b is None:
        return None
    return float(b) / float(a)


def multiview_features(lexical: dict[str, Any], syntactic: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the two views of one sample into sample-level features."""
    features: dict[str, Any] = {
        "n_nodes_lexical": lexical.get("n_nodes"),
        "n_nodes_syntactic": syntactic.get("n_nodes"),
        "n_edges_lexical": lexical.get("n_edges"),
        "n_edges_syntactic": syntactic.get("n_edges"),
        "node_count_diff": _diff(lexical.get("n_nodes"), syntactic.get("n_nodes")),
        "edge_count_diff": _diff(lexical.get("n_edges"), syntactic.get("n_edges")),
        "node_count_ratio": _ratio(lexical.get("n_nodes"), syntactic.get("n_nodes")),
        "edge_count_ratio": _ratio(lexical.get("n_edges"), syntactic.get("n_edges")),
        "density_lexical": lexical.get("density"),
        "density_syntactic": syntactic.get("density"),
        "density_diff": _diff(lexical.get("density"), syntactic.get("density")),
        "avg_degree_lexical": lexical.get("degree_mean"),
        "avg_degree_syntactic": syntactic.get("degree_mean"),
        "avg_degree_diff": _diff(lexical.get("degree_mean"), syntactic.get("degree_mean")),
        "avg_clustering_lexical": lexical.get("avg_clustering"),
        "avg_clustering_syntactic": syntactic.get("avg_clustering"),
        "avg_clustering_diff": _diff(lexical.get("avg_clustering"), syntactic.get("avg_clustering")),
        "n_components_lexical": lexical.get("n_components"),
        "n_components_syntactic": syntactic.get("n_components"),
        "n_components_diff": _diff(lexical.get("n_components"), syntactic.get("n_components")),
        "largest_cc_ratio_lexical": lexical.get("largest_component_ratio"),
        "largest_cc_ratio_syntactic": syntactic.get("largest_component_ratio"),
        "largest_cc_ratio_diff": _diff(lexical.get("largest_component_ratio"),
                                       syntactic.get("largest_component_ratio")),
        "avg_shortest_path_lexical": lexical.get("avg_shortest_path"),
        "avg_shortest_path_syntactic": syntactic.get("avg_shortest_path"),
        "avg_shortest_path_diff": _diff(lexical.get("avg_shortest_path"),
                                        syntactic.get("avg_shortest_path")),
        "structural_cosine_similarity": cosine_similarity(_vector(lexical), _vector(syntactic)),
        "degree_dist_cosine_similarity": cosine_similarity(
            [lexical.get(f"deg_bin_{i}", 0.0) for i in range(N_DEGREE_BINS)],
            [syntactic.get(f"deg_bin_{i}", 0.0) for i in range(N_DEGREE_BINS)],
        ),
        "degree_dist_js_similarity": jensen_shannon_similarity(
            [lexical.get(f"deg_bin_{i}", 0.0) for i in range(N_DEGREE_BINS)],
            [syntactic.get(f"deg_bin_{i}", 0.0) for i in range(N_DEGREE_BINS)],
        ),
        # Lexical structural differences (the node set is shared, so these
        # capture any drift between the two views' token inventories).
        "token_count_diff": _diff(lexical.get("token_count"), syntactic.get("token_count")),
        "unique_token_diff": _diff(lexical.get("unique_token_count"),
                                   syntactic.get("unique_token_count")),
        "avg_token_length_diff": _diff(lexical.get("avg_token_length"),
                                       syntactic.get("avg_token_length")),
        "vocabulary_diversity_diff": _diff(lexical.get("vocabulary_diversity"),
                                           syntactic.get("vocabulary_diversity")),
        "stopword_ratio_diff": _diff(lexical.get("stopword_ratio"),
                                     syntactic.get("stopword_ratio")),
        "punctuation_ratio_diff": _diff(lexical.get("punctuation_ratio"),
                                        syntactic.get("punctuation_ratio")),
        "pos_entropy_diff": _diff(lexical.get("pos_entropy"), syntactic.get("pos_entropy")),
        # Syntactic-view summaries surfaced at sample level for convenience.
        "dep_relation_diversity": syntactic.get("dep_relation_diversity"),
        "dep_relation_entropy": syntactic.get("dep_relation_entropy"),
        "root_count": syntactic.get("root_count"),
        "root_ratio": syntactic.get("root_ratio"),
    }
    return {key: _finite(value) for key, value in features.items()}


# --------------------------------------------------------------------------- #
# batched extraction with cache + resume
# --------------------------------------------------------------------------- #
def _chunk(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fetch_batch(conn, sample_ids: list[str], config: dict[str, Any]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    g = config["graph"]
    rel = g["relationships"]
    edge_types = [rel["lexical_edge"], rel["syntactic_edge"]]

    samples = conn.data(
        f"MATCH (s:{g['sample_label']}) WHERE s.sample_id IN $ids "
        "RETURN " + ", ".join(f"s.{p} AS {p}" for p in SAMPLE_PROPERTIES) + " ORDER BY s.sample_id",
        label="samples", ids=sample_ids,
    )
    graphs = conn.data(
        f"MATCH (s:{g['sample_label']})-[:{rel['sample_to_graph']}]->(gg:{g['graph_label']}) "
        "WHERE s.sample_id IN $ids "
        "RETURN s.sample_id AS sample_id, "
        + ", ".join(f"gg.{p} AS {p}" for p in GRAPH_PROPERTIES) + " ORDER BY gg.graph_id",
        label="graphs", ids=sample_ids,
    )
    graph_ids = [row["graph_id"] for row in graphs]

    token_rows = int(config["extraction"].get("token_page_rows", 500))
    edge_rows = int(config["extraction"].get("edge_page_rows", 800))

    token_body = (
        f"MATCH (gg:{g['graph_label']})-[:{rel['graph_to_token']}]->(t:{g['token_label']}) "
        "WHERE gg.graph_id IN $gids "
        "RETURN gg.graph_id AS graph_id, t.node_id AS node_id, t.position AS position, "
        "t.surface AS surface, t.normalized AS normalized, t.lemma AS lemma, t.pos AS pos, "
        "t.tag AS tag, t.ent_type AS ent_type, t.is_punct AS is_punct, t.is_stop AS is_stop "
        "ORDER BY gg.graph_id, t.position"
    )
    edge_body = (
        f"MATCH (gg:{g['graph_label']})-[:{rel['graph_to_token']}]->(a:{g['token_label']})-[r]->(b:{g['token_label']}) "
        "WHERE gg.graph_id IN $gids AND b.graph_id = gg.graph_id AND type(r) IN $types "
        "RETURN gg.graph_id AS graph_id, type(r) AS rel_type, a.node_id AS source, "
        "b.node_id AS target, r.dependency_label AS dependency_label, r.distance AS distance "
        "ORDER BY gg.graph_id, a.position, b.position, type(r)"
    )
    tokens = _paged_data(conn, token_body, params={"gids": graph_ids},
                         page_rows=token_rows, label="tokens")
    edges = _paged_data(conn, edge_body, params={"gids": graph_ids, "types": edge_types},
                        page_rows=edge_rows, label="edges")
    return samples, graphs, tokens, edges


def _paged_data(conn, query_body: str, *, params: dict[str, Any], page_rows: int,
                label: str) -> list[dict[str, Any]]:
    """Run an ordered query in row-bounded pages (SKIP/LIMIT).

    The Aura endpoint over this network corrupts TLS records once a single
    result exceeds roughly 100 KB, which is far below one graph's token or edge
    set. Paging keeps every Bolt result small; the ordering in ``query_body`` is
    a total order, so the pages are stable.
    """
    rows: list[dict[str, Any]] = []
    skip = 0
    while True:
        page = conn.data(
            query_body + " SKIP $skip LIMIT $limit",
            label=f"{label} (skip={skip})", skip=skip, limit=page_rows, **params,
        )
        rows.extend(page)
        if len(page) < page_rows:
            break
        skip += page_rows
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class _Connection:
    """A driver+session pair that is torn down and rebuilt on any query failure.

    The Aura endpoint over this network drops a TLS connection shortly after a
    large result is transferred, so every query is executed with retry-on-
    reconnect. This keeps a batch of pages running to completion without the
    caller managing connection state.
    """

    def __init__(self, settings: Settings, *, attempts: int, backoff: float, log: bool) -> None:
        self.settings = settings
        self.attempts = attempts
        self.backoff = backoff
        self.log = log
        self.driver = None
        self.session = None

    def open(self) -> None:
        self.driver = connect(self.settings)
        self.session = self.driver.session(database=self.settings.database)

    def reconnect(self) -> None:
        self.close()
        self.open()

    def close(self) -> None:
        try:
            if self.session is not None:
                self.session.close()
        except Exception:
            pass
        finally:
            if self.driver is not None:
                self.driver.close()
            self.session = None
            self.driver = None

    def _execute(self, label: str, query: str, extract, **params):
        retryable = retryable_exceptions()
        last: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                result = self.session.run(query, **params)
                return extract(result)  # consumption happens here; may raise
            except retryable as exc:  # noqa: PERF203 - explicit retry loop
                last = exc
                if self.log:
                    print(f"  {label}: {type(exc).__name__}; reconnecting "
                          f"(attempt {attempt}/{self.attempts})")
                time.sleep(self.backoff * attempt)
                self.reconnect()
        assert last is not None
        raise last

    def data(self, query: str, *, label: str = "query", **params) -> list[dict[str, Any]]:
        return self._execute(label, query, lambda result: result.data(), **params)

    def value(self, query: str, *, label: str = "query", **params) -> list[Any]:
        return self._execute(label, query, lambda result: result.value(), **params)


def extract(config: dict[str, Any], *, limit: int | None = None, resume: bool = True,
            log: bool = True) -> dict[str, Any]:
    started = time.time()
    settings = Settings(config)
    missing = settings.missing()
    if missing:
        raise SystemExit(
            f"Missing Neo4j setting(s) {missing}; configure {config['neo4j']['dotenv_file']}."
        )

    extraction = config["extraction"]
    guards = {
        "max_nodes_for_shortest_path": int(extraction.get("max_nodes_for_shortest_path", 5000)),
    }
    batch_size = int(extraction.get("batch_size", 25))
    cache_dir = resolve(extraction.get("cache_dir", "data/derived/graph_feature_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_cache = cache_dir / "graph_rows.jsonl"
    node_cache = cache_dir / "node_rows.jsonl"
    done_cache = cache_dir / "done_samples.txt"

    if not resume:
        for path in (graph_cache, node_cache, done_cache):
            if path.exists():
                path.unlink()

    graph_rows = _read_jsonl(graph_cache)
    node_rows = _read_jsonl(node_cache)
    done = set(done_cache.read_text(encoding="utf-8").split()) if done_cache.is_file() else set()
    graph_rows = [r for r in graph_rows if r["sample_id"] in done]
    node_rows = [r for r in node_rows if r["sample_id"] in done]

    conn = _Connection(settings, attempts=settings.max_retries,
                       backoff=settings.backoff, log=log)
    conn.open()
    try:
        vocab = discover_vocabularies(conn, config)
        schema = discover_schema(conn, config)
        all_ids = conn.value(
            f"MATCH (s:{config['graph']['sample_label']}) RETURN s.sample_id AS id ORDER BY id",
            label="list samples",
        )
    except Exception:
        conn.close()
        raise
    if limit is not None:
        all_ids = all_ids[:limit]
    todo = [sid for sid in all_ids if sid not in done]

    if log:
        print(f"Build: {config['graph']['build_id']}")
        print(f"Samples: {len(all_ids)} total, {len(done)} cached, {len(todo)} to process")
        print(f"POS vocabulary: {len(vocab['pos'])}; dependency top-{len(vocab['dependency_top'])}")
        print(f"Batch size: {batch_size}")

    pos_vocab, dep_vocab = vocab["pos"], vocab["dependency_top"]
    processed = 0
    total_batches = math.ceil(len(todo) / batch_size) if todo else 0
    try:
        for batch_index, batch in enumerate(_chunk(todo, batch_size), start=1):
            batch_started = time.time()
            samples, graphs, tokens, edges = fetch_batch(conn, batch, config)
            tokens_by_graph: dict[str, list[dict[str, Any]]] = {}
            for token in tokens:
                tokens_by_graph.setdefault(token["graph_id"], []).append(token)
            edges_by_graph: dict[str, list[dict[str, Any]]] = {}
            for edge in edges:
                edges_by_graph.setdefault(edge["graph_id"], []).append(edge)
            sample_meta = {row["sample_id"]: row for row in samples}

            new_graph_rows: list[dict[str, Any]] = []
            new_node_rows: list[dict[str, Any]] = []
            for graph in graphs:
                nodes = tokens_by_graph.get(graph["graph_id"], [])
                graph_edges = edges_by_graph.get(graph["graph_id"], [])
                features = graph_view_features(
                    nodes, graph_edges, view=graph["view"],
                    pos_vocab=pos_vocab, dep_vocab=dep_vocab, guards=guards,
                )
                meta = sample_meta.get(graph["sample_id"], {})
                row: dict[str, Any] = {
                    "sample_id": graph["sample_id"],
                    "graph_id": graph["graph_id"],
                    "view": graph["view"],
                    "build_id": graph["build_id"],
                    "graph_spec_version": graph["graph_spec_version"],
                    "spacy_version": graph["spacy_version"],
                    "model_name": graph["model_name"],
                    "parser_status": graph["parser_status"],
                    "label": meta.get("label"),
                    "source_dataset": meta.get("source_dataset"),
                    "source_task": meta.get("source_task"),
                    "text_type": meta.get("text_type"),
                    "graph_category": meta.get("graph_category"),
                    "paraphrase_status": meta.get("paraphrase_status"),
                    "domain": meta.get("domain"),
                    "ai_model": meta.get("ai_model"),
                    "padben_split": meta.get("padben_split"),
                    "split": meta.get("split"),
                    "group_id": meta.get("group_id"),
                    "source_row_id": meta.get("source_row_id"),
                }
                row.update(features)
                new_graph_rows.append(row)
                new_node_rows.extend(node_feature_rows(row, nodes, graph_edges))

            with graph_cache.open("a", encoding="utf-8") as handle:
                for row in new_graph_rows:
                    handle.write(json.dumps(row) + "\n")
            with node_cache.open("a", encoding="utf-8") as handle:
                for row in new_node_rows:
                    handle.write(json.dumps(row) + "\n")
            with done_cache.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(batch) + "\n")

            graph_rows.extend(new_graph_rows)
            node_rows.extend(new_node_rows)
            done.update(batch)
            processed += len(batch)

            if log:
                elapsed = time.time() - started
                rate = processed / elapsed if elapsed else 0.0
                remaining = (len(todo) - processed) / rate if rate else 0.0
                print(
                    f"[batch {batch_index}/{total_batches}] samples={processed}/{len(todo)} "
                    f"graphs={len(graph_rows)} elapsed={elapsed:5.1f}s "
                    f"eta={remaining:5.1f}s"
                )
    finally:
        conn.close()

    return {
        "graph_rows": graph_rows,
        "node_rows": node_rows,
        "vocab": vocab,
        "schema": schema,
        "build_id": config["graph"]["build_id"],
        "elapsed_seconds": time.time() - started,
        "samples_processed": len(done),
    }


# --------------------------------------------------------------------------- #
# column ordering and schema description
# --------------------------------------------------------------------------- #
GRAPH_ID_COLUMNS = ["sample_id", "graph_id", "view", "build_id", "graph_spec_version",
                    "spacy_version", "model_name", "parser_status"]
METADATA_COLUMNS = ["label", "source_dataset", "source_task", "text_type", "graph_category",
                    "paraphrase_status", "domain", "ai_model", "padben_split", "split",
                    "group_id", "source_row_id"]
NODE_COLUMNS = ["sample_id", "graph_id", "view", "node_id", "position", "surface_length",
                "normalized_length", "lemma_length", "is_punct", "is_stop", "pos", "tag",
                "ent_type", "lexical_degree", "in_degree", "out_degree", "is_root",
                "token_frequency", "is_repeated", "position_ratio", "is_first_token",
                "is_last_token"]


def _graph_feature_columns(graph_rows: list[dict[str, Any]]) -> list[str]:
    ordered = list(GRAPH_ID_COLUMNS) + list(METADATA_COLUMNS)
    seen = set(ordered)
    for row in graph_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _multiview_columns(multiview_rows: list[dict[str, Any]]) -> list[str]:
    ordered = ["sample_id"] + list(METADATA_COLUMNS)
    seen = set(ordered)
    for row in multiview_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def build_multiview(graph_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    lexical_view = config["graph"]["views"]["lexical"]
    syntactic_view = config["graph"]["views"]["syntactic"]
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for row in graph_rows:
        by_sample.setdefault(row["sample_id"], {})[row["view"]] = row

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample):
        views = by_sample[sample_id]
        lexical = views.get(lexical_view)
        syntactic = views.get(syntactic_view)
        if lexical is None or syntactic is None:
            continue
        row: dict[str, Any] = {"sample_id": sample_id}
        for column in METADATA_COLUMNS:
            row[column] = lexical.get(column)
        row.update(multiview_features(lexical, syntactic))
        rows.append(row)
    return rows


FEATURE_DEFINITIONS: dict[str, tuple[str, str]] = {
    # name: (family, definition)
    "n_nodes": ("structural", "Number of token nodes in the view"),
    "n_edges": ("structural", "Number of edge records in the view"),
    "n_unique_edges": ("structural", "Distinct unordered node pairs with an edge"),
    "parallel_edge_count": ("structural", "Edge records minus distinct unordered pairs"),
    "self_loop_count": ("structural", "Edges whose endpoints are identical"),
    "is_directed": ("structural", "Whether the view carries directed dependency edges"),
    "degree_mean": ("structural", "Mean degree of the undirected simple projection"),
    "degree_std": ("structural", "Population standard deviation of degree"),
    "degree_min": ("structural", "Minimum degree"),
    "degree_max": ("structural", "Maximum degree"),
    "degree_median": ("structural", "Median degree"),
    "degree_p90": ("structural", "90th percentile of degree"),
    "degree_cv": ("structural", "Coefficient of variation of degree"),
    "density": ("structural", "Edge density of the undirected simple projection"),
    "avg_clustering": ("structural", "Mean local clustering coefficient"),
    "transitivity": ("structural", "Global clustering coefficient"),
    "triangle_count": ("structural", "Number of triangles"),
    "n_components": ("structural", "Number of connected components"),
    "largest_component_size": ("structural", "Nodes in the largest connected component"),
    "largest_component_ratio": ("structural", "Largest component share of nodes"),
    "isolated_nodes": ("structural", "Nodes with degree zero"),
    "isolated_ratio": ("structural", "Isolated node share"),
    "component_size_mean": ("structural", "Mean component size"),
    "component_size_max": ("structural", "Maximum component size"),
    "avg_shortest_path": ("structural", "Mean shortest path inside the largest component"),
    "diameter": ("structural", "Diameter of the largest component"),
    "n_connected_pairs": ("structural", "Number of connected node pairs in the largest component"),
    "degree_hist_entropy": ("structural", "Entropy of the fixed-bin degree histogram"),
    "token_count": ("lexical", "Number of token nodes"),
    "unique_token_count": ("lexical", "Distinct normalized token strings"),
    "vocabulary_diversity": ("lexical", "Unique tokens per node (type-token ratio)"),
    "repeated_token_ratio": ("lexical", "1 - vocabulary diversity"),
    "avg_token_length": ("lexical", "Mean surface length in characters"),
    "std_token_length": ("lexical", "Standard deviation of surface length"),
    "avg_lemma_length": ("lexical", "Mean lemma length in characters"),
    "stopword_ratio": ("lexical", "Share of tokens flagged as stopwords"),
    "punctuation_ratio": ("lexical", "Share of tokens flagged as punctuation"),
    "n_pos_tags": ("lexical", "Distinct POS tags"),
    "pos_entropy": ("lexical", "Entropy of the POS distribution"),
    "n_entity_types": ("lexical", "Distinct entity types"),
    "entity_token_ratio": ("lexical", "Share of tokens carrying an entity type"),
    "dep_edge_count": ("syntactic", "Number of dependency edges"),
    "dep_relation_diversity": ("syntactic", "Distinct dependency labels"),
    "dep_relation_entropy": ("syntactic", "Entropy of the dependency-label distribution"),
    "dep_avg_out_degree": ("syntactic", "Dependency edges per token"),
    "dep_max_in_degree": ("syntactic", "Maximum dependency in-degree"),
    "dep_max_out_degree": ("syntactic", "Maximum dependency out-degree"),
    "root_count": ("syntactic", "Nodes with no incoming dependency edge"),
    "root_ratio": ("syntactic", "Roots per dependency-active node"),
    "dependency_distance_mean": ("syntactic", "Mean absolute head-child token distance"),
    "dependency_distance_max": ("syntactic", "Maximum absolute head-child token distance"),
    "tree_depth": ("syntactic", "Longest dependency path from a root"),
    "dep_other_count": ("syntactic", "Dependency edges whose label is outside the top-K vocabulary"),
}


def feature_schema_payload(
    graph_rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    multiview_rows: list[dict[str, Any]],
    vocab: dict[str, list[str]],
    schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    graph_columns = _graph_feature_columns(graph_rows)
    multiview_columns = _multiview_columns(multiview_rows)
    identifier = set(config["leakage"]["identifier_columns"])
    metadata = set(config["leakage"]["metadata_columns"])

    def classify(column: str) -> dict[str, Any]:
        if column in identifier:
            return {"predictive": False, "leakage_role": "identifier"}
        if column in metadata:
            return {"predictive": False, "leakage_role": "metadata"}
        if column in ("view", "graph_spec_version", "spacy_version", "model_name",
                      "parser_status", "pos", "tag", "ent_type", "node_id", "position"):
            return {"predictive": False, "leakage_role": "descriptive"}
        family, definition = FEATURE_DEFINITIONS.get(column, ("derived", "Derived feature"))
        return {"predictive": True, "leakage_role": "feature", "family": family,
                "definition": definition}

    def describe(columns: list[str], table: str) -> list[dict[str, Any]]:
        described = []
        for column in columns:
            entry = {"name": column, "table": table}
            entry.update(classify(column))
            described.append(entry)
        return described

    features = []
    for name, (family, definition) in FEATURE_DEFINITIONS.items():
        features.append({"name": name, "family": family, "definition": definition,
                         "predictive": True, "leakage_role": "feature"})
    for label in vocab["pos"]:
        features.append({"name": f"pos_{sanitize(label)}_ratio", "family": "lexical",
                         "definition": f"Share of tokens with POS tag '{label}'",
                         "predictive": True, "leakage_role": "feature"})
    for label in vocab["dependency_top"]:
        features.append({"name": f"dep_{sanitize(label)}_count", "family": "syntactic",
                         "definition": f"Count of dependency edges labelled '{label}'",
                         "predictive": True, "leakage_role": "feature"})
    features.append({"name": "deg_bin_*", "family": "structural",
                     "definition": "Normalized degree histogram over fixed bins "
                                   f"{list(DEGREE_BIN_EDGES[:-1])} + [inf)",
                     "predictive": True, "leakage_role": "feature"})

    multiview_defs = {
        "node_count_diff": "Syntactic nodes minus lexical nodes",
        "edge_count_diff": "Syntactic edges minus lexical edges",
        "node_count_ratio": "Syntactic nodes / lexical nodes",
        "edge_count_ratio": "Syntactic edges / lexical edges",
        "structural_cosine_similarity": "Cosine similarity of the two views' structural vectors",
        "degree_dist_cosine_similarity": "Cosine similarity of the two views' degree histograms",
        "degree_dist_js_similarity": "1 - Jensen-Shannon divergence of the degree histograms",
    }
    for name, definition in multiview_defs.items():
        features.append({"name": name, "family": "multiview", "definition": definition,
                         "predictive": True, "leakage_role": "feature"})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "neo4j-pilot",
        "build_id": config["graph"]["build_id"],
        "feature_set_version": FEATURE_SET_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "graph_spec_version": "spacy_multiview_pilot_r1",
        "views": config["graph"]["views"],
        "interpretation_note": (
            "The two graph views are 'lexical_cooccurrence' and 'syntactic_dependency'. "
            "The master prompt's 'original-view / paraphrased-view' multi-view features are "
            "interpreted as these two real views: per-view aggregates plus cross-view deltas."
        ),
        "null_policy": "undefined features are emitted as empty cells (NaN), never imputed",
        "degree_bin_edges": [None if math.isinf(e) else e for e in DEGREE_BIN_EDGES],
        "guards": {"max_nodes_for_shortest_path":
                   int(config["extraction"].get("max_nodes_for_shortest_path", 5000))},
        "categorical_vocabularies": vocab,
        "discovered_schema": schema,
        "leakage_policy": config["leakage"],
        "tables": {
            "graph_features": {"path": config["outputs"]["graph_features"],
                               "grain": "one row per graph view",
                               "columns": describe(graph_columns, "graph_features")},
            "sample_multiview_features": {
                "path": config["outputs"]["sample_multiview_features"],
                "grain": "one row per sample",
                "columns": describe(multiview_columns, "sample_multiview_features")},
            "node_features": {"path": config["outputs"]["node_features"],
                              "grain": "one row per token node",
                              "columns": describe(NODE_COLUMNS, "node_features")},
        },
        "features": features,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="graph_feature_config")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples")
    parser.add_argument("--no-resume", action="store_true", help="Ignore and clear the cache")
    args = parser.parse_args(argv)

    import pandas as pd

    config = load_config(args.config)
    result = extract(config, limit=args.limit, resume=not args.no_resume)

    graph_rows = result["graph_rows"]
    node_rows = result["node_rows"]
    multiview_rows = build_multiview(graph_rows, config)

    graph_frame = pd.DataFrame(graph_rows).reindex(columns=_graph_feature_columns(graph_rows))
    multiview_frame = pd.DataFrame(multiview_rows).reindex(columns=_multiview_columns(multiview_rows))
    node_frame = pd.DataFrame(node_rows).reindex(columns=NODE_COLUMNS)

    outputs = config["outputs"]
    for frame, key in ((graph_frame, "graph_features"),
                       (multiview_frame, "sample_multiview_features"),
                       (node_frame, "node_features")):
        path = resolve(outputs[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Wrote {path}  rows={len(frame)}  cols={len(frame.columns)}")

    schema_payload = feature_schema_payload(
        graph_rows, node_rows, multiview_rows, result["vocab"], result["schema"], config
    )
    schema_path = resolve(outputs["feature_schema"])
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
    print(f"Wrote {schema_path}")

    # Embeddings are intentionally NOT produced: no validated embedding method
    # is implemented in this stage (see the final report).
    print(f"Skipped embeddings ({outputs['embeddings']}): no validated embedding method.")

    print(f"Samples: {result['samples_processed']}  Graphs: {len(graph_frame)}  "
          f"Nodes: {len(node_frame)}  Elapsed: {result['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
