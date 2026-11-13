"""Lexical co-occurrence graph builder (Person 2).

Specification: ``configs/graph_config.yaml`` -> ``views.lexical``
Contract: ``shared/contracts/graph_schema.json``

Nodes are distinct normalized tokens per sample. Edges are undirected
co-occurrence counts inside a sentence-local window. Nothing in the record
identifies the label, source, task, family, text type, condition or split.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from shared.preprocessing.text import normalize_text, tokenize


def _empty_record(
    sample_id: str, graph_build_id: str, config_id: str, graph_spec_version: str
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "graph_view": "lexical",
        "graph_spec_version": graph_spec_version,
        "graph_build_id": graph_build_id,
        "config_id": config_id,
        "nodes": [],
        "edges": [],
        "empty": True,
        "truncated": False,
    }


def build_lexical_graph(
    *,
    sample_id: str,
    text: str,
    graph_build_id: str,
    config_id: str,
    view_config: dict[str, Any],
    graph_spec_version: str,
) -> dict[str, Any]:
    max_nodes = int(view_config.get("shared_max_nodes", 20000))
    window = int(view_config["edge"]["window"])

    tokens = tokenize(normalize_text(text))
    if not tokens:
        return _empty_record(sample_id, graph_build_id, config_id, graph_spec_version)

    frequencies = Counter(token.text for token in tokens)
    first_position: dict[str, int] = {}
    first_sentence: dict[str, int] = {}
    stopword: dict[str, bool] = {}
    for token in tokens:
        if token.text not in first_position:
            first_position[token.text] = token.position
            first_sentence[token.text] = token.sentence_index
            stopword[token.text] = token.is_stopword

    truncated = False
    keep = set(first_position)
    if len(keep) > max_nodes:
        ordered = sorted(keep, key=lambda token: (-frequencies[token], token))[:max_nodes]
        keep = set(ordered)
        truncated = True

    ordered_tokens = sorted(keep, key=lambda token: (first_position[token], token))
    node_id = {token: index for index, token in enumerate(ordered_tokens)}
    nodes = [
        {
            "node_id": node_id[token],
            "token": token,
            "node_type": "token",
            "position": int(first_position[token]),
            "sentence_index": int(first_sentence[token]),
            "is_stopword": bool(stopword[token]),
        }
        for token in ordered_tokens
    ]

    # Sentence-local window edges, aggregated per unordered node pair.
    edge_weights: dict[tuple[int, int], int] = {}
    edge_distance: dict[tuple[int, int], int] = {}
    sentence_tokens: dict[int, list[Any]] = {}
    for token in tokens:
        if token.text in keep:
            sentence_tokens.setdefault(token.sentence_index, []).append(token)

    for sentence in sentence_tokens.values():
        for left_index, left in enumerate(sentence):
            for right_index in range(left_index + 1, min(left_index + 1 + window, len(sentence))):
                right = sentence[right_index]
                if left.text == right.text:
                    continue
                low, high = sorted((node_id[left.text], node_id[right.text]))
                key = (low, high)
                edge_weights[key] = edge_weights.get(key, 0) + 1
                distance = int(right.position - left.position)
                edge_distance[key] = min(edge_distance.get(key, distance), distance)

    edges = [
        {
            "source": int(low),
            "target": int(high),
            "edge_type": "cooccurrence",
            "weight": int(weight),
            "directed": False,
            "distance": int(edge_distance[(low, high)]),
        }
        for (low, high), weight in sorted(edge_weights.items())
    ]

    return {
        "sample_id": sample_id,
        "graph_view": "lexical",
        "graph_spec_version": graph_spec_version,
        "graph_build_id": graph_build_id,
        "config_id": config_id,
        "nodes": nodes,
        "edges": edges,
        "empty": False,
        "truncated": truncated,
    }
