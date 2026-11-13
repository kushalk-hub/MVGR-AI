"""Syntactic dependency graph builder (Person 2).

Specification: ``configs/graph_config.yaml`` -> ``views.syntactic``
Contract: ``shared/contracts/graph_schema.json``

Nodes are non-punctuation token occurrences; edges point from head to child and
carry the dependency label and token distance. If spaCy or the configured model
is unavailable the build FAILS. It never falls back to the lexical view.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from shared.preprocessing.text import normalize_text

EXCLUDED_POS = {"PUNCT", "SPACE", "SYMBOL"}


class ParserUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=4)
def load_pipeline(model_name: str):
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ParserUnavailable(
            "spaCy is not installed. Run `pip install -r requirements/graph.txt` and "
            f"`python -m spacy download {model_name}`."
        ) from exc
    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - depends on environment
        raise ParserUnavailable(
            f"spaCy model {model_name!r} is not installed. "
            f"Run `python -m spacy download {model_name}`."
        ) from exc


def _empty_record(
    sample_id: str, graph_build_id: str, config_id: str, graph_spec_version: str
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "graph_view": "syntactic",
        "graph_spec_version": graph_spec_version,
        "graph_build_id": graph_build_id,
        "config_id": config_id,
        "nodes": [],
        "edges": [],
        "empty": True,
        "truncated": False,
    }


def build_syntactic_graph(
    *,
    sample_id: str,
    text: str,
    graph_build_id: str,
    config_id: str,
    view_config: dict[str, Any],
    graph_spec_version: str,
) -> dict[str, Any]:
    normalized = normalize_text(text)
    if not normalized:
        return _empty_record(sample_id, graph_build_id, config_id, graph_spec_version)

    nlp = load_pipeline(str(view_config.get("model", "en_core_web_sm")))
    doc = nlp(normalized)

    sentence_index: dict[int, int] = {}
    for index, sentence in enumerate(doc.sents):
        for token in sentence:
            sentence_index[token.i] = index

    tokens = [
        token
        for token in doc
        if not token.is_space and token.pos_ not in EXCLUDED_POS and token.i in sentence_index
    ]
    if not tokens:
        return _empty_record(sample_id, graph_build_id, config_id, graph_spec_version)

    node_id = {token.i: index for index, token in enumerate(tokens)}
    nodes = [
        {
            "node_id": node_id[token.i],
            "token": token.text.casefold(),
            "node_type": "token",
            "position": int(token.i),
            "sentence_index": int(sentence_index[token.i]),
            "pos": str(token.pos_),
            "is_stopword": bool(token.is_stop),
        }
        for token in tokens
    ]

    edges = []
    for token in tokens:
        head = token.head
        if head.i == token.i or head.i not in node_id:
            continue
        edges.append(
            {
                "source": int(node_id[head.i]),
                "target": int(node_id[token.i]),
                "edge_type": "dependency",
                "dependency": str(token.dep_),
                "weight": 1.0,
                "directed": True,
                "sentence_index": int(sentence_index[token.i]),
                "distance": int(abs(token.i - head.i)),
            }
        )
    edges.sort(key=lambda edge: (edge["source"], edge["target"], edge["dependency"]))

    return {
        "sample_id": sample_id,
        "graph_view": "syntactic",
        "graph_spec_version": graph_spec_version,
        "graph_build_id": graph_build_id,
        "config_id": config_id,
        "nodes": nodes,
        "edges": edges,
        "empty": False,
        "truncated": False,
    }
