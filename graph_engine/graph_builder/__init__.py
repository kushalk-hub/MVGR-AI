"""Graph builder dispatch (Person 2)."""

from __future__ import annotations

from typing import Any

from graph_engine.graph_builder.lexical_graph import build_lexical_graph
from graph_engine.graph_builder.syntactic_graph import (
    ParserUnavailable,
    build_syntactic_graph,
    load_pipeline,
)

BUILDERS = {
    "lexical": build_lexical_graph,
    "syntactic": build_syntactic_graph,
}


def build_graph(
    view: str,
    *,
    sample_id: str,
    text: str,
    graph_build_id: str,
    config_id: str,
    view_config: dict[str, Any],
    graph_spec_version: str,
) -> dict[str, Any]:
    try:
        builder = BUILDERS[view]
    except KeyError as exc:
        raise ValueError(f"unknown graph view {view!r}; known: {sorted(BUILDERS)}") from exc
    return builder(
        sample_id=sample_id,
        text=text,
        graph_build_id=graph_build_id,
        config_id=config_id,
        view_config=view_config,
        graph_spec_version=graph_spec_version,
    )


__all__ = ["BUILDERS", "ParserUnavailable", "build_graph", "load_pipeline"]
