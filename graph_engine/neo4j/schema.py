"""Neo4j schema management (Person 2).

All objects are namespaced by ``build_id`` through a synthetic ``uid`` so that
two builds can coexist and an earlier build is never overwritten.
"""

from __future__ import annotations

from typing import Any

from graph_engine.neo4j.connection import Neo4jConnection


def schema_statements(graph_config: dict[str, Any]) -> list[str]:
    neo4j_config = graph_config["neo4j"]
    return [*neo4j_config.get("constraints", []), *neo4j_config.get("indexes", [])]


def apply_schema(connection: Neo4jConnection, graph_config: dict[str, Any]) -> list[str]:
    applied: list[str] = []
    with connection.session() as session:
        for statement in schema_statements(graph_config):
            session.run(statement).consume()
            applied.append(statement)
    return applied


def register_build(
    connection: Neo4jConnection,
    *,
    build_id: str,
    graph_spec_version: str,
    dataset_version: str,
    split_id: str,
    config_id: str,
    sample_count: int,
) -> None:
    with connection.session() as session:
        session.run(
            """
            MERGE (b:GraphBuild {build_id: $build_id})
            ON CREATE SET b.created_at = datetime()
            SET b.graph_spec_version = $graph_spec_version,
                b.dataset_version = $dataset_version,
                b.split_id = $split_id,
                b.config_id = $config_id,
                b.sample_count = $sample_count
            """,
            build_id=build_id,
            graph_spec_version=graph_spec_version,
            dataset_version=dataset_version,
            split_id=split_id,
            config_id=config_id,
            sample_count=sample_count,
        ).consume()
