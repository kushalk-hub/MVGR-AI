"""Read-only Neo4j queries and coverage verification (Person 2)."""

from __future__ import annotations

from typing import Any, Iterable

from graph_engine.neo4j.connection import Neo4jConnection


def build_summary(connection: Neo4jConnection, build_id: str) -> dict[str, Any]:
    with connection.session() as session:
        build = session.run(
            "MATCH (b:GraphBuild {build_id: $build_id}) RETURN b",
            build_id=build_id,
        ).single()
        samples = session.run(
            "MATCH (s:Sample {build_id: $build_id}) RETURN count(s) AS n",
            build_id=build_id,
        ).single()["n"]
        tokens = session.run(
            "MATCH (t:Token {build_id: $build_id}) RETURN count(t) AS n",
            build_id=build_id,
        ).single()["n"]
        relationships = session.run(
            """
            MATCH (:Sample {build_id: $build_id})-[r]->()
            RETURN count(r) AS n
            """,
            build_id=build_id,
        ).single()["n"]
    return {
        "graph_build_id": build_id,
        "registered": build is not None,
        "samples": int(samples),
        "tokens": int(tokens),
        "sample_token_relationships": int(relationships),
    }


def sample_graph(
    connection: Neo4jConnection, *, build_id: str, sample_id: str
) -> dict[str, Any]:
    with connection.session() as session:
        nodes = session.run(
            """
            MATCH (s:Sample {build_id: $build_id, sample_id: $sample_id})-[:HAS_TOKEN]->(t:Token)
            RETURN t.node_id AS node_id, t.token AS token, t.node_type AS node_type,
                   t.position AS position, t.is_stopword AS is_stopword
            ORDER BY t.node_id
            """,
            build_id=build_id,
            sample_id=sample_id,
        ).data()
        edges = session.run(
            """
            MATCH (s:Sample {build_id: $build_id, sample_id: $sample_id})-[:HAS_TOKEN]->(a:Token)
            MATCH (a)-[r]->(b:Token {build_id: $build_id, sample_id: $sample_id})
            RETURN a.node_id AS source, b.node_id AS target, type(r) AS relationship,
                   r.weight AS weight, r.dep AS dependency, r.distance AS distance
            ORDER BY source, target
            """,
            build_id=build_id,
            sample_id=sample_id,
        ).data()
    return {"graph_build_id": build_id, "sample_id": sample_id, "nodes": nodes, "edges": edges}


def missing_samples(
    connection: Neo4jConnection, *, build_id: str, expected_ids: Iterable[str]
) -> list[str]:
    expected = sorted({str(value) for value in expected_ids})
    with connection.session() as session:
        present = session.run(
            "MATCH (s:Sample {build_id: $build_id}) RETURN collect(s.sample_id) AS ids",
            build_id=build_id,
        ).single()["ids"]
    present_set = set(present or [])
    return [sample_id for sample_id in expected if sample_id not in present_set]


def list_builds(connection: Neo4jConnection) -> list[dict[str, Any]]:
    with connection.session() as session:
        return session.run(
            """
            MATCH (b:GraphBuild)
            OPTIONAL MATCH (s:Sample {build_id: b.build_id})
            RETURN b.build_id AS build_id, b.graph_spec_version AS graph_spec_version,
                   b.dataset_version AS dataset_version, b.split_id AS split_id,
                   b.created_at AS created_at, count(s) AS samples
            ORDER BY b.created_at DESC
            """
        ).data()


def delete_build(connection: Neo4jConnection, *, build_id: str, confirm: str) -> int:
    """Delete one build. ``confirm`` must equal ``build_id`` exactly."""
    if confirm != build_id:
        raise ValueError("delete_build requires confirm == build_id")
    with connection.session() as session:
        result = session.run(
            """
            MATCH (n) WHERE n.build_id = $build_id
            WITH collect(n) AS nodes, count(n) AS deleted
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN deleted
            """,
            build_id=build_id,
        ).single()
        session.run(
            "MATCH (b:GraphBuild {build_id: $build_id}) DELETE b",
            build_id=build_id,
        ).consume()
    return int(result["deleted"]) if result else 0
