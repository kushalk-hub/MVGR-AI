"""Build two spaCy-based graph views for the 500-row PADBen/BiScope pilot.

The script performs a complete local preflight before connecting to Aura. It
does not store source text in Neo4j. It requires an explicit LOAD before writes.
Graph profiles are pilot engineering choices, not accepted project ADRs.
"""

from __future__ import annotations

import csv
import getpass
import sys
from pathlib import Path
from typing import Any, Iterable

import spacy
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "derived" / "mvgr_graph_pilot_500.csv"
MODEL_NAME = "en_core_web_sm"
GRAPH_SPEC_VERSION = "spacy_multiview_pilot_r1"
BUILD_ID = "mvgr_pilot500_20261114_spacy_multiview_r1"
CHUNK_SIZE = 1000
SPACY_BATCH_SIZE = 32
COOCCURRENCE_RADIUS = 2

# Keep operational headroom below the documented AuraDB Free hard quotas.
SOFT_NODE_LIMIT = 180_000
SOFT_RELATIONSHIP_LIMIT = 360_000

SAMPLE_PROPERTIES = (
    "label",
    "source_dataset",
    "source_task",
    "text_type",
    "graph_category",
    "paraphrase_status",
    "domain",
    "ai_model",
    "padben_split",
    "split",
    "group_id",
    "source_row_id",
)


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    required = {
        "sample_id", "text", "label", "source_dataset", "source_task", "graph_category"
    }
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id or sample_id in seen:
                raise ValueError("sample_id is blank or duplicated; refusing to build graphs.")
            seen.add(sample_id)
            if row.get("label") not in {"0", "1"}:
                raise ValueError(f"Unexpected label for sample_id {sample_id}.")
            if not (row.get("text") or "").strip():
                raise ValueError(f"Blank text for sample_id {sample_id}.")
            rows.append(row)
    return rows


def sample_properties(row: dict[str, str]) -> dict[str, Any]:
    props: dict[str, Any] = {"sample_id": row["sample_id"].strip()}
    for name in SAMPLE_PROPERTIES:
        value = row.get(name, "")
        if value not in (None, ""):
            props[name] = int(value) if name == "label" else value
    return props


def make_view(sample_id: str, view: str, doc, spacy_version: str):
    graph_id = f"{sample_id}::{view}::{GRAPH_SPEC_VERSION}"
    token_nodes: list[dict[str, Any]] = []
    for token in doc:
        node_id = f"{graph_id}::token::{token.i}"
        token_nodes.append(
            {
                "node_id": node_id,
                "graph_id": graph_id,
                "position": token.i,
                "surface": token.text,
                "normalized": token.text.lower(),
                "lemma": token.lemma_ or token.text.lower(),
                "pos": token.pos_ or "",
                "tag": token.tag_ or "",
                "ent_type": token.ent_type_ or "",
                "is_punct": bool(token.is_punct),
                "is_stop": bool(token.is_stop),
            }
        )

    if view == "lexical_cooccurrence":
        edges: list[dict[str, Any]] = []
        for left in range(len(token_nodes)):
            for right in range(left + 1, min(left + COOCCURRENCE_RADIUS + 1, len(token_nodes))):
                edges.append(
                    {
                        "source_node_id": token_nodes[left]["node_id"],
                        "target_node_id": token_nodes[right]["node_id"],
                        "distance": right - left,
                    }
                )
        rel_type = "CO_OCCURS"
    else:
        edges = []
        for token in doc:
            if token.head.i == token.i:  # spaCy's ROOT token points to itself.
                continue
            edges.append(
                {
                    "source_node_id": f"{graph_id}::token::{token.head.i}",
                    "target_node_id": f"{graph_id}::token::{token.i}",
                    "dependency_label": token.dep_ or "unknown",
                }
            )
        rel_type = "DEPENDS_ON"

    graph = {
        "graph_id": graph_id,
        "sample_id": sample_id,
        "view": view,
        "graph_spec_version": GRAPH_SPEC_VERSION,
        "build_id": BUILD_ID,
        "spacy_version": spacy_version,
        "model_name": MODEL_NAME,
        "token_count": len(token_nodes),
        "edge_count": len(edges),
        "parser_status": "ok",
    }
    return graph, token_nodes, edges, rel_type


def preflight(rows: list[dict[str, str]], nlp, spacy_version: str):
    samples: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    cooccurrence_edges: list[dict[str, Any]] = []
    dependency_edges: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    iterator = ((row.get("text", ""), row) for row in rows)

    for doc, row in nlp.pipe(iterator, as_tuples=True, batch_size=SPACY_BATCH_SIZE):
        sample_id = row["sample_id"].strip()
        source = row.get("source_dataset", "Unknown") or "Unknown"
        category = row.get("graph_category", "Unknown") or "Unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        samples.append({"sample_id": sample_id, "properties": sample_properties(row)})

        for view, target_edges in (
            ("lexical_cooccurrence", cooccurrence_edges),
            ("syntactic_dependency", dependency_edges),
        ):
            graph, view_tokens, view_edges, rel_type = make_view(
                sample_id, view, doc, spacy_version
            )
            graph["relationship_type"] = rel_type
            graphs.append(graph)
            tokens.extend(view_tokens)
            target_edges.extend(view_edges)

    return samples, graphs, tokens, cooccurrence_edges, dependency_edges, source_counts, category_counts


def run_query(session, query: str, rows: list[dict[str, Any]]) -> None:
    for batch in chunks(rows, CHUNK_SIZE):
        session.run(query, rows=batch).consume()


def main() -> int:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Pilot CSV not found: {CSV_PATH}")

    rows = read_rows(CSV_PATH)
    if len(rows) != 500:
        raise ValueError(f"Expected exactly 500 pilot rows; found {len(rows)}.")

    nlp = spacy.load(MODEL_NAME)
    if "parser" not in nlp.pipe_names:
        raise RuntimeError(f"{MODEL_NAME} has no dependency parser; refusing to build two views.")
    spacy_version = spacy.__version__
    (samples, graphs, tokens, cooccurrence_edges, dependency_edges,
     source_counts, category_counts) = preflight(rows, nlp, spacy_version)

    planned_nodes = len(samples) + len(graphs) + len(tokens)
    planned_relationships = len(samples) + len(tokens) + len(cooccurrence_edges) + len(dependency_edges)
    print(f"spaCy: {spacy_version}; model: {MODEL_NAME} {nlp.meta.get('version', 'unknown')}")
    print(f"Prepared {len(samples)} sample records and {len(graphs)} text graphs (2 views/sample).")
    print(f"Categories: {', '.join(f'{k}={v}' for k, v in sorted(category_counts.items()))}")
    print(f"Sources: {', '.join(f'{k}={v}' for k, v in sorted(source_counts.items()))}")
    print(f"Token-occurrence nodes: {len(tokens):,}")
    print(f"Lexical co-occurrence edges: {len(cooccurrence_edges):,}")
    print(f"Syntactic dependency edges: {len(dependency_edges):,}")
    print(f"Projected additions: {planned_nodes:,} nodes and {planned_relationships:,} relationships.")
    print("The full text is not stored in Neo4j. Labels and dataset metadata stay on Sample nodes only.")

    uri = input("Aura URI: ").strip()
    username = input("Aura username: ").strip()
    password = getpass.getpass("Aura password: ")
    database = input("Database name [neo4j]: ").strip() or "neo4j"

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            existing_nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            existing_relationships = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()["count"]
            print(f"Current database: {existing_nodes:,} nodes, {existing_relationships:,} relationships.")
            if existing_nodes + planned_nodes > SOFT_NODE_LIMIT:
                raise RuntimeError(
                    f"Capacity stop: projected total exceeds the {SOFT_NODE_LIMIT:,}-node safety limit. "
                    "No graph writes were made."
                )
            if existing_relationships + planned_relationships > SOFT_RELATIONSHIP_LIMIT:
                raise RuntimeError(
                    f"Capacity stop: projected total exceeds the {SOFT_RELATIONSHIP_LIMIT:,}-relationship "
                    "safety limit. No graph writes were made."
                )

            answer = input("Type LOAD to write this 500-row, two-view pilot to Neo4j: ").strip()
            if answer != "LOAD":
                print("Stopped before writes.")
                return 2

            for query in (
                "CREATE CONSTRAINT sample_id_unique IF NOT EXISTS "
                "FOR (n:Sample) REQUIRE n.sample_id IS UNIQUE",
                "CREATE CONSTRAINT graph_id_unique IF NOT EXISTS "
                "FOR (n:TextGraph) REQUIRE n.graph_id IS UNIQUE",
                "CREATE CONSTRAINT token_node_id_unique IF NOT EXISTS "
                "FOR (n:TokenOccurrence) REQUIRE n.node_id IS UNIQUE",
            ):
                session.run(query).consume()

            sample_query = """
            UNWIND $rows AS row
            MERGE (s:Sample {sample_id: row.sample_id})
            SET s += row.properties
            """
            graph_query = """
            UNWIND $rows AS row
            MATCH (s:Sample {sample_id: row.sample_id})
            MERGE (g:TextGraph {graph_id: row.graph_id})
            SET g.view = row.view,
                g.graph_spec_version = row.graph_spec_version,
                g.build_id = row.build_id,
                g.spacy_version = row.spacy_version,
                g.model_name = row.model_name,
                g.token_count = row.token_count,
                g.edge_count = row.edge_count,
                g.parser_status = row.parser_status
            MERGE (s)-[:HAS_GRAPH]->(g)
            """
            token_query = """
            UNWIND $rows AS row
            MATCH (g:TextGraph {graph_id: row.graph_id})
            MERGE (t:TokenOccurrence {node_id: row.node_id})
            SET t.graph_id = row.graph_id,
                t.position = row.position,
                t.surface = row.surface,
                t.normalized = row.normalized,
                t.lemma = row.lemma,
                t.pos = row.pos,
                t.tag = row.tag,
                t.ent_type = row.ent_type,
                t.is_punct = row.is_punct,
                t.is_stop = row.is_stop
            MERGE (g)-[:HAS_TOKEN]->(t)
            """
            cooccurrence_query = """
            UNWIND $rows AS row
            MATCH (a:TokenOccurrence {node_id: row.source_node_id})
            MATCH (b:TokenOccurrence {node_id: row.target_node_id})
            MERGE (a)-[r:CO_OCCURS]->(b)
            SET r.distance = row.distance,
                r.edge_semantics = 'undirected',
                r.storage_orientation = 'increasing_token_position'
            """
            dependency_query = """
            UNWIND $rows AS row
            MATCH (head:TokenOccurrence {node_id: row.source_node_id})
            MATCH (child:TokenOccurrence {node_id: row.target_node_id})
            MERGE (head)-[r:DEPENDS_ON]->(child)
            SET r.dependency_label = row.dependency_label,
                r.direction_semantics = 'head_to_dependent'
            """

            run_query(session, sample_query, samples)
            run_query(session, graph_query, graphs)
            run_query(session, token_query, tokens)
            run_query(session, cooccurrence_query, cooccurrence_edges)
            run_query(session, dependency_query, dependency_edges)

            result = session.run(
                "MATCH (s:Sample)-[:HAS_GRAPH]->(g:TextGraph {build_id: $build_id}) "
                "OPTIONAL MATCH (g)-[:HAS_TOKEN]->(t:TokenOccurrence) "
                "RETURN count(DISTINCT s) AS samples, count(DISTINCT g) AS graphs, "
                "count(DISTINCT t) AS tokens",
                build_id=BUILD_ID,
            ).single()
            # Scope edge counts to this build via endpoint graph IDs.
            stored_cooccurrence = session.run(
                "MATCH (a:TokenOccurrence)-[r:CO_OCCURS]->(b:TokenOccurrence) "
                "WHERE a.graph_id IN $ids AND b.graph_id IN $ids RETURN count(r) AS count",
                ids=[g["graph_id"] for g in graphs],
            ).single()["count"]
            stored_dependencies = session.run(
                "MATCH (a:TokenOccurrence)-[r:DEPENDS_ON]->(b:TokenOccurrence) "
                "WHERE a.graph_id IN $ids AND b.graph_id IN $ids RETURN count(r) AS count",
                ids=[g["graph_id"] for g in graphs],
            ).single()["count"]
            view_check = session.run(
                "MATCH (s:Sample)-[:HAS_GRAPH]->(g:TextGraph {build_id: $build_id}) "
                "WITH s, collect(DISTINCT g.view) AS views "
                "WHERE size(views) <> 2 OR NOT ('lexical_cooccurrence' IN views) "
                "OR NOT ('syntactic_dependency' IN views) "
                "RETURN count(s) AS invalid_samples",
                build_id=BUILD_ID,
            ).single()["invalid_samples"]
            cross_graph_edges = session.run(
                "MATCH (a:TokenOccurrence)-[r]->(b:TokenOccurrence) "
                "WHERE type(r) IN ['CO_OCCURS', 'DEPENDS_ON'] "
                "AND a.graph_id IN $ids AND b.graph_id IN $ids AND a.graph_id <> b.graph_id "
                "RETURN count(r) AS count",
                ids=[g["graph_id"] for g in graphs],
            ).single()["count"]
            print(
                f"Stored build {BUILD_ID}: {result['samples']} samples, {result['graphs']} graphs, "
                f"{result['tokens']:,} token nodes, {stored_cooccurrence:,} co-occurrence edges, "
                f"{stored_dependencies:,} dependency edges."
            )
            print(f"Samples missing either graph view: {view_check}; cross-graph edges: {cross_graph_edges}.")
            if (
                result["samples"] != len(samples)
                or result["graphs"] != len(graphs)
                or result["tokens"] != len(tokens)
                or stored_cooccurrence != len(cooccurrence_edges)
                or stored_dependencies != len(dependency_edges)
                or view_check != 0
                or cross_graph_edges != 0
            ):
                print("Warning: stored totals differ from preflight; inspect the build before scaling.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
