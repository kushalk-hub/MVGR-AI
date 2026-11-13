"""Neo4j package (Person 2). Imports are driver-free at module load."""

from graph_engine.neo4j.connection import Neo4jConnection, Neo4jSettings, Neo4jUnavailable
from graph_engine.neo4j.loader import FailureReport, GraphLoader
from graph_engine.neo4j.queries import (
    build_summary,
    delete_build,
    list_builds,
    missing_samples,
    sample_graph,
)
from graph_engine.neo4j.schema import apply_schema, register_build, schema_statements

__all__ = [
    "FailureReport",
    "GraphLoader",
    "Neo4jConnection",
    "Neo4jSettings",
    "Neo4jUnavailable",
    "apply_schema",
    "build_summary",
    "delete_build",
    "list_builds",
    "missing_samples",
    "register_build",
    "sample_graph",
    "schema_statements",
]
