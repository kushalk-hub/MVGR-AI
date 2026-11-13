"""Idempotent, batched Neo4j ingestion (Person 2).

Every node carries a synthetic ``uid`` = ``build_id:sample_id:node_id``. That
makes ingestion idempotent without composite constraints and keeps builds
isolated. Failures are collected per sample and written to a failure report;
they are never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from graph_engine.graph_export.export import read_export
from graph_engine.neo4j.connection import Neo4jConnection
from shared.contracts.validators import utc_now, write_json


@dataclass
class FailureReport:
    build_id: str
    graph_view: str
    attempted: int = 0
    loaded: int = 0
    failed: int = 0
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    failures: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphLoader:
    def __init__(self, connection: Neo4jConnection, *, batch_size: int = 500) -> None:
        self.connection = connection
        self.batch_size = batch_size

    def _sample_uid(self, build_id: str, sample_id: str) -> str:
        return f"{build_id}:{sample_id}"

    def _token_uid(self, build_id: str, sample_id: str, node_id: int) -> str:
        return f"{build_id}:{sample_id}:{node_id}"

    def _write_record(self, tx: Any, record: dict[str, Any], build_id: str) -> None:
        sample_id = str(record["sample_id"])
        sample_uid = self._sample_uid(build_id, sample_id)
        nodes = [
            {
                "uid": self._token_uid(build_id, sample_id, int(node["node_id"])),
                "node_id": int(node["node_id"]),
                "token": node["token"],
                "node_type": node["node_type"],
                "position": node.get("position"),
                "is_stopword": node.get("is_stopword"),
            }
            for node in record["nodes"]
        ]
        tx.run(
            """
            MERGE (s:Sample {uid: $sample_uid})
            SET s.build_id = $build_id,
                s.sample_id = $sample_id,
                s.graph_view = $graph_view,
                s.node_count = $node_count,
                s.edge_count = $edge_count,
                s.empty = $empty
            WITH s
            UNWIND $nodes AS node
            MERGE (t:Token {uid: node.uid})
            SET t.build_id = $build_id,
                t.sample_id = $sample_id,
                t.node_id = node.node_id,
                t.token = node.token,
                t.node_type = node.node_type,
                t.position = node.position,
                t.is_stopword = node.is_stopword
            MERGE (s)-[:HAS_TOKEN]->(t)
            """,
            sample_uid=sample_uid,
            build_id=build_id,
            sample_id=sample_id,
            graph_view=record["graph_view"],
            node_count=len(record["nodes"]),
            edge_count=len(record["edges"]),
            empty=bool(record["empty"]),
            nodes=nodes,
        ).consume()

        grouped: dict[str, list[dict[str, Any]]] = {"cooccurrence": [], "dependency": []}
        for edge in record["edges"]:
            grouped[edge["edge_type"]].append(
                {
                    "source_uid": self._token_uid(build_id, sample_id, int(edge["source"])),
                    "target_uid": self._token_uid(build_id, sample_id, int(edge["target"])),
                    "weight": float(edge["weight"]),
                    "distance": edge.get("distance"),
                    "dep": edge.get("dependency"),
                }
            )

        if grouped["cooccurrence"]:
            tx.run(
                """
                UNWIND $edges AS edge
                MATCH (a:Token {uid: edge.source_uid})
                MATCH (b:Token {uid: edge.target_uid})
                MERGE (a)-[r:CO_OCCURS_WITH]->(b)
                SET r.weight = edge.weight,
                    r.distance = edge.distance,
                    r.build_id = $build_id
                """,
                edges=grouped["cooccurrence"],
                build_id=build_id,
            ).consume()
        if grouped["dependency"]:
            tx.run(
                """
                UNWIND $edges AS edge
                MATCH (a:Token {uid: edge.source_uid})
                MATCH (b:Token {uid: edge.target_uid})
                MERGE (a)-[r:DEPENDS_ON]->(b)
                SET r.dep = edge.dep,
                    r.weight = edge.weight,
                    r.distance = edge.distance,
                    r.build_id = $build_id
                """,
                edges=grouped["dependency"],
                build_id=build_id,
            ).consume()

    def _write_batch(self, tx: Any, batch: list[dict[str, Any]], build_id: str) -> None:
        for record in batch:
            self._write_record(tx, record, build_id)

    def load_export(
        self,
        export_path: str | Path,
        *,
        build_id: str,
        graph_view: str,
        report_path: str | Path | None = None,
        expected_samples: int | None = None,
    ) -> FailureReport:
        report = FailureReport(build_id=build_id, graph_view=graph_view)
        batch: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal batch
            if not batch:
                return
            try:
                with self.connection.session() as session:
                    session.execute_write(self._write_batch, batch, build_id)
                report.loaded += len(batch)
            except Exception as batch_error:  # noqa: BLE001 - isolate per record
                for record in batch:
                    try:
                        with self.connection.session() as session:
                            session.execute_write(self._write_record, record, build_id)
                        report.loaded += 1
                    except Exception as record_error:  # noqa: BLE001 - recorded
                        report.failed += 1
                        if len(report.failures) < 100:
                            report.failures.append(
                                {
                                    "sample_id": str(record.get("sample_id")),
                                    "reason": f"{type(record_error).__name__}: {record_error}",
                                    "batch_error": f"{type(batch_error).__name__}: {batch_error}",
                                }
                            )
            batch = []

        for record in read_export(export_path):
            report.attempted += 1
            batch.append(record)
            if len(batch) >= self.batch_size:
                flush()
        flush()

        report.finished_at = utc_now()
        if expected_samples is not None and report.loaded != expected_samples:
            report.failures.append(
                {
                    "sample_id": "<coverage>",
                    "reason": f"loaded {report.loaded} of {expected_samples} expected samples",
                }
            )
        if report_path is not None:
            write_json(report_path, report.as_dict())
        return report
