"""Schema loading and validation for every shared contract.

Fail closed: a contract violation raises ``ContractError``. There is no
"best effort" mode, because silently accepting a malformed artifact is how
label leakage and irreproducible results enter a project.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker

from . import versions as V

CONTRACT_DIR = Path(__file__).resolve().parent

SCHEMA_FILES: dict[str, str] = {
    "graph": "graph_schema.json",
    "prediction": "prediction_schema.json",
    "manifest": "manifest_schema.json",
    "family_index": "family_index_schema.json",
    "experiment": "experiment_schema.json",
    "analysis_report": "analysis_report_schema.json",
}

CONTRACT_VERSIONS: dict[str, str] = {
    "graph": V.GRAPH_SCHEMA_VERSION,
    "prediction": V.PREDICTION_SCHEMA_VERSION,
    "manifest": V.MANIFEST_SCHEMA_VERSION,
    "family_index": V.FAMILY_INDEX_SCHEMA_VERSION,
    "experiment": V.EXPERIMENT_SCHEMA_VERSION,
    "analysis_report": V.REPORT_SCHEMA_VERSION,
}


class ContractError(ValueError):
    """Raised when an artifact does not satisfy its published contract."""


def load_schema(name: str) -> dict[str, Any]:
    try:
        filename = SCHEMA_FILES[name]
    except KeyError as exc:
        raise ContractError(f"Unknown contract {name!r}. Known: {sorted(SCHEMA_FILES)}") from exc
    return json.loads((CONTRACT_DIR / filename).read_text(encoding="utf-8"))


def validate(name: str, instance: Any) -> None:
    """Validate ``instance`` against contract ``name``; raise on failure."""
    validator = Draft202012Validator(load_schema(name), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if errors:
        rendered = []
        for err in errors[:5]:
            location = "/".join(str(p) for p in err.absolute_path) or "<root>"
            rendered.append(f"{location}: {err.message}")
        more = "" if len(errors) <= 5 else f" (+{len(errors) - 5} more)"
        raise ContractError(f"{name} contract violation{more}: " + "; ".join(rendered))


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{number}: invalid JSON ({exc})") from exc


def validate_graph_record(record: dict[str, Any]) -> None:
    """Full validation for one graph export record, including internal consistency."""
    validate("graph", record)
    nodes, edges = record["nodes"], record["edges"]
    node_ids = [node["node_id"] for node in nodes]
    if node_ids != list(range(len(nodes))):
        raise ContractError("graph: node_id values must be 0..N-1 in array order")
    if record["empty"] and (nodes or edges):
        raise ContractError("graph: empty=true requires zero nodes and zero edges")
    if not record["empty"] and not nodes:
        raise ContractError("graph: empty=false requires at least one node")
    for edge in edges:
        if edge["source"] not in range(len(nodes)) or edge["target"] not in range(len(nodes)):
            raise ContractError(f"graph: edge references unknown node in {record['sample_id']!r}")
    if record["graph_view"] == "syntactic" and any(e["edge_type"] != "dependency" for e in edges):
        raise ContractError("graph: syntactic view may only contain dependency edges")
    if record["graph_view"] == "lexical" and any(e["edge_type"] != "cooccurrence" for e in edges):
        raise ContractError("graph: lexical view may only contain cooccurrence edges")


def validate_graph_file(path: str | Path) -> int:
    path = Path(path)
    count = 0
    for _, record in _iter_jsonl(path):
        validate_graph_record(record)
        count += 1
    return count


def validate_prediction_file(path: str | Path) -> int:
    path = Path(path)
    count = 0
    seen: set[str] = set()
    for _, record in _iter_jsonl(path):
        validate("prediction", record)
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ContractError(f"{path}: duplicate prediction for sample_id {sample_id!r}")
        seen.add(sample_id)
        count += 1
    return count


def validate_manifest(manifest: dict[str, Any]) -> None:
    validate("manifest", manifest)
    if not manifest["manifest_id"].startswith("man_"):
        raise ContractError("manifest: manifest_id must use the man_ prefix")


def validate_analysis_report(report: dict[str, Any]) -> None:
    validate("analysis_report", report)
    if report["watermark"]["status"] == "not_detected" and report["classification"].get("label") == "human":
        raise ContractError(
            "analysis_report: absence of a watermark must not be reported as human authorship"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path
