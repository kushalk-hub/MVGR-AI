"""Versioned graph export writer and build runner (Person 2).

Exports are the interface between Person 2 and Person 1. They contain only
structure and identifier metadata. Labels, source, task, family, text type,
condition and split are never written per sample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from graph_engine.graph_builder import build_graph
from shared.contracts.ids import config_id as make_config_id
from shared.contracts.ids import graph_build_id as make_graph_build_id
from shared.contracts.validators import (
    utc_now,
    validate_graph_record,
    write_json,
)

LAST_BUILD_POINTER = "graph_builds/LATEST.json"


class ExportError(ValueError):
    pass


def read_export(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExportError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_view_export(
    records: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    build_meta: dict[str, Any],
    validate: bool = True,
) -> dict[str, Any]:
    """Write one graph view as JSONL plus a per-view manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    view = None
    count = 0
    empty = 0
    truncated = 0
    nodes = 0
    edges = 0
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    sample_ids: set[str] = set()

    path = output_dir / f"{build_meta['graph_view']}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            view = record["graph_view"]
            if validate:
                validate_graph_record(record)
            if record["sample_id"] in sample_ids:
                raise ExportError(f"duplicate sample_id in export: {record['sample_id']}")
            sample_ids.add(record["sample_id"])
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            empty += int(bool(record["empty"]))
            truncated += int(bool(record.get("truncated", False)))
            nodes += len(record["nodes"])
            edges += len(record["edges"])
            for node in record["nodes"]:
                node_types[node["node_type"]] = node_types.get(node["node_type"], 0) + 1
            for edge in record["edges"]:
                edge_types[edge["edge_type"]] = edge_types.get(edge["edge_type"], 0) + 1

    payload = path.read_bytes()
    manifest = {
        **build_meta,
        "graph_view": view,
        "created_at": utc_now(),
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "counts": {
            "samples": count,
            "empty_graphs": empty,
            "truncated_graphs": truncated,
            "nodes": nodes,
            "edges": edges,
            "node_types": node_types,
            "edge_types": edge_types,
        },
    }
    write_json(output_dir / f"{build_meta['graph_view']}.manifest.json", manifest)
    return manifest


def build_view_export(
    *,
    frame: pd.DataFrame,
    sample_ids: Iterable[str],
    view: str,
    output_root: str | Path,
    dataset_version: str,
    split_id: str,
    graph_config: dict[str, Any],
    config_snapshot: dict[str, Any] | None = None,
    source_description: str = "",
) -> dict[str, Any]:
    """Build and export one graph view for exactly ``sample_ids``.

    ``frame`` must contain only ``sample_id`` and ``text``. Passing a frame that
    also contains labels is rejected, so target metadata cannot leak into an
    export by accident.
    """
    forbidden = {"label", "source_dataset", "source_task", "text_type", "family_id", "split"}
    leaked = sorted(forbidden & set(frame.columns))
    if leaked:
        raise ExportError(
            f"graph build input contains target metadata columns {leaked}. "
            "Pass sample_id/text only; labels must never enter a graph artifact."
        )

    allowed = {str(value) for value in sample_ids}
    if not allowed:
        raise ExportError("no sample ids supplied for graph build")
    subset = frame[frame["sample_id"].astype(str).isin(allowed)][["sample_id", "text"]].copy()
    missing = allowed - set(subset["sample_id"].astype(str))
    if missing:
        raise ExportError(f"{len(missing)} requested sample ids are absent from the dataset frame")
    subset["sample_id"] = subset["sample_id"].astype(str)
    subset = subset.sort_values("sample_id").reset_index(drop=True)

    spec_version = str(graph_config["graph_spec_version"])
    view_config = dict(graph_config["views"][view])
    view_config["shared_max_nodes"] = int(graph_config["shared"]["max_nodes"])
    cfg_id = make_config_id(
        {
            "graph_spec_version": spec_version,
            "view": view,
            "view_config": view_config,
            "tokenizer": graph_config["shared"]["tokenizer_version"],
        }
    )
    build_id = make_graph_build_id(
        dataset_version=dataset_version,
        split_id_value=split_id,
        graph_spec_version=spec_version,
        config=view_config,
        sample_ids=sorted(allowed),
    )

    output_dir = Path(output_root) / build_id
    failures: list[dict[str, str]] = []

    def records() -> Iterator[dict[str, Any]]:
        for row in subset.itertuples(index=False):
            try:
                yield build_graph(
                    view,
                    sample_id=str(row.sample_id),
                    text=str(row.text),
                    graph_build_id=build_id,
                    config_id=cfg_id,
                    view_config=view_config,
                    graph_spec_version=spec_version,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                failures.append({"sample_id": str(row.sample_id), "reason": repr(exc)})
                if view == "syntactic" and "ParserUnavailable" in type(exc).__name__:
                    raise
                if len(failures) >= 50:
                    raise ExportError(
                        f"aborting graph build after {len(failures)} failures; first={failures[0]}"
                    ) from exc

    manifest = write_view_export(
        records(),
        output_dir=output_dir,
        build_meta={
            "graph_build_id": build_id,
            "graph_view": view,
            "graph_spec_version": spec_version,
            "config_id": cfg_id,
            "dataset_version": dataset_version,
            "split_id": split_id,
            "tokenizer_version": graph_config["shared"]["tokenizer_version"],
            "config": config_snapshot or view_config,
            "source_description": source_description,
            "target_metadata_in_export": False,
        },
        validate=True,
    )

    if failures:
        write_json(output_dir / f"{view}.failures.json", failures)
    manifest["failures"] = len(failures)

    write_json(
        Path(output_root) / LAST_BUILD_POINTER,
        {
            "graph_build_id": build_id,
            "graph_view": view,
            "path": str(output_dir / f"{view}.jsonl"),
            "manifest": str(output_dir / f"{view}.manifest.json"),
        },
    )
    return manifest


def write_build_manifest(
    *,
    build_id: str,
    output_root: str | Path,
    view_manifests: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
    dataset_version: str,
    split_id: str,
) -> Path:
    """Write the build-level manifest that references every view export."""
    payload = {
        "graph_build_id": build_id,
        "graph_spec_version": config_snapshot.get("graph_spec_version"),
        "dataset_version": dataset_version,
        "split_id": split_id,
        "created_at": utc_now(),
        "config": config_snapshot,
        "views": {
            manifest["graph_view"]: {
                "path": manifest["path"],
                "sha256": manifest["sha256"],
                "counts": manifest["counts"],
                "failures": manifest.get("failures", 0),
            }
            for manifest in view_manifests
        },
        "note": "Exports contain no labels, source, task, family, text_type, condition or split per sample.",
    }
    return write_json(Path(output_root) / build_id / "build.manifest.json", payload)
