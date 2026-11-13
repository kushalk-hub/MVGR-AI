"""Experiment registry and artifact tracking shared by both owners.

File-based and append-only. Every run is registered before any metric exists,
so a metric can never be orphaned from its configuration and split.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from shared.contracts.ids import digest
from shared.contracts.validators import utc_now, validate, write_json
from shared.paths import REPO_ROOT, experiment_config, resolve


def git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": "no-git", "dirty": True}


def artifact_root(config: dict[str, Any] | None = None) -> Path:
    config = config or experiment_config()
    return resolve(config["experiment"]["artifact_root"])


def experiment_dir(experiment_id: str, config: dict[str, Any] | None = None) -> Path:
    path = artifact_root(config) / "runs" / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def register_experiment(
    *,
    experiment_id: str,
    config_id: str,
    run_config: dict[str, Any],
    dataset_version: str,
    split_id: str,
    graph_build_id: str,
    graph_spec_version: str,
    model_family: str,
    seeds: list[int],
    notes: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = git_state()
    record = {
        "experiment_schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "config_id": config_id,
        "config": run_config,
        "dataset_version": dataset_version,
        "split_id": split_id,
        "graph_build_id": graph_build_id,
        "graph_spec_version": graph_spec_version,
        "model_family": model_family,
        "seeds": seeds,
        "git_commit": state["commit"],
        "status": "registered",
        "notes": notes,
    }
    validate("experiment", record)
    root = experiment_dir(experiment_id, config)
    write_json(root / "experiment.json", record)
    registry = artifact_root(config) / "runs" / "registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def update_status(
    experiment_id: str, status: str, *, config: dict[str, Any] | None = None
) -> None:
    if status not in {"registered", "running", "completed", "failed", "invalidated"}:
        raise ValueError(f"unknown experiment status {status!r}")
    root = experiment_dir(experiment_id, config)
    path = root / "experiment.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = status
    validate("experiment", record)
    write_json(path, record)


def log_metrics(
    experiment_id: str, metrics: dict[str, Any], *, config: dict[str, Any] | None = None
) -> Path:
    path = artifact_root(config) / "runs" / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"experiment_id": experiment_id, "logged_at": utc_now(), "metrics": metrics}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def run_dir_for(experiment_id: str, config: dict[str, Any] | None = None) -> Path:
    return experiment_dir(experiment_id, config)


def config_fingerprint(run_config: dict[str, Any]) -> str:
    return digest(run_config)
