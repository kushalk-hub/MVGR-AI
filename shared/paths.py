"""Repository paths and config loading.

All scripts resolve paths through this module so that no module hard-codes a
working directory. Paths in YAML are repository-relative unless absolute.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(ValueError):
    pass


def resolve(path_like: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base or REPO_ROOT) / path


@lru_cache(maxsize=16)
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` (``name`` may include the .yaml suffix)."""
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = CONFIG_DIR / filename
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Config {path} must contain a YAML mapping")
    return payload


def dataset_config() -> dict[str, Any]:
    return load_config("dataset_config")


def graph_config() -> dict[str, Any]:
    return load_config("graph_config")


def evaluation_config() -> dict[str, Any]:
    return load_config("evaluation_config")


def experiment_config() -> dict[str, Any]:
    return load_config("experiment_config")


def canonical_dataset_path(config: dict[str, Any] | None = None) -> Path:
    config = config or dataset_config()
    return resolve(config["dataset"]["canonical"]["parquet"])


def source_dataset_path(config: dict[str, Any] | None = None) -> Path:
    """Immutable upstream parquet that the canonical copy must match byte-for-byte."""
    config = config or dataset_config()
    dataset = config["dataset"]
    source = dataset.get("source_parquet")
    if not source:
        raise ConfigError(
            "dataset.source_parquet is not configured; provenance cannot be verified."
        )
    return resolve(source)


def manifest_root(config: dict[str, Any] | None = None) -> Path:
    config = config or dataset_config()
    return resolve(config["paths"]["manifest_root"])


def derived_root(config: dict[str, Any] | None = None) -> Path:
    config = config or dataset_config()
    return resolve(config["paths"]["derived_root"])


def ensure_repo_dirs() -> None:
    for relative in (
        "data/derived",
        "data/manifests/candidate",
        "experiments/configs",
        "experiments/predictions",
        "experiments/checkpoints",
        "experiments/logs",
        "experiments/runs",
        "reports/generated",
    ):
        (REPO_ROOT / relative).mkdir(parents=True, exist_ok=True)
