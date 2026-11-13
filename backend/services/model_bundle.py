"""Model-bundle loading for the online pipeline.

A bundle is a directory holding a frozen scoring rule and, optionally, a
train-only reference index. Missing or invalid pieces are reported explicitly;
they are never replaced by a fabricated score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.contracts.versions import FEATURE_SET_VERSION, GRAPH_SPEC_VERSION
from shared.paths import REPO_ROOT


@dataclass
class ModelBundle:
    root: Path
    rule: dict[str, Any] | None = None
    reference_index: Any | None = None
    model_version: str = "unavailable"
    graph_spec_version: str = GRAPH_SPEC_VERSION
    feature_set_version: str = FEATURE_SET_VERSION
    reason: str = ""
    view: str = "lexical"
    loaded: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.rule is not None

    @property
    def identifier(self) -> str:
        return self.model_version


def bundle_root() -> Path:
    configured = os.environ.get("MVGR_API_MODEL_BUNDLE", "experiments/checkpoints/current")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def load_bundle(root: str | Path | None = None) -> ModelBundle:
    path = Path(root) if root else bundle_root()
    bundle = ModelBundle(root=path)

    rule_path = path / "scoring_rule.json"
    if not rule_path.exists():
        bundle.reason = (
            f"No scoring rule at {rule_path}. Train and freeze a deterministic rule "
            "before enabling online classification."
        )
    else:
        try:
            from deterministic.scoring import load_rule

            bundle.rule = load_rule(rule_path)
            bundle.model_version = (
                f"deterministic_scoring.{bundle.rule.get('rule_hash', 'unknown')[:12]}"
            )
            bundle.feature_set_version = str(
                bundle.rule.get("feature_set_version", FEATURE_SET_VERSION)
            )
            bundle.view = str(bundle.rule.get("view", "lexical"))
            bundle.loaded.append("scoring_rule")
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            bundle.reason = f"Scoring rule could not be loaded: {type(exc).__name__}: {exc}"

    index_path = path / "reference_index.npz"
    if index_path.exists():
        try:
            from fingerprinting.index import ReferenceIndex

            bundle.reference_index = ReferenceIndex.load(index_path)
            bundle.loaded.append("reference_index")
        except Exception as exc:  # noqa: BLE001 - reported, never hidden
            prefix = f"{bundle.reason} " if bundle.reason else ""
            bundle.reason = (
                prefix
                + f"Reference index could not be loaded: {type(exc).__name__}: {exc}"
            )
    return bundle


__all__ = ["ModelBundle", "bundle_root", "load_bundle"]
