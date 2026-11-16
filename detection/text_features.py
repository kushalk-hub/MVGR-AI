"""Extract the deterministic feature vector from raw text.

This is the bridge between the raw document and :class:`DeterministicDetector`.
It reuses the *existing* fingerprinting implementation — the same stylometric
extractor, the same writeprints extractor, the same GPT-2 based GLTR analyzer
and the same watermark verifiers — so a prediction from raw text sees exactly
the features the model was trained on.

Only the columns the model actually asks for are kept: the *training* feature
list is authoritative. Feature selection is **not** re-run here, because
selecting on a single document would silently change the feature space.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from detection.dataset_loader import ID_COLUMN, TEXT_COLUMN
from detection.feature_alignment import FAMILY_PREFIXES


class TextFeatureError(RuntimeError):
    """Raised when raw text cannot be turned into a feature vector."""


def _branch_source(feature_name: str) -> tuple[str, str]:
    """Map a namespaced feature name back to (family, source column)."""
    for family, prefix in FAMILY_PREFIXES.items():
        if feature_name.startswith(prefix):
            return family, feature_name[len(prefix) :]
    raise TextFeatureError(f"unknown feature family for column {feature_name!r}")


def _placeholders(config: dict[str, Any]) -> dict[str, str]:
    """Neutral metadata values for a single synthetic row.

    Metadata never reaches the feature matrix, so these values only have to be
    present, not meaningful.
    """
    columns = list(config["input"].get("metadata_columns", []))
    placeholders = {column: "" for column in columns}
    placeholders["label"] = "0"
    placeholders["source_dataset"] = "unknown"
    placeholders["source_task"] = "unknown"
    placeholders["text_type"] = "original"
    placeholders["graph_category"] = "original"
    placeholders["domain"] = "unknown"
    placeholders["ai_model"] = "unknown"
    placeholders[ID_COLUMN] = "inference:0"
    return placeholders


def extract_features_for_text(
    text: str,
    *,
    config_name: str = "fingerprint_config_10k",
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract the feature values for one document.

    ``feature_names`` selects which columns to return; when it is ``None`` the
    union of every branch's numeric columns is returned.
    """
    import sys

    from shared.paths import load_config

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from fingerprint_pipeline import (  # type: ignore[import-not-found]
        _parse_corpus,
        run_gltr,
        run_stylometric,
        run_watermark,
        run_writeprints,
    )

    config = copy.deepcopy(load_config(config_name))
    # Never let a one-off inference overwrite the long-run checkpoint.
    checkpoint = config.setdefault("gltr", {}).setdefault("checkpoint", {})
    checkpoint["enabled"] = True
    checkpoint["path"] = str(Path(tempfile.gettempdir()) / "mvgr_gltr_inference_checkpoint.csv")

    frame = pd.DataFrame([{**_placeholders(config), TEXT_COLUMN: str(text)}])

    corpus = _parse_corpus(
        [str(text)],
        config.get("stylometric", {}).get("spacy_model", "en_core_web_sm"),
        list(config.get("stylometric", {}).get("spacy_disable", ["ner"])),
    )
    stylometric_table, _ = run_stylometric(config, frame, corpus=corpus)
    writeprints_table, _ = run_writeprints(config, frame, corpus=corpus)
    gltr_table, _ = run_gltr(config, frame, resume=False)
    watermark_table, _ = run_watermark(config, frame)

    tables = {
        "stylometric": stylometric_table,
        "writeprints": writeprints_table,
        "gltr": gltr_table,
        "watermark": watermark_table,
    }

    values: dict[str, float] = {}
    if feature_names is None:
        for family, table in tables.items():
            if table is None or table.empty:
                continue
            prefix = FAMILY_PREFIXES.get(family, f"{family}__")
            for column in table.columns:
                if column == ID_COLUMN:
                    continue
                if not pd.api.types.is_numeric_dtype(table[column]):
                    continue
                values[f"{prefix}{column}"] = float(table[column].iloc[0])
        return values

    for feature_name in feature_names:
        family, source_column = _branch_source(feature_name)
        table = tables.get(family)
        if table is None or table.empty or source_column not in table.columns:
            values[feature_name] = float("nan")
            continue
        raw = table[source_column].iloc[0]
        values[feature_name] = float(raw) if pd.notna(raw) else float("nan")
    return values


__all__ = ["TextFeatureError", "extract_features_for_text"]
