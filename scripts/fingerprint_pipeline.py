"""Offline fingerprinting / GLTR / watermark / fusion pipeline.

Runs the four new analysis branches over the 500-sample engineering pilot and
fuses them with the existing graph features. This stage performs **feature
extraction only**:

* it never trains a model,
* it never generates a formal split,
* it never evaluates against the test set,
* it never modifies the graph artifacts, the datasets, Neo4j or any manifest.

All outputs are new and versioned under ``data/processed/fingerprinting/`` with
reports under ``outputs/fingerprint_audit/``.

Subcommands
-----------
    all            stylometric + writeprints + gltr + watermark + fuse + reports
    stylometric    Module A only
    writeprints    Module B only
    gltr           Module C only (reports unavailable when no local model)
    watermark      Module D only
    fuse           Module E only (consumes the branches written above)
    reports        write the audit reports from the artifacts already on disk
    estimate-gltr  estimate GLTR model/time requirements without running it

Usage
-----
    python scripts/fingerprint_pipeline.py all
    python scripts/fingerprint_pipeline.py all --limit 20      # smoke run
    python scripts/fingerprint_pipeline.py estimate-gltr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fingerprinting.stylometry import StylometricExtractor, feature_names as stylometric_names  # noqa: E402
from fingerprinting.writeprints import WriteprintsExtractor  # noqa: E402
from fusion.multiview_feature_fusion import (  # noqa: E402
    BRANCH_ORDER,
    fuse_multiview,
    split_graph_branches,
)
from gltr.local_model import GltrConfig  # noqa: E402
from gltr.pipeline import GltrAnalyzer, estimate_requirements  # noqa: E402
from shared.contracts.validators import utc_now  # noqa: E402
from shared.paths import load_config, resolve  # noqa: E402
from shared.preprocessing.text import normalize_text, split_sentences, tokenize  # noqa: E402
from watermark.topic import TopicWatermarkVerifier  # noqa: E402
from watermark.unigram import UnigramWatermarkVerifier  # noqa: E402

CONFIG_NAME = "fingerprint_config"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_records(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _missing_stats(frame: pd.DataFrame, *, id_column: str = "sample_id") -> dict[str, Any]:
    feature_columns = [column for column in frame.columns if column != id_column]
    if not feature_columns:
        return {"feature_columns": 0, "rows": int(len(frame)), "missing_fraction": {}}
    missing = frame[feature_columns].isna().mean()
    return {
        "feature_columns": len(feature_columns),
        "rows": int(len(frame)),
        "overall_missing_fraction": float(missing.mean()),
        "max_missing_fraction": float(missing.max()),
        "columns_above_30pct_missing": int((missing > 0.30).sum()),
        "missing_fraction": {column: float(value) for column, value in missing.items()},
    }


def _load_pilot(config: dict[str, Any], limit: int | None) -> pd.DataFrame:
    path = resolve(config["input"]["pilot_csv"])
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame


def _parse_corpus(
    texts: Iterable[str], spacy_model: str, disable: list[str]
) -> list[dict[str, Any]]:
    """Tokenize + parse every text once, so both branches share the spaCy pass."""
    import spacy

    try:
        nlp = spacy.load(spacy_model, disable=disable)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"spaCy model {spacy_model!r} is unavailable: {exc}. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc

    corpus: list[dict[str, Any]] = []
    for text in texts:
        normalized = normalize_text(text)
        doc = nlp(normalized)
        corpus.append(
            {
                "text": normalized,
                "tokens": tokenize(normalized),
                "sentences": split_sentences(normalized),
                "doc": doc,
                "pos_tags": [token.pos_ for token in doc if not token.is_space],
            }
        )
    return corpus


# --------------------------------------------------------------------------- #
# Module A — stylometric
# --------------------------------------------------------------------------- #
def run_stylometric(
    config: dict[str, Any],
    frame: pd.DataFrame,
    corpus: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config.get("stylometric", {})
    extractor = StylometricExtractor(config=cfg)
    if corpus is None:
        corpus = _parse_corpus(
            frame[config["input"]["text_column"]].tolist(),
            cfg.get("spacy_model", "en_core_web_sm"),
            list(cfg.get("spacy_disable", ["ner"])),
        )

    rows: list[dict[str, Any]] = []
    for sample_id, parsed in zip(frame[config["input"]["id_column"]], corpus):
        features = extractor.extract(
            parsed["text"],
            tokens=parsed["tokens"],
            sentences=parsed["sentences"],
            doc=parsed["doc"],
        )
        rows.append({"sample_id": sample_id, **features})

    table = pd.DataFrame(rows)
    status = {
        "module": "stylometric",
        "version": config["stylometric_version"],
        "available": True,
        "samples": int(len(table)),
        "features": int(table.shape[1] - 1),
        "spacy_model": cfg.get("spacy_model", "en_core_web_sm"),
        "spacy_available": extractor.spacy_available,
    }
    return table, status


# --------------------------------------------------------------------------- #
# Module B — writeprints
# --------------------------------------------------------------------------- #
def run_writeprints(
    config: dict[str, Any],
    frame: pd.DataFrame,
    corpus: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config.get("writeprints", {})
    stylo_cfg = config.get("stylometric", {})
    if corpus is None:
        corpus = _parse_corpus(
            frame[config["input"]["text_column"]].tolist(),
            stylo_cfg.get("spacy_model", "en_core_web_sm"),
            list(stylo_cfg.get("spacy_disable", ["ner"])),
        )

    extractor = WriteprintsExtractor(config=cfg)
    extractor.fit(
        [parsed["text"] for parsed in corpus],
        [parsed["tokens"] for parsed in corpus],
        [parsed["pos_tags"] for parsed in corpus],
    )

    rows: list[dict[str, Any]] = []
    sparse_rows: list[np.ndarray] = []
    for sample_id, parsed in zip(frame[config["input"]["id_column"]], corpus):
        rows.append(
            {
                "sample_id": sample_id,
                **extractor.transform(parsed["text"], parsed["tokens"], parsed["pos_tags"]),
            }
        )
        sparse_rows.append(extractor.sparse_row(parsed["text"], parsed["tokens"], parsed["pos_tags"]))

    table = pd.DataFrame(rows)
    vocabulary = extractor.vocabulary()

    # Persist the bounded n-gram block as a sparse matrix (CSR .npz) too.
    if cfg.get("save_sparse_npz", True) and sparse_rows:
        from scipy import sparse

        matrix = sparse.csr_matrix(np.vstack(sparse_rows))
        sparse_path = resolve(config["outputs"]["writeprints_sparse"])
        sparse_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(sparse_path, matrix)
        vocabulary["sparse_shape"] = list(matrix.shape)
        vocabulary["sparse_path"] = str(sparse_path.relative_to(resolve(".")))

    status = {
        "module": "writeprints",
        "version": config["writeprints_version"],
        "available": True,
        "samples": int(len(table)),
        "features": int(table.shape[1] - 1),
        "normalization": extractor.normalization,
        "vocab_sizes": {group: len(items) for group, items in vocabulary["groups"].items()},
    }
    return table, {"table": table, "vocabulary": vocabulary, "status": status}


# --------------------------------------------------------------------------- #
# Module C — GLTR
# --------------------------------------------------------------------------- #
def run_gltr(config: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = config.get("gltr", {})
    analyzer = GltrAnalyzer.from_config(cfg)
    available, reason = analyzer.availability()

    columns = ["sample_id", *analyzer.feature_names]
    if not available:
        # Honest placeholder: the module is unavailable and no download is
        # attempted. Every feature cell is empty; fusion treats the branch as
        # unavailable (see run_fuse) so it can never become a fake signal.
        placeholder = pd.DataFrame(
            [{"sample_id": sample_id, **{name: None for name in analyzer.feature_names}}
             for sample_id in frame[config["input"]["id_column"]]],
            columns=columns,
        )
        status = {
            "module": "gltr",
            "version": config["gltr_version"],
            "available": False,
            "reason": reason,
            "model_name": cfg.get("model_name"),
            "allow_download": bool(cfg.get("allow_download", False)),
            "samples": int(len(frame)),
            "features": len(analyzer.feature_names),
        }
        return placeholder, status

    analyzer.load()
    rows: list[dict[str, Any]] = []
    for sample_id, text in zip(
        frame[config["input"]["id_column"]], frame[config["input"]["text_column"]]
    ):
        rows.append({"sample_id": sample_id, **analyzer.extract(str(text))})
    table = pd.DataFrame(rows)
    status = {
        "module": "gltr",
        "version": config["gltr_version"],
        "available": True,
        "model": analyzer.model.describe() if analyzer.model else {},
        "samples": int(len(table)),
        "features": int(table.shape[1] - 1),
    }
    return table, status


# --------------------------------------------------------------------------- #
# Module D — watermark
# --------------------------------------------------------------------------- #
def run_watermark(config: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    import os

    cfg = config.get("watermark", {})
    key_env = cfg.get("key_env", "MVGR_WATERMARK_KEY")
    key = os.environ.get(key_env) or None
    key_configured = key is not None  # NEVER log the key itself

    provider_cfg = {entry["id"]: entry for entry in cfg.get("providers", []) if isinstance(entry, dict)}
    unigram = UnigramWatermarkVerifier.from_config(provider_cfg.get("unigram_greenlist", {}))
    topic = TopicWatermarkVerifier.from_config(provider_cfg.get("topic_conditioned", {}))

    rows: list[dict[str, Any]] = []
    for sample_id, text in zip(
        frame[config["input"]["id_column"]], frame[config["input"]["text_column"]]
    ):
        result = unigram.verify(str(text), key=key)
        topic_result = topic.verify(str(text), key=key)
        evidence = result.evidence or {}
        rows.append(
            {
                "sample_id": sample_id,
                "watermark_status": result.status.value,
                "watermark_provider": result.provider,
                "watermark_detected": (
                    None if result.status.value in ("unsupported", "inconclusive")
                    else result.status.value == "detected"
                ),
                "watermark_confidence": result.confidence,
                "watermark_z_score": evidence.get("z_score"),
                "watermark_green_count": evidence.get("green_list_token_count"),
                "watermark_green_proportion": evidence.get("green_list_proportion"),
                "watermark_expected_proportion": evidence.get("expected_green_proportion"),
                "watermark_p_value": evidence.get("p_value_exact"),
                "watermark_watermarked_token_count": evidence.get("watermarked_token_count"),
                "watermark_topic_status": topic_result.status.value,
                "watermark_details": result.details,
            }
        )
    table = pd.DataFrame(rows)
    status_counts = table["watermark_status"].value_counts().to_dict() if not table.empty else {}
    status = {
        "module": "watermark",
        "version": config["watermark_version"],
        "available": True,
        "key_env": key_env,
        "key_configured": key_configured,
        "providers": [entry.get("id") for entry in cfg.get("providers", []) if isinstance(entry, dict)],
        "implemented_detectors": ["unigram_greenlist"],
        "unsupported_detectors": ["topic_conditioned"],
        "samples": int(len(table)),
        "features": int(table.shape[1] - 1),
        "status_counts": {str(key_): int(value) for key_, value in status_counts.items()},
        "interpretation": cfg.get("interpret_absence_as"),
    }
    return table, status


# --------------------------------------------------------------------------- #
# Module E — fusion
# --------------------------------------------------------------------------- #
def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


def run_fuse(config: dict[str, Any], frame: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    outputs = config["outputs"]
    id_column = config["input"]["id_column"]

    # The fusion config carries the fusion section plus the top-level experiment
    # declarations, so the schema can record which branches each experiment uses.
    fusion_config = dict(config.get("fusion", {}))
    fusion_config["experiments"] = config.get("experiments", {})

    metadata_columns = list(config["input"]["metadata_columns"])
    metadata = frame[[id_column, *metadata_columns]].copy()

    stylometric = _read_if_exists(resolve(outputs["stylometric"]))
    writeprints = _read_if_exists(resolve(outputs["writeprints"]))
    gltr_table = _read_if_exists(resolve(outputs["gltr"]))
    watermark = _read_if_exists(resolve(outputs["watermark"]))

    # Only treat GLTR as a branch when it actually produced values.
    gltr_available = gltr_table is not None and gltr_table.drop(
        columns=[id_column], errors="ignore"
    ).notna().any().any()
    if not gltr_available:
        gltr_table = None

    graph_path = resolve(config["input"]["graph_features"])
    branch_frames: dict[str, pd.DataFrame | None] = {
        "stylometric": stylometric,
        "writeprints": writeprints,
        "gltr": gltr_table,
        "lexical_graph": None,
        "syntactic_graph": None,
    }
    graph_branch_note = "graph_features not found; graph branches unavailable"
    if graph_path.exists():
        graph_features = pd.read_csv(graph_path)
        try:
            graph_branches = split_graph_branches(graph_features)
            branch_frames["lexical_graph"] = graph_branches.get("lexical_graph")
            branch_frames["syntactic_graph"] = graph_branches.get("syntactic_graph")
            graph_branch_note = "graph branches loaded from existing artifacts (read-only)"
        except Exception as exc:  # noqa: BLE001
            graph_branch_note = f"graph branch preparation failed: {exc}"

    result = fuse_multiview(
        branch_frames=branch_frames,
        metadata=metadata,
        watermark_frame=watermark,
        config=fusion_config,
    )
    result.report["graph_branch_note"] = graph_branch_note
    result.report.setdefault("available", True)
    return result, result.report


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
def _report_payload(
    config: dict[str, Any],
    statuses: dict[str, Any],
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    elapsed: float,
    peak_bytes: int | None,
) -> dict[str, Any]:
    return {
        "generated_at": _stamp(),
        "fingerprint_version": config["fingerprint_version"],
        "pilot": {
            "csv": config["input"]["pilot_csv"],
            "samples": int(len(frame)),
        },
        "modules": statuses,
        "artifacts": artifacts,
        "execution": {
            "seconds": round(elapsed, 3),
            "peak_memory_bytes": peak_bytes,
            "peak_memory_mb": (round(peak_bytes / (1024 * 1024), 2) if peak_bytes else None),
        },
        "scope": {
            "feature_extraction_only": True,
            "trained_models": False,
            "formal_split_generated": False,
            "test_set_evaluated": False,
            "neo4j_modified": False,
            "existing_artifacts_overwritten": False,
        },
    }


def _write_reports(
    config: dict[str, Any],
    payload: dict[str, Any],
    comparison: dict[str, Any],
    offline: dict[str, Any],
) -> None:
    outputs = config["outputs"]
    report_dir = resolve(outputs["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(resolve(outputs["report_json"]), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")

    lines: list[str] = [
        "# Fingerprint extraction report",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Feature set version: `{payload['fingerprint_version']}`",
        f"- Pilot: `{payload['pilot']['csv']}` ({payload['pilot']['samples']} samples)",
        "- Scope: feature extraction only; no training, no formal split, no test evaluation.",
        "",
        "## Modules",
        "",
        "| Module | Available | Samples | Features | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for name, status in payload["modules"].items():
        notes = status.get("reason") or ""
        if not notes:
            normalization = status.get("normalization")
            if isinstance(normalization, dict):
                notes = f"normalization={normalization.get('method', 'none')}"
            elif normalization:
                notes = str(normalization)
        lines.append(
            f"| {name} | {status.get('available')} | {status.get('samples', '')} | "
            f"{status.get('features', '')} | {notes} |"
        )
    lines += [
        "",
        "## Execution",
        "",
        f"- Elapsed: {payload['execution']['seconds']} s",
        f"- Peak memory: {payload['execution']['peak_memory_mb']} MB",
        "",
        "## Missing-value statistics",
        "",
    ]
    for name, stats in comparison.get("missing", {}).items():
        lines.append(
            f"- **{name}**: {stats['feature_columns']} features, "
            f"overall missing {stats['overall_missing_fraction']:.3f}, "
            f"max column missing {stats['max_missing_fraction']:.3f}"
        )
    lines += ["", "## Limitations", ""]
    for note in comparison.get("limitations", []):
        lines.append(f"- {note}")
    lines.append("")

    with open(resolve(outputs["report_md"]), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    # feature comparison report
    comp_lines: list[str] = [
        "# Feature comparison report",
        "",
        f"- Generated: {payload['generated_at']}",
        "",
        "| Branch | Features | Rows | Overall missing | Max column missing |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in comparison.get("branches", {}).items():
        comp_lines.append(
            f"| {name} | {stats['feature_columns']} | {stats['rows']} | "
            f"{stats['overall_missing_fraction']:.3f} | {stats['max_missing_fraction']:.3f} |"
        )
    comp_lines += ["", "## Fused feature selection", ""]
    selection = comparison.get("fusion_selection", {})
    comp_lines.append(f"- Candidate features: {selection.get('candidate_features', '')}")
    comp_lines.append(f"- Selected features: {selection.get('selected_features', '')}")
    comp_lines.append(f"- Dropped (missing): {len(selection.get('dropped_missing', []))}")
    comp_lines.append(f"- Dropped (constant): {len(selection.get('dropped_constant', []))}")
    comp_lines.append(f"- Dropped (correlated): {len(selection.get('dropped_correlated', []))}")
    comp_lines += ["", "## Experiments (declarations only, nothing trained)", ""]
    comp_lines.append("| Experiment | Branches | Feature columns |")
    comp_lines.append("|---|---|---:|")
    for key, definition in config.get("experiments", {}).items():
        comp_lines.append(
            f"| {key} | {', '.join(definition.get('branches', []))} | "
            f"{len(comparison.get('experiments', {}).get(key, []))} |"
        )
    comp_lines.append("")
    with open(resolve(outputs["comparison_md"]), "w", encoding="utf-8") as handle:
        handle.write("\n".join(comp_lines))

    # offline execution report
    off_lines: list[str] = [
        "# Offline execution report",
        "",
        f"- Generated: {payload['generated_at']}",
        "",
        "| Check | Value |",
        "|---|---|",
    ]
    for key, value in offline.items():
        off_lines.append(f"| {key} | {value} |")
    off_lines += [
        "",
        "## Guarantees",
        "",
        "- No API calls, no cloud services, no telemetry.",
        "- GLTR never downloads a model: `allow_download` defaults to false and the "
        "module reports itself unavailable when no local model is cached.",
        "- The watermark key is read from an environment variable and is never "
        "written to disk or printed.",
        "- spaCy and the Hugging Face cache are used locally only.",
        "",
    ]
    with open(resolve(outputs["offline_md"]), "w", encoding="utf-8") as handle:
        handle.write("\n".join(off_lines))


# --------------------------------------------------------------------------- #
# subcommand drivers
# --------------------------------------------------------------------------- #
def cmd_stylometric(
    config: dict[str, Any],
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    corpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    table, status = run_stylometric(config, frame, corpus=corpus)
    _to_records(table, resolve(config["outputs"]["stylometric"]))
    artifacts["stylometric_features.csv"] = int(len(table))
    return status


def cmd_writeprints(
    config: dict[str, Any],
    frame: pd.DataFrame,
    artifacts: dict[str, Any],
    corpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    table, payload = run_writeprints(config, frame, corpus=corpus)
    _to_records(table, resolve(config["outputs"]["writeprints"]))
    vocab_path = resolve(config["outputs"]["writeprints_vocab"])
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(payload["vocabulary"], handle, indent=2)
        handle.write("\n")
    artifacts["writeprints_features.csv"] = int(len(table))
    return payload["status"]


def cmd_gltr(config: dict[str, Any], frame: pd.DataFrame, artifacts: dict[str, Any]) -> dict[str, Any]:
    table, status = run_gltr(config, frame)
    _to_records(table, resolve(config["outputs"]["gltr"]))
    artifacts["gltr_features.csv"] = int(len(table))
    return status


def cmd_watermark(config: dict[str, Any], frame: pd.DataFrame, artifacts: dict[str, Any]) -> dict[str, Any]:
    table, status = run_watermark(config, frame)
    _to_records(table, resolve(config["outputs"]["watermark"]))
    artifacts["watermark_results.csv"] = int(len(table))
    return status


def cmd_fuse(config: dict[str, Any], frame: pd.DataFrame, artifacts: dict[str, Any]) -> dict[str, Any]:
    result, report = run_fuse(config, frame)
    _to_records(result.features, resolve(config["outputs"]["fused"]))
    _to_records(result.metadata, resolve(config["outputs"]["fused_metadata"]))
    schema_path = resolve(config["outputs"]["schema"])
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as handle:
        json.dump(result.schema, handle, indent=2)
        handle.write("\n")
    artifacts["fused_multiview_features.csv"] = int(len(result.features))
    artifacts["fused_multiview_metadata.csv"] = int(len(result.metadata))
    artifacts["fingerprint_feature_schema.json"] = 1
    return report


def _collect_comparison(config: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    outputs = config["outputs"]
    id_column = config["input"]["id_column"]
    branches: dict[str, Any] = {}
    missing: dict[str, Any] = {}
    for name, key in (
        ("stylometric", "stylometric"),
        ("writeprints", "writeprints"),
        ("gltr", "gltr"),
        ("watermark", "watermark"),
        ("fused", "fused"),
    ):
        table = _read_if_exists(resolve(outputs[key]))
        if table is None:
            continue
        stats = _missing_stats(table, id_column=id_column)
        branches[name] = stats
        missing[name] = stats

    fused_schema = None
    schema_path = resolve(outputs["schema"])
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as handle:
            fused_schema = json.load(handle)

    comparison = {
        "branches": branches,
        "missing": missing,
        "fusion_selection": (fused_schema or {}).get("selection", {}),
        "experiments": (fused_schema or {}).get("experiments", {}),
        "limitations": [
            "Stylometric and writeprints features describe writing style; they are "
            "statistical signals, not proof of authorship.",
            "Writeprints vocabulary is fit on the pilot corpus for feature extraction "
            "only; a real experiment must fit it on the train split.",
            "GLTR-style predictability is unavailable without a local causal LM and is "
            "never downloaded automatically.",
            "Watermark verification is only meaningful for the exact scheme and key "
            "used at generation time; absence of detection is not evidence of human "
            "authorship.",
            "The pilot is an engineering sample; no formal split exists and no model "
            "was trained or evaluated.",
        ],
    }
    return comparison


def _collect_offline(config: dict[str, Any], statuses: dict[str, Any]) -> dict[str, Any]:
    import importlib.util

    gltr_status = statuses.get("gltr", {})
    return {
        "transformers installed": importlib.util.find_spec("transformers") is not None,
        "torch installed": importlib.util.find_spec("torch") is not None,
        "spacy installed": importlib.util.find_spec("spacy") is not None,
        "gltr available": gltr_status.get("available"),
        "gltr reason": gltr_status.get("reason", ""),
        "gltr allow_download": gltr_status.get("allow_download", False),
        "watermark key configured": statuses.get("watermark", {}).get("key_configured"),
        "network required": False,
        "auto model download": False,
    }


def cmd_all(config: dict[str, Any], frame: pd.DataFrame, limit: int | None) -> int:
    tracemalloc.start()
    started = time.perf_counter()
    artifacts: dict[str, Any] = {}
    statuses: dict[str, Any] = {}

    # Parse the corpus once and share the spaCy pass across both branches.
    stylo_cfg = config.get("stylometric", {})
    corpus = _parse_corpus(
        frame[config["input"]["text_column"]].tolist(),
        stylo_cfg.get("spacy_model", "en_core_web_sm"),
        list(stylo_cfg.get("spacy_disable", ["ner"])),
    )

    statuses["stylometric"] = cmd_stylometric(config, frame, artifacts, corpus=corpus)
    statuses["writeprints"] = cmd_writeprints(config, frame, artifacts, corpus=corpus)
    statuses["gltr"] = cmd_gltr(config, frame, artifacts)
    statuses["watermark"] = cmd_watermark(config, frame, artifacts)
    statuses["fusion"] = cmd_fuse(config, frame, artifacts)

    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    comparison = _collect_comparison(config, frame)
    offline = _collect_offline(config, statuses)
    payload = _report_payload(config, statuses, frame, artifacts, elapsed, peak)
    payload["fusion"] = statuses["fusion"]
    _write_reports(config, payload, comparison, offline)

    print(json.dumps(
        {
            "samples": int(len(frame)),
            "modules": {name: status.get("available") for name, status in statuses.items()},
            "features": {name: status.get("features") for name, status in statuses.items() if "features" in status},
            "fused_features": statuses["fusion"].get("features"),
            "seconds": round(elapsed, 2),
            "peak_memory_mb": round(peak / (1024 * 1024), 2) if peak else None,
            "gltr_reason": statuses["gltr"].get("reason"),
        },
        indent=2,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "command",
        choices=[
            "all", "stylometric", "writeprints", "gltr", "watermark",
            "fuse", "reports", "estimate-gltr",
        ],
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N pilot rows")
    parser.add_argument("--config", default=CONFIG_NAME)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    frame = _load_pilot(config, args.limit)

    if args.command == "estimate-gltr":
        analyzer = GltrAnalyzer.from_config(config.get("gltr", {}))
        estimate = estimate_requirements(
            frame[config["input"]["text_column"]].astype(str).tolist(),
            GltrConfig.from_config(config.get("gltr", {})),
        )
        print(json.dumps(estimate, indent=2))
        return 0

    if args.command == "all":
        return cmd_all(config, frame, args.limit)

    artifacts: dict[str, Any] = {}
    statuses: dict[str, Any] = {}
    if args.command == "stylometric":
        statuses["stylometric"] = cmd_stylometric(config, frame, artifacts)
    elif args.command == "writeprints":
        statuses["writeprints"] = cmd_writeprints(config, frame, artifacts)
    elif args.command == "gltr":
        statuses["gltr"] = cmd_gltr(config, frame, artifacts)
    elif args.command == "watermark":
        statuses["watermark"] = cmd_watermark(config, frame, artifacts)
    elif args.command == "fuse":
        statuses["fusion"] = cmd_fuse(config, frame, artifacts)
    elif args.command == "reports":
        # Reuse the module statuses / artifacts recorded by the last `all` run so
        # the reports can be regenerated without re-running the extraction.
        existing: dict[str, Any] = {}
        report_path = resolve(config["outputs"]["report_json"])
        if report_path.exists():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
        statuses = existing.get("modules", {})
        artifacts = existing.get("artifacts", {})
        comparison = _collect_comparison(config, frame)
        offline = _collect_offline(config, statuses)
        payload = _report_payload(config, statuses, frame, artifacts, 0.0, None)
        if existing.get("execution"):
            payload["execution"] = existing["execution"]
        payload["fusion"] = statuses.get("fusion")
        _write_reports(config, payload, comparison, offline)

    if statuses:
        print(json.dumps({"command": args.command, "statuses": statuses}, indent=2, default=str))
    else:
        print(json.dumps({"command": args.command, "artifacts": artifacts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
