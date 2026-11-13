"""Online analysis service.

The service assembles a contract-valid ``analysis_report`` from the evidence
that is actually available. Every block carries an explicit status. A missing
model bundle, a missing reference index or an unconfigured watermark provider is
reported as unavailable/unsupported -- never as evidence of human authorship.
"""

from __future__ import annotations

import os
from typing import Any

from backend.services.model_bundle import ModelBundle, load_bundle
from shared.contracts.ids import graph_build_id as make_graph_build_id
from shared.contracts.ids import report_id as make_report_id
from shared.contracts.validators import utc_now, validate_analysis_report
from shared.contracts.versions import (
    FEATURE_SET_VERSION,
    GRAPH_SPEC_VERSION,
    REPORT_SCHEMA_VERSION,
)
from shared.paths import graph_config
from shared.preprocessing.text import normalize_text, sha256_text, word_count

BASE_LIMITATIONS = (
    "Scores describe the training distribution; they are not proof of authorship.",
    "The classifier is not calibrated for arbitrary text or unseen domains.",
    "BiScope paraphrase families cannot be reconstructed from the current dataset, "
    "so pooled paraphrase-family claims are unsupported.",
    "PADBen task semantics are still pending confirmation against the dataset card.",
    "Text similarity is evidence of textual overlap, never of common authorship.",
    "Absence of a watermark is not evidence of human authorship.",
)


def _online_build_id(view: str, view_config: dict[str, Any]) -> str:
    return make_graph_build_id(
        dataset_version="online",
        split_id_value="split_online",
        graph_spec_version=GRAPH_SPEC_VERSION,
        config={"view": view, "config": view_config},
        sample_ids=["<online>"],
    )


def _graph_block(text: str) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """Build one lexical graph and its deterministic features."""
    try:
        from deterministic.feature_extractor import extract_graph_features
        from graph_engine.graph_builder.lexical_graph import build_lexical_graph
    except ImportError as exc:
        reason = f"Graph analysis dependency unavailable: {type(exc).__name__}: {exc}"
        return {"status": "unavailable", "views": {}, "reason": reason}, None, reason

    config = graph_config()
    view = "lexical"
    view_config = dict(config["views"][view])
    view_config["shared_max_nodes"] = int(config["shared"]["max_nodes"])
    build_id = _online_build_id(view, view_config)

    try:
        record = build_lexical_graph(
            sample_id="<online>",
            text=text,
            graph_build_id=build_id,
            config_id="cfg_0000000000000000",
            view_config=view_config,
            graph_spec_version=str(config["graph_spec_version"]),
        )
        features = extract_graph_features(record)
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        reason = f"Graph analysis failed: {type(exc).__name__}: {exc}"
        return {"status": "unavailable", "views": {}, "reason": reason}, None, reason

    numeric = {
        name: (float(value) if value is not None else None) for name, value in features.items()
    }
    block = {
        "status": "available",
        "views": {view: numeric},
        "reason": (
            "Lexical co-occurrence graph and features.v1 deterministic features. "
            "No label, source, task or split information participates."
        ),
    }
    return block, features, None


def _classification_block(
    features: dict[str, Any] | None,
    bundle: ModelBundle,
    graph_reason: str | None,
) -> dict[str, Any]:
    if bundle.rule is None:
        return {
            "status": "unavailable",
            "reason": bundle.reason or "No approved model bundle is available.",
        }
    if features is None:
        return {
            "status": "unavailable",
            "reason": graph_reason
            or "Graph features are unavailable, so the model cannot score this text.",
        }
    try:
        import numpy as np
        import pandas as pd

        from deterministic.scoring import apply_rule

        scores, _ = apply_rule(pd.DataFrame([features]), bundle.rule)
        score = float(np.clip(scores[0], 0.0, 1.0))
        threshold = float(bundle.rule["threshold"])
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        return {
            "status": "unavailable",
            "reason": f"Scoring failed: {type(exc).__name__}: {exc}",
        }
    return {
        "status": "available",
        "label": "ai" if score >= threshold else "human",
        "p_ai": score,
        "confidence": float(max(score, 1.0 - score)),
        "calibrated": False,
        "model_version": bundle.model_version,
        "reason": (
            f"Frozen validation-selected deterministic rule at threshold {threshold:.4f}; "
            "score is uncalibrated and describes this training distribution only."
        ),
    }


def _fingerprint_block(text: str, bundle: ModelBundle) -> dict[str, Any]:
    exact = sha256_text(text)
    if bundle.reference_index is None:
        return {
            "status": "no_reference_index",
            "exact_sha256": exact,
            "candidates": [],
            "reason": (
                "No reference index is configured. Similarity requires a train-only "
                "reference index; indexing test data would leak labels."
            ),
        }
    try:
        candidates = bundle.reference_index.query(text, top_k=5, threshold=0.5)
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        return {
            "status": "unavailable",
            "exact_sha256": exact,
            "candidates": [],
            "reason": f"Reference lookup failed: {type(exc).__name__}: {exc}",
        }
    return {
        "status": "available",
        "exact_sha256": exact,
        "candidates": [
            {
                "reference_id": str(item["reference_id"]),
                "method": (
                    "exact"
                    if item.get("normalized_exact")
                    else "shingle_jaccard"
                    if item.get("method") == "shingle_jaccard"
                    else "minhash"
                ),
                "score": float(item["score"]),
            }
            for item in candidates
        ],
        "reason": "Similarity to a train-only reference index; not authorship evidence.",
    }


def _watermark_block(text: str) -> dict[str, Any]:
    try:
        from watermark.verifier import verify_text

        provider = os.environ.get("MVGR_WATERMARK_PROVIDER") or None
        key = os.environ.get("MVGR_WATERMARK_KEY") or None
        result = verify_text(text, provider=provider, key=key)
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        return {
            "status": "inconclusive",
            "details": f"Watermark verification failed: {type(exc).__name__}: {exc}",
            "assumptions": ["A failed verifier says nothing about authorship."],
        }
    return result.as_dict()


def _evidence(
    classification: dict[str, Any],
    graph: dict[str, Any],
    fingerprint: dict[str, Any],
    watermark: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if classification["status"] == "available":
        label = classification["label"]
        entries.append(
            {
                "source": "classification",
                "status": "supports_ai" if label == "ai" else "supports_human",
                "statement": (
                    f"Deterministic model score p_ai={classification['p_ai']:.4f} "
                    f"({classification['model_version']}). Uncalibrated."
                ),
                "strength": float(classification["confidence"]),
            }
        )
    else:
        entries.append(
            {
                "source": "classification",
                "status": "unsupported",
                "statement": classification.get("reason", "No classification available."),
            }
        )

    if graph["status"] == "available":
        entries.append(
            {
                "source": "graph",
                "status": "neutral",
                "statement": "Graph structure was extracted as evidence, not as a verdict.",
            }
        )
    else:
        entries.append(
            {
                "source": "graph",
                "status": "unsupported",
                "statement": graph.get("reason", "Graph analysis unavailable."),
            }
        )

    if fingerprint["status"] == "available" and fingerprint["candidates"]:
        best = fingerprint["candidates"][0]
        entries.append(
            {
                "source": "fingerprint",
                "status": "neutral",
                "statement": (
                    f"Closest reference {best['reference_id']!r} at {best['score']:.3f} "
                    f"({best['method']}). Similarity is not authorship proof."
                ),
                "strength": float(best["score"]),
            }
        )
    else:
        entries.append(
            {
                "source": "fingerprint",
                "status": "unsupported",
                "statement": fingerprint.get("reason", "No reference index is available."),
            }
        )

    watermark_status = watermark["status"]
    entries.append(
        {
            "source": "watermark",
            "status": "neutral" if watermark_status == "detected" else "unsupported",
            "statement": (
                f"Watermark status: {watermark_status}. "
                "Absence of detection is not evidence of human authorship."
            ),
        }
    )
    return entries


def analyze(
    text: str,
    *,
    include_graph: bool = True,
    include_fingerprint: bool = True,
    bundle: ModelBundle | None = None,
) -> dict[str, Any]:
    """Produce a contract-valid analysis report for one text."""
    bundle = bundle or load_bundle()
    normalized = normalize_text(text)
    exact = sha256_text(text)

    if include_graph and normalized:
        graph, features, graph_reason = _graph_block(text)
    else:
        graph_reason = (
            "Graph analysis was not requested." if not include_graph else "Text is empty."
        )
        graph = {"status": "unavailable", "views": {}, "reason": graph_reason}
        features = None

    classification = _classification_block(features, bundle, graph_reason)
    fingerprint = (
        _fingerprint_block(text, bundle)
        if include_fingerprint
        else {
            "status": "no_reference_index",
            "exact_sha256": exact,
            "candidates": [],
            "reason": "Fingerprint lookup was not requested.",
        }
    )
    watermark = _watermark_block(text)

    limitations = list(BASE_LIMITATIONS)
    if classification["status"] != "available":
        limitations.append(
            "No trained bundle was available, so this report contains no authorship score."
        )

    report = {
        "report_id": make_report_id(exact, bundle.identifier),
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "input": {
            "characters": len(text),
            "words": word_count(text),
            "text_sha256": exact,
        },
        "classification": classification,
        "graph_analysis": graph,
        "fingerprint": fingerprint,
        "watermark": watermark,
        "evidence": _evidence(classification, graph, fingerprint, watermark),
        "limitations": limitations,
        "versions": {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "graph_spec_version": GRAPH_SPEC_VERSION,
            "model_version": bundle.model_version,
            "feature_set_version": FEATURE_SET_VERSION,
        },
    }
    validate_analysis_report(report)
    return report


def health() -> dict[str, Any]:
    bundle = load_bundle()
    return {
        "status": "ok",
        "model_bundle": str(bundle.root),
        "model_available": bundle.available,
        "loaded": bundle.loaded,
        "reason": bundle.reason,
    }


__all__ = ["BASE_LIMITATIONS", "analyze", "health"]
