"""Deterministic scoring pipeline (Person 1).

Consumes a features.v1 Parquet table produced from a Person 2 graph export,
fits a frozen validation-selected rule and writes contract-valid predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset_audit.split_manifest import (
    read_assignments,
    require_approved_manifest,
)
from deterministic.feature_spec import FEATURE_NAMES
from deterministic.scoring import apply_rule, fit_scoring_rule, save_rule
from evaluation.metrics import binary_metrics, evaluate_predictions
from evaluation.test_lock import request_test_access
from shared.contracts.ids import config_id, experiment_id, split_id as make_split_id
from shared.contracts.validators import utc_now, validate, write_jsonl
from shared.paths import REPO_ROOT, dataset_config, evaluation_config, experiment_config
from shared.tracking import experiment_dir, log_metrics, register_experiment, update_status


def _predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    predicted: np.ndarray,
    *,
    experiment: str,
    split_identity: str,
    config_identity: str,
    build_id: str,
    threshold: float,
) -> list[dict[str, Any]]:
    produced_at = utc_now()
    records = []
    for sample_id, score, label in zip(frame["sample_id"], scores, predicted):
        record = {
            "sample_id": str(sample_id),
            "experiment_id": experiment,
            "split_id": split_identity,
            "graph_build_id": build_id,
            "config_id": config_identity,
            "model_family": "deterministic_scoring",
            "model_version": "features.v1+scoring.v1",
            "score_semantics": "p_ai_authored",
            "predicted_score": float(np.clip(score, 0.0, 1.0)),
            "predicted_class": int(label),
            "threshold": float(threshold),
            "produced_at": produced_at,
        }
        validate("prediction", record)
        records.append(record)
    return records


def run_deterministic_scoring(
    *,
    features_path: str | Path,
    view: str = "lexical",
    manifest_path: str | Path | None = None,
    allow_test: bool = False,
    override_test_lock: bool = False,
    top_k: int = 12,
) -> dict[str, Any]:
    dataset_cfg = dataset_config()
    eval_cfg = evaluation_config()
    exp_cfg = experiment_config()

    manifest = require_approved_manifest(manifest_path, config=dataset_cfg)
    split_of = {
        sample_id: split
        for split, ids in read_assignments(manifest).items()
        for sample_id in ids
    }

    features = pd.read_parquet(features_path)
    if "graph_view" not in features:
        raise ValueError("features table has no graph_view column")
    features = features[features["graph_view"] == view].copy()
    if features.empty:
        raise ValueError(f"no features for view {view!r}")
    builds = features["graph_build_id"].unique().tolist()
    if len(builds) != 1:
        raise ValueError(f"features must come from one graph build, found {builds}")
    build_id = str(builds[0])

    features["split"] = features["sample_id"].map(split_of)
    if features["split"].isna().any():
        raise ValueError("features contain samples that are not in the approved manifest")

    missing = [name for name in FEATURE_NAMES if name not in features.columns]
    if missing:
        raise ValueError(f"features table is missing {len(missing)} features.v1 columns: {missing[:5]}")

    model_config = {
        "model": "deterministic_scoring",
        "view": view,
        "feature_set_version": "features.v1",
        "top_k": top_k,
        "threshold_split": "validation",
        "seed": int(exp_cfg["experiment"]["primary_seed"]),
    }
    cfg_id = config_id(model_config)
    split_identity = make_split_id(manifest["manifest_id"])
    exp_id = experiment_id(
        config=model_config, split_id_value=split_identity, graph_build_id_value=build_id
    )
    register_experiment(
        experiment_id=exp_id,
        config_id=cfg_id,
        run_config=model_config,
        dataset_version=manifest["dataset_version"],
        split_id=split_identity,
        graph_build_id=build_id,
        graph_spec_version=str(features["graph_spec_version"].iloc[0]),
        model_family="deterministic_scoring",
        seeds=[model_config["seed"]],
        notes="Deterministic graph features with a frozen validation-selected scoring rule.",
    )
    update_status(exp_id, "running")
    run_dir = experiment_dir(exp_id)

    train = features[features["split"] == "train"]
    validation = features[features["split"] == "validation"]
    test = features[features["split"] == "test"]
    if train.empty or validation.empty:
        raise ValueError("train and validation splits must be non-empty")

    rule = fit_scoring_rule(
        train[list(FEATURE_NAMES)],
        train["label"].to_numpy(),
        validation[list(FEATURE_NAMES)],
        validation["label"].to_numpy(),
        top_k=top_k,
        config_id=cfg_id,
    )
    save_rule(rule, run_dir / "scoring_rule.json")
    save_rule(rule, REPO_ROOT / "experiments" / "configs" / f"frozen_thresholds_{view}.json")

    labels = features[["sample_id", "label", "source_dataset", "source_task", "text_type"]]
    strata = [column for column in eval_cfg["stratification"]["always_report"] if column != "label"]
    predictions_dir = REPO_ROOT / exp_cfg["experiment"]["prediction_dir"] / exp_id
    predictions_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "experiment_id": exp_id,
        "config_id": cfg_id,
        "split_id": split_identity,
        "graph_build_id": build_id,
        "view": view,
        "rule": {
            "features": [feature["name"] for feature in rule["features"]],
            "threshold": rule["threshold"],
            "validation_metrics": rule["validation_metrics"],
            "rule_hash": rule["rule_hash"],
        },
        "splits": {},
    }

    for split_name, split_frame in (("validation", validation), ("test", test), ("train", train)):
        if split_frame.empty:
            continue
        if split_name == "test":
            if not allow_test:
                summary["splits"]["test"] = {
                    "status": "skipped",
                    "reason": "test split requires --allow-test and is recorded in the test lock",
                }
                continue
            request_test_access(
                experiment_id=exp_id,
                purpose=f"deterministic scoring evaluation ({view})",
                override=override_test_lock,
            )
        scores, predicted = apply_rule(split_frame[list(FEATURE_NAMES)], rule)
        records = _predictions(
            split_frame,
            scores,
            predicted,
            experiment=exp_id,
            split_identity=split_identity,
            config_identity=cfg_id,
            build_id=build_id,
            threshold=rule["threshold"],
        )
        file_path = predictions_dir / f"{split_name}.jsonl"
        write_jsonl(file_path, records)
        summary["splits"][split_name] = evaluate_predictions(
            pd.DataFrame(records),
            labels,
            strata=strata,
            min_cell_size=int(eval_cfg["stratification"]["minimum_cell_size"]),
            bootstrap=split_name == "test",
            resamples=int(eval_cfg["uncertainty"]["resamples"]),
            seed=int(eval_cfg["uncertainty"]["seed"]),
        )
        summary["splits"][split_name]["predictions_file"] = str(file_path.relative_to(REPO_ROOT))

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log_metrics(exp_id, {"summary": summary["splits"]})
    update_status(exp_id, "completed")
    return summary
