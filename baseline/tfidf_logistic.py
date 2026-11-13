"""Text-only baseline: TF-IDF + logistic regression (Person 1).

The baseline exists to answer one question: how much of the result is explained
by surface lexical content alone? It uses the approved split manifest, fits the
vectorizer on the training split only, and never touches the test split without
recording a test-lock access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from dataset_audit.split_manifest import (
    SPLITS,
    read_assignments,
    require_approved_manifest,
)
from evaluation.metrics import binary_metrics, evaluate_predictions
from evaluation.test_lock import request_test_access
from shared.contracts.ids import config_id, experiment_id, split_id as make_split_id
from shared.contracts.validators import utc_now, validate, write_jsonl
from shared.paths import REPO_ROOT, dataset_config, evaluation_config, experiment_config
from shared.tracking import (
    experiment_dir,
    log_metrics,
    register_experiment,
    update_status,
)

FORBIDDEN_INPUT_COLUMNS = {"search_text", "label"}


def _prediction_records(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    experiment: str,
    split_identity: str,
    config_identity: str,
) -> list[dict[str, Any]]:
    produced_at = utc_now()
    records: list[dict[str, Any]] = []
    for sample_id, score in zip(frame["sample_id"], scores):
        record = {
            "sample_id": str(sample_id),
            "experiment_id": experiment,
            "split_id": split_identity,
            "graph_build_id": "none",
            "config_id": config_identity,
            "model_family": "tfidf_logistic",
            "model_version": "tfidf_logistic.v1",
            "score_semantics": "p_ai_authored",
            "predicted_score": float(np.clip(score, 0.0, 1.0)),
            "predicted_class": int(score >= 0.5),
            "threshold": 0.5,
            "produced_at": produced_at,
        }
        validate("prediction", record)
        records.append(record)
    return records


def run_tfidf_logistic(
    *,
    manifest_path: str | Path | None = None,
    allow_test: bool = False,
    override_test_lock: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset_cfg = dataset_config()
    eval_cfg = evaluation_config()
    exp_cfg = experiment_config()
    model_cfg = exp_cfg["models"]["tfidf_logistic"]

    manifest = require_approved_manifest(manifest_path, config=dataset_cfg)
    assignments = read_assignments(manifest)
    split_of = {
        sample_id: split for split, ids in assignments.items() for sample_id in ids
    }

    path = Path(output_root) if output_root else Path(dataset_cfg["dataset"]["canonical"]["parquet"])
    frame = pd.read_parquet(
        path,
        columns=["sample_id", "text", "label", "source_dataset", "source_task", "text_type"],
    )
    frame = frame[frame["sample_id"].isin(split_of)].copy()
    frame["split"] = frame["sample_id"].map(split_of)
    if len(frame) != len(split_of):
        raise RuntimeError(
            f"manifest references {len(split_of)} samples but only {len(frame)} were found in the dataset"
        )

    resolved_config = {
        "model": "tfidf_logistic",
        "as_of": "2026-11-13",
        "vectorizer": model_cfg["vectorizer"],
        "classifier": model_cfg["classifier"],
        "seed": int(model_cfg.get("seed", exp_cfg["experiment"]["primary_seed"])),
        "stratification": eval_cfg["stratification"]["always_report"],
    }
    cfg_id = config_id(resolved_config)
    split_identity = make_split_id(manifest["manifest_id"])
    exp_id = experiment_id(
        config=resolved_config, split_id_value=split_identity, graph_build_id_value="none"
    )

    register_experiment(
        experiment_id=exp_id,
        config_id=cfg_id,
        run_config=resolved_config,
        dataset_version=manifest["dataset_version"],
        split_id=split_identity,
        graph_build_id="none",
        graph_spec_version="none",
        model_family="tfidf_logistic",
        seeds=[resolved_config["seed"]],
        notes="Text-only baseline on the approved split manifest.",
    )
    update_status(exp_id, "running")
    run_dir = experiment_dir(exp_id)

    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    if train.empty or validation.empty:
        raise RuntimeError("train and validation splits must be non-empty")

    vectorizer = TfidfVectorizer(**model_cfg["vectorizer"])
    classifier = LogisticRegression(**model_cfg["classifier"])

    x_train = vectorizer.fit_transform(train["text"])
    classifier.fit(x_train, train["label"])

    predictions_dir = Path(exp_cfg["experiment"]["prediction_dir"]) / exp_id
    if not predictions_dir.is_absolute():
        predictions_dir = REPO_ROOT / predictions_dir
    predictions_dir.mkdir(parents=True, exist_ok=True)
    evaluation_labels = frame[
        ["sample_id", "label", "source_dataset", "source_task", "text_type"]
    ]
    strata = [
        column
        for column in eval_cfg["stratification"]["always_report"]
        if column != "label"
    ]

    summary: dict[str, Any] = {
        "experiment_id": exp_id,
        "config_id": cfg_id,
        "split_id": split_identity,
        "manifest_id": manifest["manifest_id"],
        "dataset_version": manifest["dataset_version"],
        "counts": {split: int((frame["split"] == split).sum()) for split in SPLITS},
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "splits": {},
    }

    for split_name, split_frame in (("train", train), ("validation", validation), ("test", test)):
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
                purpose="tfidf_logistic baseline evaluation",
                override=override_test_lock,
            )
        scores = classifier.predict_proba(vectorizer.transform(split_frame["text"]))[:, 1]
        records = _prediction_records(
            split_frame,
            scores,
            experiment=exp_id,
            split_identity=split_identity,
            config_identity=cfg_id,
        )
        predictions_file = predictions_dir / f"{split_name}.jsonl"
        write_jsonl(predictions_file, records)

        predictions_frame = pd.DataFrame(records)
        metrics = evaluate_predictions(
            predictions_frame,
            evaluation_labels,
            strata=strata,
            min_cell_size=int(eval_cfg["stratification"]["minimum_cell_size"]),
            bootstrap=split_name == "test",
            resamples=int(eval_cfg["uncertainty"]["resamples"]),
            seed=int(eval_cfg["uncertainty"]["seed"]),
        )
        metrics["predictions_file"] = str(predictions_file.relative_to(REPO_ROOT))
        summary["splits"][split_name] = metrics

    # Length-only control: how much signal is just text length?
    length_scores = frame["text"].str.len().astype(float)
    length_scores = (length_scores - length_scores.min()) / max(
        float(length_scores.max() - length_scores.min()), 1.0
    )
    summary["controls"] = {
        "length_only": binary_metrics(frame["label"], length_scores.to_numpy()),
        "majority_class": {
            "label": int(frame["label"].mode().iloc[0]),
            "share": float(frame["label"].value_counts(normalize=True).iloc[0]),
        },
    }

    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier},
        run_dir / "model.joblib",
    )
    coefficients = classifier.coef_.ravel()
    names = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(coefficients)
    summary["top_features"] = {
        "toward_human": [
            {"feature": str(names[index]), "weight": float(coefficients[index])}
            for index in order[:25]
        ],
        "toward_ai": [
            {"feature": str(names[index]), "weight": float(coefficients[index])}
            for index in order[-25:][::-1]
        ],
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    log_metrics(exp_id, {"summary": summary["splits"]})
    update_status(exp_id, "completed")
    return summary
