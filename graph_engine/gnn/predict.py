"""GNN inference (Person 2).

Predictions are contract artifacts. Empty graphs are assigned the training
positive prior and flagged, rather than silently dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from graph_engine.gnn.dataset import TokenVocabulary, build_dataset
from graph_engine.gnn.model import build_model
from shared.contracts.validators import utc_now, validate, write_jsonl


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(Path(path), map_location="cpu")


def predict_checkpoint(
    *,
    checkpoint_path: str | Path,
    export_path: str | Path,
    sample_ids: set[str],
    output_path: str | Path,
    experiment_id: str,
    split_id: str,
    fallback_prior: float = 0.5,
) -> dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    payload = load_checkpoint(checkpoint_path)
    model_config = payload["model_config"]
    vocabulary = TokenVocabulary(
        token_to_index={str(k): int(v) for k, v in payload["vocabulary"].items()},
        min_count=int(payload["vocabulary_meta"]["min_count"]),
        max_size=int(payload["vocabulary_meta"]["max_size"]),
    )
    graphs, kept, empty = build_dataset(export_path, vocabulary=vocabulary, sample_ids=sample_ids)
    model = build_model(
        architecture=model_config["architecture"],
        vocabulary_size=vocabulary.size,
        embedding_dim=int(model_config.get("embedding_dim", 32)),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    scores: dict[str, float] = {}
    if graphs:
        loader = DataLoader(graphs, batch_size=int(model_config["batch_size"]), shuffle=False)
        with torch.no_grad():
            for batch in loader:
                probabilities = torch.sigmoid(model(batch)).cpu().numpy()
                for sample_id, score in zip(batch.sample_id, probabilities):
                    scores[str(sample_id)] = float(score)
    for sample_id in empty:
        scores[sample_id] = float(fallback_prior)

    produced_at = utc_now()
    records = []
    for sample_id in sorted(sample_ids):
        score = float(np.clip(scores.get(sample_id, fallback_prior), 0.0, 1.0))
        record = {
            "sample_id": str(sample_id),
            "experiment_id": experiment_id,
            "split_id": split_id,
            "graph_build_id": payload["graph_build_id"],
            "config_id": payload["config_id"],
            "model_family": f"gnn_{model_config['architecture']}",
            "model_version": f"gnn_{model_config['architecture']}.v1",
            "score_semantics": "p_ai_authored",
            "predicted_score": score,
            "predicted_class": int(score >= 0.5),
            "threshold": 0.5,
            "produced_at": produced_at,
        }
        validate("prediction", record)
        records.append(record)
    write_jsonl(output_path, records)
    return {
        "predictions": str(output_path),
        "samples": len(records),
        "empty_graphs_fallback": len(empty),
        "checkpoint": str(checkpoint_path),
    }
