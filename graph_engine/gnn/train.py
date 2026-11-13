"""GNN training loop (Person 2).

Training never evaluates the test split. Early stopping and checkpoint
selection use the validation split only. A checkpoint is bound to its
experiment, config, split and graph build identities.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from graph_engine.gnn.dataset import (
    TokenVocabulary,
    build_dataset,
    load_records,
)
from graph_engine.gnn.model import build_model
from shared.contracts.ids import config_id as make_config_id
from shared.contracts.ids import experiment_id as make_experiment_id
from shared.contracts.validators import utc_now, write_json
from shared.tracking import experiment_dir, log_metrics, register_experiment, update_status


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _evaluate(model, loader, device, criterion) -> dict[str, float]:
    import torch
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            losses.append(float(loss.item()))
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch.y.cpu().numpy().astype(int).tolist())
    predictions = [1 if score >= 0.5 else 0 for score in scores]
    result = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
    if len(set(labels)) > 1:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    return result


def run_gnn_training(
    *,
    export_path: str | Path,
    labels_frame,
    train_ids: set[str],
    validation_ids: set[str],
    graph_build_id: str,
    graph_spec_version: str,
    split_id: str,
    dataset_version: str,
    model_config: dict[str, Any],
    seeds: list[int],
    view: str,
) -> dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    cfg_id = make_config_id(model_config)
    exp_id = make_experiment_id(
        config=model_config, split_id_value=split_id, graph_build_id_value=graph_build_id
    )
    register_experiment(
        experiment_id=exp_id,
        config_id=cfg_id,
        run_config=model_config,
        dataset_version=dataset_version,
        split_id=split_id,
        graph_build_id=graph_build_id,
        graph_spec_version=graph_spec_version,
        model_family=f"gnn_{model_config['architecture']}",
        seeds=seeds,
        notes=f"Single-view GNN ({view}); validation-only model selection.",
    )
    update_status(exp_id, "running")
    run_dir = experiment_dir(exp_id)

    records = load_records(export_path)
    train_records = [record for record in records if str(record["sample_id"]) in train_ids]
    vocabulary = TokenVocabulary.fit(
        train_records,
        min_count=int(model_config.get("min_token_count", 2)),
        max_size=int(model_config.get("max_vocab", 20000)),
    )
    vocabulary.save(run_dir / "vocabulary.json")

    labels = dict(
        zip(
            labels_frame["sample_id"].astype(str),
            labels_frame["label"].astype(int),
        )
    )

    train_graphs, train_kept, train_empty = build_dataset(
        export_path, vocabulary=vocabulary, sample_ids=train_ids
    )
    validation_graphs, validation_kept, validation_empty = build_dataset(
        export_path, vocabulary=vocabulary, sample_ids=validation_ids
    )
    for graph in train_graphs:
        graph.y = torch.tensor([labels[graph.sample_id]], dtype=torch.long)
    for graph in validation_graphs:
        graph.y = torch.tensor([labels[graph.sample_id]], dtype=torch.long)
    if not train_graphs or not validation_graphs:
        raise RuntimeError("GNN training requires non-empty train and validation graph sets")

    positives = sum(1 for graph in train_graphs if int(graph.y.item()) == 1)
    negatives = len(train_graphs) - positives
    positive_weight = torch.tensor([max(negatives / max(positives, 1), 1.0)], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary: dict[str, Any] = {
        "experiment_id": exp_id,
        "config_id": cfg_id,
        "split_id": split_id,
        "graph_build_id": graph_build_id,
        "view": view,
        "vocabulary_size": vocabulary.size,
        "graphs": {
            "train": len(train_graphs),
            "validation": len(validation_graphs),
            "empty_excluded_train": len(train_empty),
            "empty_excluded_validation": len(validation_empty),
        },
        "seeds": {},
    }

    for seed in seeds:
        _seed_everything(seed)
        model = build_model(
            architecture=model_config["architecture"],
            vocabulary_size=vocabulary.size,
            embedding_dim=int(model_config.get("embedding_dim", 32)),
            hidden_dim=int(model_config["hidden_dim"]),
            layers=int(model_config["layers"]),
            dropout=float(model_config["dropout"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(model_config["learning_rate"])
        )
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight.to(device))
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_graphs,
            batch_size=int(model_config["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_graphs, batch_size=int(model_config["batch_size"]), shuffle=False
        )

        best = {"epoch": -1, "macro_f1": -1.0, "state": None, "metrics": {}}
        patience = int(model_config["early_stopping_patience"])
        stale = 0
        log_path = run_dir / f"training_seed{seed}.jsonl"
        for epoch in range(1, int(model_config["epochs"]) + 1):
            model.train()
            epoch_losses: list[float] = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y.float())
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            metrics = _evaluate(model, validation_loader, device, criterion)
            entry = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry) + "\n")
            if metrics["macro_f1"] > best["macro_f1"] + 1e-6:
                best = {
                    "epoch": epoch,
                    "macro_f1": metrics["macro_f1"],
                    "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "metrics": metrics,
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        checkpoint_path = run_dir / f"checkpoint_seed{seed}.pt"
        torch.save(
            {
                "experiment_id": exp_id,
                "config_id": cfg_id,
                "split_id": split_id,
                "graph_build_id": graph_build_id,
                "graph_spec_version": graph_spec_version,
                "view": view,
                "model_config": model_config,
                "seed": seed,
                "epoch": best["epoch"],
                "validation_metrics": best["metrics"],
                "state_dict": best["state"],
                "vocabulary": vocabulary.token_to_index,
                "vocabulary_meta": {"min_count": vocabulary.min_count, "max_size": vocabulary.max_size},
                "created_at": utc_now(),
            },
            checkpoint_path,
        )
        summary["seeds"][str(seed)] = {
            "best_epoch": best["epoch"],
            "validation_metrics": best["metrics"],
            "checkpoint": str(checkpoint_path),
        }

    write_json(run_dir / "training_summary.json", summary)
    log_metrics(exp_id, {"gnn": summary["seeds"]})
    update_status(exp_id, "completed")
    return summary
