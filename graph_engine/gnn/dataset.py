"""GNN dataset preparation (Person 2).

Node features are label-free. The token vocabulary is fit on the training split
only and stored with the checkpoint, so validation/test tokens are never part of
the model's feature space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from graph_engine.graph_export.export import read_export
from shared.contracts.validators import validate_graph_record

NUMERIC_FEATURES = ("is_stopword", "is_punctuation", "position_norm", "degree_norm")


class GnnDependencyError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - optional extra
        raise GnnDependencyError(
            "GNN extras are not installed. Run `pip install -r requirements/gnn.txt`."
        ) from exc
    return torch, Data


@dataclass
class TokenVocabulary:
    token_to_index: dict[str, int]
    min_count: int = 2
    max_size: int = 20000

    @property
    def size(self) -> int:
        return len(self.token_to_index) + 1  # index 0 = unknown

    @classmethod
    def fit(cls, records: Iterable[dict[str, Any]], *, min_count: int = 2, max_size: int = 20000) -> "TokenVocabulary":
        counts: dict[str, int] = {}
        for record in records:
            for node in record["nodes"]:
                token = str(node["token"])
                counts[token] = counts.get(token, 0) + 1
        ordered = sorted(
            ((token, count) for token, count in counts.items() if count >= min_count),
            key=lambda item: (-item[1], item[0]),
        )[:max_size]
        return cls(
            token_to_index={token: index + 1 for index, (token, _) in enumerate(ordered)},
            min_count=min_count,
            max_size=max_size,
        )

    def encode(self, token: str) -> int:
        return self.token_to_index.get(token, 0)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "min_count": self.min_count,
                    "max_size": self.max_size,
                    "token_to_index": self.token_to_index,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TokenVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            token_to_index={str(k): int(v) for k, v in payload["token_to_index"].items()},
            min_count=int(payload["min_count"]),
            max_size=int(payload["max_size"]),
        )


def record_to_data(record: dict[str, Any], vocabulary: TokenVocabulary):
    torch, Data = _require_torch()
    validate_graph_record(record)
    n_nodes = len(record["nodes"])
    if n_nodes == 0:
        return None
    token_ids = [vocabulary.encode(str(node["token"])) for node in record["nodes"]]

    degree = np.zeros(n_nodes, dtype=np.float32)
    for edge in record["edges"]:
        degree[int(edge["source"])] += 1.0
        degree[int(edge["target"])] += 1.0
    max_degree = float(degree.max()) if n_nodes else 1.0
    positions = np.asarray(
        [float(node.get("position", index)) for index, node in enumerate(record["nodes"])],
        dtype=np.float32,
    )
    max_position = float(positions.max()) if n_nodes else 1.0

    numeric = np.stack(
        [
            np.asarray([1.0 if node.get("is_stopword") else 0.0 for node in record["nodes"]], dtype=np.float32),
            np.asarray([1.0 if node.get("node_type") == "punctuation" else 0.0 for node in record["nodes"]], dtype=np.float32),
            positions / max(max_position, 1.0),
            degree / max(max_degree, 1.0),
        ],
        axis=1,
    )

    sources: list[int] = []
    targets: list[int] = []
    for edge in record["edges"]:
        source, target = int(edge["source"]), int(edge["target"])
        if source == target:
            continue
        sources.extend([source, target])
        targets.extend([target, source])

    edge_index = (
        torch.tensor([sources, targets], dtype=torch.long)
        if sources
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.tensor(numeric, dtype=torch.float32),
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        edge_index=edge_index,
        sample_id=str(record["sample_id"]),
        num_nodes=n_nodes,
    )


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return list(read_export(path))


def build_dataset(
    path: str | Path,
    *,
    vocabulary: TokenVocabulary,
    sample_ids: set[str] | None = None,
) -> tuple[list[Any], list[str], list[str]]:
    """Return (graphs, kept_ids, empty_ids). Empty graphs are excluded."""
    graphs: list[Any] = []
    kept: list[str] = []
    empty: list[str] = []
    for record in read_export(path):
        sample_id = str(record["sample_id"])
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        data = record_to_data(record, vocabulary)
        if data is None:
            empty.append(sample_id)
            continue
        graphs.append(data)
        kept.append(sample_id)
    return graphs, kept, empty
