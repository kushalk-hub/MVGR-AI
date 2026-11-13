"""GNN model factory (Person 2).

One graph view, one simple GNN first. Multi-view fusion is intentionally not
implemented here until single-view models have been evaluated on the same split.
"""

from __future__ import annotations

from typing import Any

from graph_engine.gnn.dataset import NUMERIC_FEATURES, GnnDependencyError


def build_model(
    *,
    architecture: str = "gcn",
    vocabulary_size: int,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    layers: int = 2,
    dropout: float = 0.2,
):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
    except ImportError as exc:  # pragma: no cover - optional extra
        raise GnnDependencyError(
            "GNN extras are not installed. Run `pip install -r requirements/gnn.txt`."
        ) from exc

    conv_types = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
    if architecture not in conv_types:
        raise ValueError(f"unknown architecture {architecture!r}; choose from {sorted(conv_types)}")
    conv_type = conv_types[architecture]

    class GraphClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
            input_dim = embedding_dim + len(NUMERIC_FEATURES)
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.convolutions = nn.ModuleList()
            self.normalizations = nn.ModuleList()
            for _ in range(layers):
                if architecture == "gat":
                    self.convolutions.append(conv_type(hidden_dim, hidden_dim, heads=2, concat=False))
                else:
                    self.convolutions.append(conv_type(hidden_dim, hidden_dim))
                self.normalizations.append(nn.LayerNorm(hidden_dim))
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, data: Any):
            embedded = self.embedding(data.token_ids)
            x = torch.cat([embedded, data.x], dim=1)
            x = self.input_projection(x)
            for convolution, normalization in zip(self.convolutions, self.normalizations):
                x = convolution(x, data.edge_index)
                x = normalization(x)
                x = F.relu(x)
                x = self.dropout(x)
            pooled = global_mean_pool(x, data.batch)
            return self.head(pooled).squeeze(-1)

    return GraphClassifier()
