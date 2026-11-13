"""GNN package (Person 2). Heavy imports are lazy."""

from graph_engine.gnn.dataset import TokenVocabulary, build_dataset, record_to_data
from graph_engine.gnn.model import build_model

__all__ = ["TokenVocabulary", "build_dataset", "build_model", "record_to_data"]
