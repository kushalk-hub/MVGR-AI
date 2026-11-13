"""Versioned contracts shared by Person 1, Person 2 and the backend.

Nothing in this package may import person1.*, person2.* or backend.*.
It is the dependency root of the repository.
"""

from .versions import (
    CONTRACT_VERSIONS,
    EXPORT_FORMAT,
    FEATURE_SET_VERSION,
    GRAPH_SCHEMA_VERSION,
    GRAPH_SPEC_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
    TOKENIZER_VERSION,
)

__all__ = [
    "CONTRACT_VERSIONS",
    "EXPORT_FORMAT",
    "FEATURE_SET_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "GRAPH_SPEC_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "SPLIT_POLICY_VERSION",
    "TOKENIZER_VERSION",
]
