"""Single source of truth for every contract version in MVGR-AI.

Rules
-----
* A version is bumped in the same pull request that changes the corresponding
  schema or semantics. Never bump silently.
* Changing an existing version in place is forbidden once any artifact with
  that version exists. Add a new version and keep readers for the old one.
"""

from __future__ import annotations

GRAPH_SCHEMA_VERSION = "1.0.0"
GRAPH_SPEC_VERSION = "1.0.0"
EXPORT_FORMAT = "jsonl+json-manifest"
PREDICTION_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.0.0"
FAMILY_INDEX_SCHEMA_VERSION = "1.0.0"
EXPERIMENT_SCHEMA_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0.0"

SPLIT_POLICY_VERSION = "split.v1"
TOKENIZER_VERSION = "tokenizer.v1"
FEATURE_SET_VERSION = "features.v1"

CONTRACT_VERSIONS: dict[str, str] = {
    "graph_schema": GRAPH_SCHEMA_VERSION,
    "graph_spec": GRAPH_SPEC_VERSION,
    "prediction_schema": PREDICTION_SCHEMA_VERSION,
    "manifest_schema": MANIFEST_SCHEMA_VERSION,
    "family_index_schema": FAMILY_INDEX_SCHEMA_VERSION,
    "experiment_schema": EXPERIMENT_SCHEMA_VERSION,
    "analysis_report_schema": REPORT_SCHEMA_VERSION,
    "split_policy": SPLIT_POLICY_VERSION,
    "tokenizer": TOKENIZER_VERSION,
    "feature_set": FEATURE_SET_VERSION,
}
