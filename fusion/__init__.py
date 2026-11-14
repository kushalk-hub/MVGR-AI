"""Module E — multi-view feature fusion package.

Fuses the existing graph branches with the new offline fingerprinting branches
(stylometric, writeprints, GLTR) and keeps watermark results as auxiliary
evidence. Identifiers and metadata are always kept out of the feature matrix.
"""

from __future__ import annotations

from fusion.multiview_feature_fusion import (
    BRANCH_ORDER,
    FusionError,
    FusionResult,
    build_schema,
    fuse_multiview,
    normalize_features,
    select_features,
    split_graph_branches,
)

__all__ = [
    "BRANCH_ORDER",
    "FusionError",
    "FusionResult",
    "build_schema",
    "fuse_multiview",
    "normalize_features",
    "select_features",
    "split_graph_branches",
]
