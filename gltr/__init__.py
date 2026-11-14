"""Module C — GLTR-style local token analysis.

GLTR ("Giant Language model Test Room") scores each token by how predictable it
is under a reference language model. This package implements the *offline*
version: a locally cached causal language model, no network, no telemetry.

These features are **statistical signals, not proof of authorship**. A text with
many high-probability tokens is not thereby "AI-generated"; human text also
contains predictable tokens. The module is only meaningful alongside the other
evidence branches.

If no local causal model is available, the module reports itself as
``unavailable`` and never downloads one automatically.
"""

from __future__ import annotations

from gltr.document_features import aggregate_document_features
from gltr.local_model import (
    GltrConfig,
    GltrModel,
    GltrUnavailableError,
    local_model_available,
    transformers_available,
)
from gltr.pipeline import GltrAnalyzer, estimate_requirements

__all__ = [
    "GltrAnalyzer",
    "GltrConfig",
    "GltrModel",
    "GltrUnavailableError",
    "aggregate_document_features",
    "estimate_requirements",
    "local_model_available",
    "transformers_available",
]
