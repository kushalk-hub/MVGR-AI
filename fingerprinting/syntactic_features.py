"""Module A(e) — syntactic features.

Dependency structure comes from the local spaCy parser, the same model that
builds the syntactic graph view. Distance is the absolute token-index gap between
a token and its head, matching the graph view's ``distance`` definition.
"""

from __future__ import annotations

from typing import Any, Sequence

from fingerprinting.stats_utils import (
    counter_from,
    entropy,
    mean,
    normalized_entropy,
    safe_div,
    stdev,
)

DEFAULT_TOP_RELATIONS = 15
DEFAULT_PASSIVE_LABELS = ("auxpass", "nsubjpass", "aux:pass", "nsubj:pass")
DEFAULT_SUBORDINATE_LABELS = ("mark", "advcl", "ccomp", "xcomp", "acl", "relcl", "csubj")


def _tree_depth(doc: Any) -> tuple[int | None, int | None]:
    """Return ``(max_depth, root_count)`` of the dependency tree."""
    tokens = [token for token in doc if not token.is_space]
    if not tokens:
        return None, None
    depth_cache: dict[int, int] = {}
    roots = 0

    def depth(token: Any, seen: frozenset[int]) -> int:
        if token.i in depth_cache:
            return depth_cache[token.i]
        if token.head.i == token.i:
            depth_cache[token.i] = 0
            return 0
        if token.head.i in seen:  # defensive cycle guard
            return 0
        value = 1 + depth(token.head, seen | {token.i})
        depth_cache[token.i] = value
        return value

    for token in tokens:
        if token.head.i == token.i:
            roots += 1
    max_depth = max((depth(token, frozenset()) for token in tokens), default=0)
    return max_depth, roots


def extract(
    text: str,
    *,
    tokens: Sequence[Any],
    sentences: Sequence[str],
    doc: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | None]:
    config = config or {}
    top_k = int(config.get("top_dependency_relations", DEFAULT_TOP_RELATIONS))
    passive_labels = set(config.get("passive_labels", DEFAULT_PASSIVE_LABELS))
    subordinate_labels = set(config.get("subordinate_labels", DEFAULT_SUBORDINATE_LABELS))

    if doc is None:
        keys = [
            "syn_dependency_count",
            "syn_distinct_relations",
            "syn_dependency_diversity",
            "syn_dependency_entropy",
            "syn_dependency_normalized_entropy",
            "syn_distance_mean",
            "syn_distance_max",
            "syn_distance_std",
            "syn_tree_depth",
            "syn_root_count",
            "syn_passive_count",
            "syn_passive_ratio",
            "syn_subordinate_count",
            "syn_subordinate_ratio",
        ]
        keys += [f"syn_rel_share_{index}" for index in range(top_k)]
        keys.append("syn_rel_share_other")
        return {key: None for key in keys}

    words = [token for token in doc if not token.is_space]
    total = len(words)
    relations = [token.dep_ for token in words]
    counts = counter_from(relations)

    distances = [
        abs(token.i - token.head.i) for token in words if token.head.i != token.i
    ]

    passive = sum(
        1
        for token in words
        if token.dep_ in passive_labels or "Voice=Pass" in str(token.morph)
    )
    subordinate = sum(1 for token in words if token.dep_ in subordinate_labels)
    max_depth, roots = _tree_depth(doc)

    features: dict[str, float | int | None] = {
        "syn_dependency_count": total,
        "syn_distinct_relations": len(counts),
        "syn_dependency_diversity": safe_div(len(counts), total),
        "syn_dependency_entropy": entropy(counts.values()),
        "syn_dependency_normalized_entropy": normalized_entropy(counts.values()),
        "syn_distance_mean": mean(distances),
        "syn_distance_max": max(distances) if distances else None,
        "syn_distance_std": stdev(distances),
        "syn_tree_depth": max_depth,
        "syn_root_count": roots,
        "syn_passive_count": passive,
        "syn_passive_ratio": safe_div(passive, total),
        "syn_subordinate_count": subordinate,
        "syn_subordinate_ratio": safe_div(subordinate, total),
    }

    # Top-K relations by count, deterministic tie-break by label; the rest are
    # folded into a single "other" share so the column count stays bounded.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    for index in range(top_k):
        features[f"syn_rel_share_{index}"] = (
            safe_div(ranked[index][1], total) if index < len(ranked) else 0.0
        )
    kept = sum(value for _, value in ranked)
    features["syn_rel_share_other"] = safe_div(total - kept, total)
    return features


__all__ = [
    "DEFAULT_PASSIVE_LABELS",
    "DEFAULT_SUBORDINATE_LABELS",
    "DEFAULT_TOP_RELATIONS",
    "extract",
]
