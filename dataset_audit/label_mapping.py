"""Label semantics and task reconciliation (Person 1).

The dataset labels authorship (0 = human, 1 = AI). Paraphrase condition is a
separate property. This module refuses to reinterpret labels and reports any
place where task, condition and label cannot be separated experimentally.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CANONICAL_LABELS = {0: "human_authored", 1: "ai_authored"}


class LabelContractError(ValueError):
    pass


def validate_labels(frame: pd.DataFrame) -> dict[str, Any]:
    if "label" not in frame:
        raise LabelContractError("label column missing")
    nulls = int(frame["label"].isna().sum())
    if nulls:
        raise LabelContractError(f"{nulls} rows have a null label")
    observed = sorted(int(value) for value in frame["label"].unique())
    unexpected = [value for value in observed if value not in CANONICAL_LABELS]
    if unexpected:
        raise LabelContractError(
            f"Unexpected label values {unexpected}; canonical labels are {CANONICAL_LABELS}. "
            "Do not remap without documenting the source task semantics."
        )
    return {
        "observed": {str(value): CANONICAL_LABELS[value] for value in observed},
        "counts": {str(value): int((frame["label"] == value).sum()) for value in observed},
    }


def task_semantics(config: dict[str, Any]) -> pd.DataFrame:
    """Task reconciliation table assembled from the documented config."""
    rows: list[dict[str, Any]] = []
    for source, source_cfg in config.get("sources", {}).items():
        if source_cfg.get("tasks"):
            for task, task_cfg in source_cfg["tasks"].items():
                rows.append(
                    {
                        "source_dataset": source,
                        "source_task": task,
                        "authorship_comparable": bool(task_cfg.get("authorship_comparable")),
                        "condition": task_cfg.get("condition"),
                        "condition_confidence": task_cfg.get("condition_confidence"),
                        "status": task_cfg.get("status"),
                        "caveat": task_cfg.get("caveat", ""),
                    }
                )
        else:
            rows.append(
                {
                    "source_dataset": source,
                    "source_task": source,
                    "authorship_comparable": bool(source_cfg.get("role") == "pooled_authorship"),
                    "condition": "row_level_paraphrase_flag",
                    "condition_confidence": "medium",
                    "status": "documented",
                    "caveat": source_cfg.get("notes", ""),
                }
            )
    return pd.DataFrame(rows)


def observed_task_condition_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Actual label distribution per (source task, text type) cell."""
    matrix = (
        frame.groupby(["source_dataset", "source_task", "text_type", "label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return matrix


def detect_confounding(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Find cells/tasks where condition determines label (or vice versa).

    Such cells cannot support a claim that paraphrasing caused a difference,
    because the condition and the authorship label move together.
    """
    findings: list[dict[str, Any]] = []
    grouped = frame.groupby(["source_dataset", "source_task", "text_type"], dropna=False)
    for (source, task, text_type), cell in grouped:
        labels = set(int(value) for value in cell["label"].unique())
        if len(labels) == 1:
            findings.append(
                {
                    "source_dataset": source,
                    "source_task": task,
                    "text_type": text_type,
                    "only_label": labels.pop(),
                    "rows": int(len(cell)),
                    "kind": "condition_predicts_label",
                    "consequence": "Cannot be used as evidence of a paraphrase effect; report as confounded.",
                }
            )
    # A label that occurs with exactly one text_type inside a task is the
    # symmetric problem (class is condition-specific).
    for (source, task), task_frame in frame.groupby(["source_dataset", "source_task"], dropna=False):
        for label, label_frame in task_frame.groupby("label", dropna=False):
            text_types = [str(value) for value in label_frame["text_type"].dropna().unique()]
            if len(text_types) == 1:
                findings.append(
                    {
                        "source_dataset": source,
                        "source_task": task,
                        "label": int(label),
                        "only_text_type": text_types[0],
                        "rows": int(len(label_frame)),
                        "kind": "label_predicts_condition",
                        "consequence": "Condition-specific label; pooled condition claims are invalid.",
                    }
                )
    return findings
