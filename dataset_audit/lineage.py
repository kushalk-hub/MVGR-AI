"""Family lineage recovery (Person 1).

Family identity protects the split from paraphrase leakage. The current
dataset exposes one usable signal: ``group_id``. For PADBen it is pair-level.
For BiScope it is an exact-text hash, which is NOT a paraphrase family link.

This module never invents a link. Unrecoverable relationships are recorded as
``exact_text_only`` or ``unresolved`` and are surfaced in the audit and the
split gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from shared.contracts.validators import validate, write_jsonl
from shared.paths import derived_root
from shared.preprocessing.text import normalize_for_hashing, sha256_text


def _source_policy(config: dict[str, Any], source: str) -> tuple[str, str]:
    source_cfg = config.get("sources", {}).get(source, {})
    resolution = str(source_cfg.get("lineage_resolution", "unresolved"))
    if resolution.startswith("pair_level"):
        return "source_group_id", "resolved"
    if resolution.startswith("unresolved_exact_text_only"):
        return "exact_text_hash", "exact_text_only"
    return "singleton_fallback", "unresolved"


def build_family_index(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Return one schema-valid family record per sample.

    The function is deterministic and does not depend on row order.
    """
    required = {"sample_id", "text", "source_dataset", "group_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"family index requires columns {sorted(missing)}")

    records: list[dict[str, Any]] = []
    for row in frame[["sample_id", "text", "source_dataset", "group_id"]].itertuples(index=False):
        source = str(row.source_dataset)
        group = "" if pd.isna(row.group_id) else str(row.group_id).strip()
        family_source, status = _source_policy(config, source)

        if family_source == "source_group_id" and group:
            family_id = f"{source}:{group}"
            evidence = "PADBen pair/source grouping from group_id; both pair members share one family."
        elif family_source == "exact_text_hash":
            digest = sha256_text(normalize_for_hashing(str(row.text)))[:20]
            family_id = f"{source}:exact:{digest}"
            evidence = (
                "No paraphrase-family identifier exists in this source. Family is exact normalized "
                "text only, so lexical paraphrases of the same source text can cross splits."
            )
        else:
            digest = sha256_text(normalize_for_hashing(str(row.text)))[:20]
            family_id = f"{source}:singleton:{digest}"
            evidence = "No group identifier; singleton fallback by exact normalized text."

        records.append(
            {
                "sample_id": str(row.sample_id),
                "family_id": family_id,
                "family_source": family_source,
                "resolution_status": status,
                "evidence": evidence,
            }
        )

    index = pd.DataFrame.from_records(records)
    if index["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id while building family index")
    return index.sort_values("sample_id").reset_index(drop=True)


def family_summary(index: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    merged = frame[["sample_id", "source_dataset", "split", "label"]].merge(
        index[["sample_id", "family_id", "resolution_status"]], on="sample_id", how="left"
    )
    sizes = merged.groupby("family_id").size()
    label_mix = merged.groupby("family_id")["label"].nunique()
    split_mix = merged.groupby("family_id")["split"].nunique()

    return {
        "rows": int(len(merged)),
        "families": int(merged["family_id"].nunique()),
        "resolution_status": {
            str(key): int(value)
            for key, value in merged["resolution_status"].fillna("missing").value_counts().items()
        },
        "family_size_distribution": {
            str(key): int(value) for key, value in sizes.value_counts().sort_index().items()
        },
        "families_with_both_labels": int((label_mix > 1).sum()),
        "families_spanning_splits": int((split_mix > 1).sum()),
        "by_source": {
            str(source): {
                "rows": int(len(group)),
                "families": int(group["family_id"].nunique()),
                "resolved": int((group["resolution_status"] == "resolved").sum()),
                "exact_text_only": int((group["resolution_status"] == "exact_text_only").sum()),
                "unresolved": int((group["resolution_status"] == "unresolved").sum()),
            }
            for source, group in merged.groupby("source_dataset")
        },
    }


def write_family_index(
    index: pd.DataFrame,
    *,
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    if config is not None:
        path = path or (derived_root(config) / "family_index.jsonl")
    path = Path(path or (derived_root() / "family_index.jsonl"))
    records = index.to_dict(orient="records")
    for record in records:
        validate("family_index", record)
    return write_jsonl(path, records)


def read_family_index(path: str | Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True, dtype=False)
    for record in frame.to_dict(orient="records"):
        validate("family_index", record)
    return frame
