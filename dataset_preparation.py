from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import get_dataset_config_names, load_dataset, load_dataset_builder
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "datasets" / "raw"
PROCESSED_DIR = ROOT / "datasets" / "processed"
OUTPUT_DIR = ROOT / "outputs"

BISCOPE_ID = "HanxiGuo/BiScope_Data"
PADBEN_ID = "JonathanZha/PADBen"

# Only these PADBen task configurations are candidates for the main
# binary authorship experiment. Actual available config names are discovered.
PADBEN_TASK_PATTERNS = ("task1", "task2", "task5")

TEXT_ALIASES = (
    "text", "sentence", "content", "document", "paragraph",
    "generated_text", "response", "essay", "text_a"
)
SOURCE_ALIASES = ("source", "author", "origin", "writer", "text_source")
LABEL_ALIASES = ("label", "labels", "class", "target", "is_ai", "ai_generated")
TASK_ALIASES = ("task", "task_name", "category", "domain", "dataset_task")
PARAPHRASE_ALIASES = ("paraphrased", "is_paraphrased", "paraphrase", "paraphrase_status")
MODEL_ALIASES = ("model", "model_name", "generator", "llm", "ai_model")
ID_ALIASES = ("id", "sample_id", "example_id", "row_id", "uid")
GROUP_ALIASES = (
    "original_id", "source_id", "pair_id", "group_id",
    "prompt_id", "document_id", "original_text_id"
)

def ensure_dirs() -> None:
    for p in (RAW_DIR / "biscope", RAW_DIR / "padben", PROCESSED_DIR, OUTPUT_DIR):
        p.mkdir(parents=True, exist_ok=True)

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "names"):  # datasets ClassLabel
        return {"type": value.__class__.__name__, "names": list(value.names)}
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)

def discover_schema() -> dict[str, Any]:
    report: dict[str, Any] = {"datasets": {}, "notes": []}
    for ds_id, key in ((BISCOPE_ID, "BiScope"), (PADBEN_ID, "PADBen")):
        print(f"\nDiscovering {key}: {ds_id}")
        try:
            configs = get_dataset_config_names(ds_id)
            report["datasets"][key] = {"dataset_id": ds_id, "configs": {}}
            for config in configs:
                try:
                    builder = load_dataset_builder(ds_id, name=config)
                    info = builder.info
                    features = {}
                    if info.features:
                        for col, feature in info.features.items():
                            features[col] = {
                                "feature_type": feature.__class__.__name__,
                                "description": getattr(feature, "description", None),
                                "names": list(feature.names) if hasattr(feature, "names") else None,
                                "dtype": getattr(feature, "dtype", None),
                            }
                    report["datasets"][key]["configs"][config] = {
                        "splits": list(info.splits.keys()) if info.splits else [],
                        "features": features,
                        "description": (info.description or "")[:3000],
                        "homepage": info.homepage,
                        "license": info.license,
                    }
                except Exception as exc:
                    report["datasets"][key]["configs"][config] = {"inspection_error": str(exc)}
        except Exception as exc:
            report["datasets"][key] = {"dataset_id": ds_id, "error": str(exc)}
            report["notes"].append(f"Could not inspect {ds_id}: {exc}")
    return report

def write_schema_report(report: dict[str, Any]) -> None:
    (OUTPUT_DIR / "dataset_schema_report.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = ["MVGR-AI DATASET SCHEMA REPORT", "=" * 34]
    for ds_name, ds in report.get("datasets", {}).items():
        lines += ["", ds_name, "-" * len(ds_name), f"Dataset: {ds.get('dataset_id')}"]
        if ds.get("error"):
            lines.append(f"ERROR: {ds['error']}")
            continue
        lines.append("Configurations:")
        for config, info in ds.get("configs", {}).items():
            lines.append(f"\n  [{config}]")
            lines.append(f"  Splits: {info.get('splits', [])}")
            for col, feat in info.get("features", {}).items():
                names = f" names={feat['names']}" if feat.get("names") else ""
                lines.append(f"    - {col}: {feat['feature_type']}{names}")
            if info.get("inspection_error"):
                lines.append(f"  Inspection error: {info['inspection_error']}")
    lines += ["", "Notes:", *[f"- {n}" for n in report.get("notes", [])]]
    (OUTPUT_DIR / "dataset_schema_report.txt").write_text("\n".join(lines), encoding="utf-8")

def norm_col(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")

def first_matching(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {norm_col(c): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None

def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (list, dict)):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()

def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None

def infer_label_from_name(value: Any) -> int | None:
    s = str(value).strip().lower()
    if any(x in s for x in ("human", "person", "human-written", "human_written")):
        return 0
    if any(x in s for x in ("ai", "machine", "llm", "gpt", "claude", "gemini", "generated")):
        return 1
    return None

def get_class_names(feature: Any) -> list[str] | None:
    # datasets.ClassLabel exposes .names
    names = getattr(feature, "names", None)
    return list(names) if names else None

def map_padben_authorship_label(value: Any, task_name: str) -> int | None:
    task_match = re.search(r"task(\d+)", task_name.lower())
    task = f"task{task_match.group(1)}" if task_match else None
    if task not in PADBEN_TASK_PATTERNS:
        return None

    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        label = int(value)
    elif isinstance(value, str) and value.strip() in {"0", "1"}:
        label = int(value.strip())
    else:
        return None

    # PADBen documents 0 as human and 1 as machine for tasks 1, 2, and 5.
    return label if label in (0, 1) else None

def label_from_row(
    row: dict[str, Any],
    dataset_name: str,
    features: Any,
    task_name: str,
) -> int:
    # First, explicit semantic label fields.
    label_col = first_matching(list(row.keys()), LABEL_ALIASES)
    if label_col:
        raw = row[label_col]
        feature = features.get(label_col) if features else None
        names = get_class_names(feature) if feature is not None else None
        if names is not None:
            try:
                class_name = names[int(raw)]
                mapped = infer_label_from_name(class_name)
                if mapped is not None:
                    return mapped
            except Exception:
                pass
        mapped = infer_label_from_name(raw)
        if mapped is not None:
            return mapped
        if dataset_name == "PADBen":
            mapped = map_padben_authorship_label(raw, task_name)
            if mapped is not None:
                return mapped
        # Numeric labels without explicit class names are unsafe to infer.
        raise ValueError(
            f"{dataset_name}: label value {raw!r} in column {label_col!r} has no "
            "explicit human/AI class name. Check outputs/dataset_schema_report.json "
            "and configure the mapping in label_from_row(); do not guess 0/1."
        )

    # BiScope commonly exposes a source-like field instead of a numeric label.
    if dataset_name == "BiScope":
        source_col = first_matching(list(row.keys()), SOURCE_ALIASES)
        if source_col:
            mapped = infer_label_from_name(row[source_col])
            if mapped is not None:
                return mapped

    raise ValueError(
        f"{dataset_name}: no unambiguous authorship label found in columns "
        f"{list(row.keys())}. Inspect outputs/dataset_schema_report.json."
    )

def extract_text(row: dict[str, Any], dataset_name: str) -> str:
    col = first_matching(list(row.keys()), TEXT_ALIASES)
    if col is None:
        candidates = [c for c, v in row.items() if isinstance(v, str) and len(v.strip()) > 30]
        if len(candidates) == 1:
            col = candidates[0]
        else:
            raise ValueError(
                f"{dataset_name}: cannot safely identify one text field. "
                f"Columns={list(row.keys())}; long string candidates={candidates}. "
                "Review schema and update TEXT_ALIASES."
            )
    text = normalize_text(row[col])
    if not text:
        raise ValueError(f"{dataset_name}: selected text field {col!r} is empty.")
    return text

def classify_text_type(
    row: dict[str, Any],
    task_name: str,
    dataset_name: str,
    label: int | None = None,
) -> tuple[str, bool | None]:
    pcol = first_matching(list(row.keys()), PARAPHRASE_ALIASES)
    if pcol:
        flag = parse_bool(row[pcol])
        if flag is not None:
            return ("paraphrased" if flag else "original"), flag
    task = str(task_name).lower()
    if "task5" in task:
        return ("original", False) if label == 0 else ("deep_paraphrased", True)
    if "task1" in task and dataset_name == "PADBen":
        return "paraphrased", True
    if dataset_name == "BiScope":
        return "unknown", None
    return "unknown", None

def normalize_dataset(ds, dataset_name: str, task_name: str, config_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    features = getattr(ds, "features", None)
    for idx, row in enumerate(tqdm(ds, desc=f"Normalizing {dataset_name}/{config_name}")):
        if "sentence_pair" in row or "label_pair" in row:
            sentence_pair = row.get("sentence_pair")
            label_pair = row.get("label_pair")
            if (
                not isinstance(sentence_pair, (list, tuple))
                or not isinstance(label_pair, (list, tuple))
                or len(sentence_pair) != len(label_pair)
                or not sentence_pair
            ):
                raise ValueError(
                    f"{dataset_name}: expected non-empty sentence_pair and label_pair "
                    f"lists of equal length; got {sentence_pair!r} and {label_pair!r}."
                )
            text_labels = []
            for text_value, raw_label in zip(sentence_pair, label_pair):
                text = normalize_text(text_value)
                label = map_padben_authorship_label(raw_label, task_name)
                if not text:
                    raise ValueError(f"{dataset_name}: sentence_pair contains empty text.")
                if label is None:
                    raise ValueError(
                        f"{dataset_name}: unsupported label {raw_label!r} for {task_name!r}."
                    )
                text_labels.append((text, label))
        else:
            text_labels = [
                (
                    extract_text(row, dataset_name),
                    label_from_row(row, dataset_name, features, task_name),
                )
            ]

        domain_col = first_matching(list(row.keys()), TASK_ALIASES)
        model_col = first_matching(list(row.keys()), MODEL_ALIASES)
        id_col = first_matching(list(row.keys()), ID_ALIASES)
        group_col = first_matching(list(row.keys()), GROUP_ALIASES)

        source_row_value = row[id_col] if id_col and row[id_col] is not None else row.get("idx", idx)
        source_row_id = str(source_row_value)
        group_value = row[group_col] if group_col and row[group_col] is not None else None
        if group_value is None and "sentence_pair" in row:
            group_value = source_row_value
        # Prefix groups by source/config so unrelated ID namespaces cannot collide.
        if group_value is not None:
            group_id = f"{dataset_name}:{config_name}:{group_value}"
        else:
            group_id = None

        domain = normalize_text(row[domain_col]) if domain_col else "Unknown"
        ai_model = normalize_text(row[model_col]) if model_col else "Unknown"
        original_split = getattr(ds, "_split", None)

        for pair_index, (text, label) in enumerate(text_labels):
            text_type, paraphrase_status = classify_text_type(
                row, task_name, dataset_name, label
            )
            records.append({
                "sample_id": f"{dataset_name}_{config_name}_{idx}_{pair_index}",
                "text": text,
                "label": int(label),
                "source_dataset": dataset_name,
                "source_task": task_name or config_name,
                "text_type": text_type,
                "paraphrase_status": paraphrase_status,
                "domain": domain or "Unknown",
                "ai_model": ai_model or "Unknown",
                "padben_split": str(original_split) if dataset_name == "PADBen" and original_split else "",
                "split": "",
                "group_id": group_id or f"{dataset_name}:{stable_hash(text)}",
                "source_row_id": source_row_id,
            })
    return records

def load_and_normalize() -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    print("\nLoading BiScope...")
    biscope = load_dataset(BISCOPE_ID)
    if isinstance(biscope, dict):
        split_names = list(biscope.keys())
        chosen = "train" if "train" in biscope else split_names[0]
        print(f"BiScope splits found: {split_names}; using {chosen}")
        records.extend(normalize_dataset(biscope[chosen], "BiScope", "BiScope", chosen))
    else:
        records.extend(normalize_dataset(biscope, "BiScope", "BiScope", "default"))

    print("\nDiscovering PADBen configurations...")
    configs = get_dataset_config_names(PADBEN_ID)
    selected = [c for c in configs if any(p in c.lower().replace("-", "") for p in ("task1", "task2", "task5"))]
    if not selected:
        raise RuntimeError(
            "Could not find PADBen configurations matching Task 1, Task 2, or Task 5. "
            f"Available configs: {configs}. Review the schema report and update PADBEN_TASK_PATTERNS."
        )

    print(f"Selected PADBen configs: {selected}")
    for config in selected:
        print(f"\nLoading PADBen config: {config}")
        loaded = load_dataset(PADBEN_ID, name=config)
        if isinstance(loaded, dict):
            for split_name, split_ds in loaded.items():
                records.extend(normalize_dataset(split_ds, "PADBen", config, f"{config}_{split_name}"))
        else:
            records.extend(normalize_dataset(loaded, "PADBen", config, config))

    if not records:
        raise RuntimeError("No records were normalized.")
    return pd.DataFrame.from_records(records)

def assign_group_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    # GroupShuffleSplit prevents rows sharing the same known group_id from crossing splits.
    groups = df["group_id"].fillna(df["sample_id"]).astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 3:
        raise ValueError("Need at least three distinct groups to create train/validation/test splits.")

    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(first.split(df, groups=groups))
    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    train_val_groups = train_val["group_id"].fillna(train_val["sample_id"]).astype(str)
    second = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=seed)
    train_idx, val_idx = next(second.split(train_val, groups=train_val_groups))
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)

def run_quality_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(df)
    df["text"] = df["text"].map(normalize_text)
    empty_count = int((df["text"] == "").sum())
    df = df[df["text"] != ""].copy()

    # Normalize for exact duplicate detection.
    df["normalized_text_hash"] = df["text"].str.lower().map(stable_hash)
    conflict_hashes = (
        df.groupby("normalized_text_hash")["label"].nunique().loc[lambda x: x > 1].index.tolist()
    )
    conflict_rows = int(df["normalized_text_hash"].isin(conflict_hashes).sum())
    # Remove all conflicting exact-text groups rather than choosing a label arbitrarily.
    if conflict_hashes:
        df = df[~df["normalized_text_hash"].isin(conflict_hashes)].copy()

    # Exact duplicate texts with same label: retain first and preserve the provenance of that row.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["normalized_text_hash"], keep="first").copy()
    duplicates_removed = before_dedup - len(df)

    report = {
        "rows_before_checks": before,
        "empty_text_rows_removed": empty_count,
        "conflicting_label_rows_removed": conflict_rows,
        "same_label_exact_duplicates_removed": duplicates_removed,
        "rows_after_checks": len(df),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "source_counts": {str(k): int(v) for k, v in df["source_dataset"].value_counts().items()},
        "task_counts": {str(k): int(v) for k, v in df["source_task"].value_counts().items()},
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "group_count": int(df["group_id"].nunique()),
        "warning": (
            "Exact-text deduplication does not detect semantic/near-duplicate paraphrases. "
            "Group leakage is only controlled where source group IDs are available; "
            "otherwise exact normalized text is used as the group."
        ),
    }
    return df, report

def export_category_csvs(df: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filters = {
        "mvgr_ai_paraphrased.csv": df["text_type"].eq("paraphrased"),
        "mvgr_ai_deeply_paraphrased.csv": df["text_type"].eq("deep_paraphrased"),
        "mvgr_ai_ai.csv": df["label"].eq(1),
        "mvgr_ai_human.csv": df["label"].eq(0),
    }
    counts = {}
    for filename, mask in filters.items():
        subset = df.loc[mask]
        subset.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
        counts[filename] = len(subset)
    return counts

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare BiScope + PADBen for MVGR-AI.")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect schemas; do not download full datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group-aware splitting.")
    args = parser.parse_args()
    ensure_dirs()

    print("=" * 68)
    print("MVGR-AI DATASET PREPARATION")
    print("=" * 68)
    print("Label convention: 0 = Human-authored, 1 = AI-authored")
    print("Paraphrasing does not change the authorship label.")

    print("\n[1/4] Inspecting available dataset schemas...")
    schema = discover_schema()
    write_schema_report(schema)
    print(f"Schema report saved: {OUTPUT_DIR / 'dataset_schema_report.txt'}")
    if args.inspect_only:
        print("\nInspection complete. Review the schema report before preparation.")
        return 0

    print("\n[2/4] Downloading and normalizing selected datasets...")
    df = load_and_normalize()

    print("\n[3/4] Assigning group-aware train/validation/test splits...")
    df = assign_group_splits(df, seed=args.seed)

    print("\n[4/4] Running quality checks and exporting...")
    df, quality = run_quality_checks(df)

    # Remove internal helper hash from the public CSV schema.
    df = df.drop(columns=["normalized_text_hash"], errors="ignore")
    columns = [
        "sample_id", "text", "label", "source_dataset", "source_task",
        "text_type", "paraphrase_status", "domain", "ai_model",
        "padben_split", "split", "group_id", "source_row_id"
    ]
    df = df[columns].sort_values(["split", "source_dataset", "source_task"]).reset_index(drop=True)

    csv_path = PROCESSED_DIR / "mvgr_ai_combined.csv"
    parquet_path = PROCESSED_DIR / "mvgr_ai_combined.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)

    for split_name in ("train", "validation", "test"):
        df[df["split"] == split_name].to_csv(
            PROCESSED_DIR / f"mvgr_ai_{split_name}.csv", index=False, encoding="utf-8-sig"
        )
    category_counts = export_category_csvs(df)

    (OUTPUT_DIR / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    df.groupby(["source_dataset", "label"]).size().unstack(fill_value=0).to_csv(
        OUTPUT_DIR / "source_distribution.csv"
    )
    df.groupby(["split", "label"]).size().unstack(fill_value=0).to_csv(
        OUTPUT_DIR / "label_distribution.csv"
    )

    print("\nSUCCESS")
    print(f"Total usable rows: {len(df):,}")
    print(f"CSV:     {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Report:  {OUTPUT_DIR / 'data_quality_report.json'}")
    for filename, count in category_counts.items():
        print(f"{filename}: {count:,} rows")
    print("\nReview the quality report and verify task/label semantics before model training.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nDATASET PREPARATION STOPPED")
        print(f"Reason: {exc}")
        print("No ambiguous label mapping is guessed. Check outputs/dataset_schema_report.json.")
        raise
