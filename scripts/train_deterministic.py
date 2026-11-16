"""Train and evaluate the deterministic AI-text detector on ``10000.csv``.

Usage
-----
.. code-block:: powershell

    # everything (audit -> train -> evaluate -> ablation -> robustness -> reports)
    .\\.venv\\Scripts\\python.exe scripts\\train_deterministic.py --stage all

    # a single stage
    .\\.venv\\Scripts\\python.exe scripts\\train_deterministic.py --stage audit
    .\\.venv\\Scripts\\python.exe scripts\\train_deterministic.py --stage train

Order of operations, and why it is that order
---------------------------------------------
1. **Audit** the dataset and resolve the leakage-safe split.
2. **Align** the extracted features to the dataset (every id must match).
3. **Select features on the training rows only.** The shipped 424-column set
   was selected with the whole 500-row pilot, so it is contaminated; a
   train-only selection replaces it for these experiments.
4. **Fit the preprocessing on the training rows only.**
5. **Tune** each model with group-aware cross-validation on the training
   split, then refit on the training split and score on **validation**.
6. **Choose** the best model by validation ROC-AUC.
7. **Evaluate** every model once on the untouched **test** split.
8. **Ablate** the feature families, **stress-test** paraphrase robustness and
   **interpret** the winning model, all with the same split and metrics.

The test split is read exactly once, in step 7, and never influences
anything upstream of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detection.dataset_loader import (  # noqa: E402
    ID_COLUMN,
    LABEL_AI,
    LABEL_HUMAN,
    load_bundle,
)
from detection.evaluate_models import (  # noqa: E402
    evaluate_model,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    results_frame,
    write_results,
)
from detection.feature_ablation import build_ablation_sets, run_ablation  # noqa: E402
from detection.feature_alignment import (  # noqa: E402
    align_features,
    coverage_report,
    select_features_train_only,
)
from detection.interpretability import (  # noqa: E402
    error_profile,
    family_summary,
    logistic_coefficients,
    random_forest_importances,
    shap_values,
    xgboost_importances,
)
from detection.predictor import DetectorBundle, save_bundle, save_metadata  # noqa: E402
from detection.preprocessing import fit_preprocessor  # noqa: E402
from detection.robustness_evaluation import (  # noqa: E402
    evaluate_categories,
    robustness_table,
    summarise_robustness,
)
from detection.train_models import (  # noqa: E402
    fit_final,
    model_specs,
    positive_probability,
    run_hyperparameter_search,
)
from shared.paths import load_config, resolve  # noqa: E402

CONFIG_NAME = "deterministic_config"
STAGES = (
    "split",
    "audit",
    "train",
    "ablation",
    "robustness",
    "interpretability",
    "report",
    "all",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _model_size_bytes(estimator: Any) -> int:
    """Approximate the in-memory size of a fitted estimator.

    The estimator's own numpy arrays are summed first. Ensembles such as
    XGBoost and Random Forest keep their trees inside opaque C/Cython objects
    that expose no numpy arrays, so the walk returns almost nothing for them;
    in that case the pickled size is used instead, which is a practical proxy
    for the in-memory footprint. This is not an exact RSS measurement.
    """
    total = 0
    seen: set[int] = set()

    def walk(obj: Any, depth: int = 0) -> None:
        nonlocal total
        if depth > 3 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            total += int(obj.nbytes)
            return
        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value, depth + 1)

    walk(estimator)
    try:
        import pickle

        pickled = len(pickle.dumps(estimator))
    except Exception:  # noqa: BLE001 - a non-picklable estimator keeps the walk
        pickled = 0
    return max(total, pickled)


def _estimator_arrays(estimator: Any) -> int:
    """Number of numeric arrays held by a fitted estimator."""
    count = 0
    seen: set[int] = set()

    def walk(obj: Any, depth: int = 0) -> None:
        nonlocal count
        if depth > 3 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            count += 1
            return
        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value, depth + 1)

    walk(estimator)
    return count


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars to plain Python types for JSON output."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _delta(value: Any) -> str:
    """Format a signed difference, or ``n/a`` when it is undefined."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:+.4f}"


def _write_json(payload: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    return str(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    """Write a CSV, creating the parent directory if it is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _write_text(text: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return str(path)


def _markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if frame.empty:
        return "_(no rows)_\n"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else floatfmt.format(value)
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    header = "| " + " | ".join(str(c) for c in display.columns) + " |"
    divider = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in display.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows]) + "\n"


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
class SplitData:
    """One split's rows, labels, groups and metadata — always in the same order."""

    def __init__(
        self,
        frame: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        metadata: pd.DataFrame,
    ):
        self.frame = frame
        self.labels = labels
        self.groups = groups
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.frame)


class ExperimentData:
    """The aligned, split and selected feature matrices."""

    def __init__(
        self,
        bundle: Any,
        aligned: pd.DataFrame,
        selected_columns: list[str],
        selection_report: dict[str, Any],
        coverage: dict[str, Any],
        watermark: pd.DataFrame | None,
    ):
        self.bundle = bundle
        self.aligned = aligned
        self.selected_columns = selected_columns
        self.selection_report = selection_report
        self.coverage = coverage
        self.watermark = watermark

    # ------------------------------------------------------------------ #
    def split_data(self, split: str) -> SplitData:
        """Return one split with the features, labels, groups and metadata aligned.

        Everything is derived from the same ordered frame, so a row's feature
        vector, label, group and metadata can never drift apart.
        """
        part = self.bundle.split_frame(split).reset_index(drop=True)
        ids = set(part[ID_COLUMN].astype(str))
        frame = self.aligned[self.aligned[ID_COLUMN].astype(str).isin(ids)].reset_index(
            drop=True
        )
        # Re-order the feature frame to follow the dataset exactly.
        order = {sample_id: index for index, sample_id in enumerate(part[ID_COLUMN].astype(str))}
        frame = (
            frame.assign(_order=frame[ID_COLUMN].astype(str).map(order))
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )
        labels = (
            pd.to_numeric(part["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        )
        groups = part["group_id"].astype(str).to_numpy() if "group_id" in part.columns else None
        keep = [
            column
            for column in (
                "sample_id",
                "label",
                "text_type",
                "paraphrase_status",
                "source_dataset",
            )
            if column in part.columns
        ]
        return SplitData(frame, labels, groups, part[keep].reset_index(drop=True))

    def frame(self, split: str) -> pd.DataFrame:
        return self.split_data(split).frame

    def labels(self, split: str) -> np.ndarray:
        return self.split_data(split).labels

    def groups(self, split: str) -> np.ndarray:
        return self.split_data(split).groups

    def metadata(self, split: str) -> pd.DataFrame:
        return self.split_data(split).metadata


def prepare(config: dict[str, Any], *, require_features: bool = True) -> ExperimentData:
    """Load the dataset, align the features and select on the training rows."""
    print("[det] loading dataset ...", flush=True)
    bundle = load_bundle(
        REPO_ROOT / config["dataset"]["file"],
        manifest_dir=resolve(config["split"]["manifest_dir"]),
        manifest_id=config["split"]["manifest"],
    )
    print(
        f"[det] dataset {bundle.dataset_file}: {bundle.audit['rows']} rows, "
        f"{bundle.audit['label_human']} human / {bundle.audit['label_ai']} AI, "
        f"split {bundle.counts()}",
        flush=True,
    )

    fused_path = resolve(config["features"]["fused"])
    if not fused_path.exists():
        if require_features:
            raise SystemExit(
                f"[det] {fused_path} does not exist; run the fingerprinting "
                "pipeline for the 10,000-row dataset first"
            )
        # The split stage only needs the dataset and the manifest.
        empty = pd.DataFrame({"sample_id": bundle.frame[ID_COLUMN].astype(str)})
        return ExperimentData(
            bundle=bundle,
            aligned=empty,
            selected_columns=[],
            selection_report={},
            coverage={},
            watermark=None,
        )

    features = pd.read_csv(fused_path, dtype={ID_COLUMN: str})
    coverage = coverage_report(features, bundle.frame)
    print(
        f"[det] feature coverage: {coverage['covered_records']}/{coverage['dataset_records']} "
        f"records, {coverage['feature_columns']} columns, "
        f"missing fraction {coverage['missing_fraction_in_covered_rows']:.4f}",
        flush=True,
    )

    aligned = align_features(features, bundle.frame)
    train_frame = aligned[
        aligned[ID_COLUMN].isin(
            set(bundle.frame.loc[bundle.frame["model_split"] == "train", ID_COLUMN])
        )
    ].reset_index(drop=True)

    selection_config = {
        key: value
        for key, value in config["selection"].items()
        if key in (
            "max_missing_fraction",
            "constant_feature_tolerance",
            "high_correlation_threshold",
            "drop_constant",
            "drop_high_correlation",
        )
    }
    selected_columns, selection_report = select_features_train_only(
        train_frame, **selection_config
    )
    print(
        f"[det] train-only selection: {selection_report['candidate_features']} candidates -> "
        f"{len(selected_columns)} selected "
        f"(missing {len(selection_report['dropped_missing'])}, "
        f"constant {len(selection_report['dropped_constant'])}, "
        f"correlated {len(selection_report['dropped_correlated'])})",
        flush=True,
    )

    watermark = None
    watermark_path = resolve(config["features"]["watermark"])
    if watermark_path.exists():
        watermark = pd.read_csv(watermark_path, dtype={ID_COLUMN: str})

    return ExperimentData(
        bundle=bundle,
        aligned=aligned,
        selected_columns=selected_columns,
        selection_report=selection_report,
        coverage=coverage,
        watermark=watermark,
    )


# --------------------------------------------------------------------------- #
# stage: audit
# --------------------------------------------------------------------------- #
def stage_audit(config: dict[str, Any], data: ExperimentData) -> dict[str, Any]:
    """Write the dataset and feature analysis reports."""
    report_dir = resolve(config["output"]["report_dir"])
    artifact_dir = resolve(config["output"]["artifact_dir"])
    audit = data.bundle.audit
    split = data.bundle.split_report

    _write_json(audit, artifact_dir / "dataset_audit.json")
    _write_json(split, artifact_dir / "split_integrity.json")
    _write_json(data.coverage, artifact_dir / "feature_coverage.json")
    _write_json(data.selection_report, artifact_dir / "feature_selection.json")

    # An explicit split table, so the hybrid integration can align on it
    # without re-deriving the assignment.
    split_table = data.bundle.frame[
        [
            column
            for column in (
                "sample_id",
                "model_split",
                "group_id",
                "label",
                "text_type",
                "paraphrase_status",
                "source_dataset",
            )
            if column in data.bundle.frame.columns
        ]
    ].copy()
    _write_csv(split_table, artifact_dir / "split_assignment.csv")

    lines = [
        "# Dataset analysis — `10000.csv`",
        "",
        f"- Generated: {_stamp()}",
        f"- Dataset file: `{audit['dataset_file']}`",
        f"- Resolved path: `{audit['dataset_path']}`",
        f"- SHA-256: `{audit['dataset_sha256']}`",
        f"- Rows in the CSV: **{audit['rows']}**",
        f"- Valid records (non-empty text, parsable label): **{audit['valid_records']}**",
        f"- Invalid records: **{audit['invalid_records']}**"
        + (
            f" — {audit['invalid_reasons']}" if audit["invalid_reasons"] else " — none"
        ),
        "",
        "## Label distribution (`0 = Human`, `1 = AI`)",
        "",
        f"- Human: **{audit['label_human']}**",
        f"- AI: **{audit['label_ai']}**",
        f"- AI/Human imbalance ratio: {audit['imbalance_ratio']}",
        "",
        "## Composition",
        "",
        f"- Source dataset: {audit['source_dataset_counts']}",
        f"- Text type: {audit['text_type_counts']}",
        f"- Paraphrase status: {audit['paraphrase_status_counts']}",
        f"- Graph category: {audit['graph_category_counts']}",
        f"- Domain: {audit['domain_counts']}",
        f"- AI model: {audit['ai_model_counts']}",
        "",
        "## Data quality",
        "",
        f"- Duplicate `sample_id`: {audit['duplicate_sample_ids']}",
        f"- Duplicate texts (exact): {audit['duplicate_texts_exact']}",
        f"- Duplicate texts (whitespace-normalised): {audit['duplicate_texts_normalized']}",
        f"- Text length (chars): min {audit['text_length_chars']['min']}, "
        f"median {audit['text_length_chars']['median']}, "
        f"max {audit['text_length_chars']['max']}",
        "",
        "### Missing values per column (empty strings)",
        "",
        _markdown_table(
            pd.DataFrame(
                [{"column": k, "empty": v} for k, v in audit["missing_values_by_column"].items()]
            )
        ),
        "",
        "## Paraphrase families",
        "",
        f"- Family key: `{audit['families']['key']}`",
        f"- Unique families: {audit['families']['unique']}",
        f"- Families with more than one member: "
        f"{audit['families']['families_with_multiple_members']} "
        f"({audit['families']['rows_in_multi_member_families']} rows)",
        f"- Families containing both a human and an AI version: "
        f"{audit['families']['families_spanning_multiple_labels']}",
        "",
        "## Split",
        "",
        f"The split column shipped in the CSV disagrees with the repository's candidate "
        f"manifest, so the resolved split is built as follows: the "
        f"`{data.bundle.manifest_id}` manifest is reused for every row it covers "
        f"(PADBen), and the remaining rows (BiScope, which the source-scoped manifest "
        f"does not cover) are assigned with the repository's own family-disjoint "
        f"policy, the same seed and the same ratios.",
        "",
        f"- Resolved counts: {split['counts']}",
        f"- Families spanning more than one split: {split['families_spanning_multiple_splits']}",
        f"- Duplicate texts spanning more than one split: "
        f"{split['duplicate_texts_spanning_splits']}",
        f"- Leakage-free: **{split['leakage_free']}**",
        "",
        "### Split composition",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "split": split_name,
                        "label": label,
                        "n": count,
                    }
                    for split_name, labels in split["counts_by_label"].items()
                    for label, count in labels.items()
                ]
            )
        ),
        "",
        "### Split composition by source dataset",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"split": split_name, "source": source, "n": count}
                    for split_name, sources in split["counts_by_source"].items()
                    for source, count in sources.items()
                ]
            )
        ),
        "",
        "### Split composition by text type",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"split": split_name, "text_type": text_type, "n": count}
                    for split_name, types in split["counts_by_text_type"].items()
                    for text_type, count in types.items()
                ]
            )
        ),
        "",
        "The split shipped inside the CSV (`split` column) was also checked: "
        f"{audit['source_csv_families_spanning_multiple_splits']} families span more than "
        "one of its splits. It is group-safe too, but it does not follow the "
        "repository's split policy, so it is not used for these experiments.",
        "",
    ]
    _write_text("\n".join(lines), report_dir / "dataset_analysis.md")
    print(f"[det] wrote {report_dir / 'dataset_analysis.md'}", flush=True)
    return {"audit": audit, "split": split, "coverage": data.coverage}


# --------------------------------------------------------------------------- #
# stage: split
# --------------------------------------------------------------------------- #
def stage_split(config: dict[str, Any], data: ExperimentData) -> dict[str, Any]:
    """Write the resolved split assignment.

    The split depends only on the dataset and the manifest, never on the
    features, so it can be resolved **before** the extraction. The writeprints
    vocabulary is a learned step and is therefore fitted on the training rows
    only — it needs this assignment to exist first.
    """
    artifact_dir = resolve(config["output"]["artifact_dir"])
    split_table = data.bundle.frame[
        [
            column
            for column in (
                "sample_id",
                "model_split",
                "group_id",
                "label",
                "text_type",
                "paraphrase_status",
                "source_dataset",
            )
            if column in data.bundle.frame.columns
        ]
    ].copy()
    _write_csv(split_table, artifact_dir / "split_assignment.csv")
    _write_json(data.bundle.split_report, artifact_dir / "split_integrity.json")
    _write_json(data.bundle.audit, artifact_dir / "dataset_audit.json")
    print(
        f"[det] wrote {artifact_dir / 'split_assignment.csv'} "
        f"({data.bundle.counts()})",
        flush=True,
    )
    return {"counts": data.bundle.counts()}


# --------------------------------------------------------------------------- #
# stage: feature analysis
# --------------------------------------------------------------------------- #
def stage_feature_analysis(config: dict[str, Any], data: ExperimentData) -> dict[str, Any]:
    """Describe the extracted features and the train-only selection."""
    report_dir = resolve(config["output"]["report_dir"])
    artifact_dir = resolve(config["output"]["artifact_dir"])

    train_data = data.split_data("train")
    train_frame = train_data.frame
    labels = train_data.labels

    # Per-family view of the fused matrix and of the train-only selection.
    families = ["stylometric", "writeprints", "gltr"]
    selected_by_family = {
        family: [
            column for column in data.selected_columns if column.startswith(f"{family}__")
        ]
        for family in families
    }
    candidates_by_family = {
        family: [
            column
            for column in train_frame.columns
            if column.startswith(f"{family}__")
        ]
        for family in families
    }
    dropped = {
        key: value
        for key, value in data.selection_report.items()
        if key.startswith("dropped_")
    }

    rows = []
    for family in families:
        candidates = candidates_by_family[family]
        selected = selected_by_family[family]
        missing = train_frame[candidates].isna().mean() if candidates else pd.Series(dtype=float)
        rows.append(
            {
                "family": family,
                "candidates": len(candidates),
                "selected": len(selected),
                "dropped": len(candidates) - len(selected),
                "max_missing_fraction": round(float(missing.max()), 6) if len(missing) else 0.0,
                "mean_missing_fraction": round(float(missing.mean()), 6) if len(missing) else 0.0,
            }
        )
    family_frame = pd.DataFrame(rows)

    # Univariate signal, computed on the training rows only.
    from sklearn.metrics import roc_auc_score

    univariate = []
    for column in data.selected_columns:
        values = train_frame[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 20 or len(np.unique(values[finite])) < 2:
            continue
        filled = np.where(finite, values, np.nanmedian(values[finite]))
        try:
            auc = float(roc_auc_score(labels, filled))
        except Exception:  # noqa: BLE001
            continue
        univariate.append(
            {
                "feature": column,
                "family": column.split("__", 1)[0],
                "univariate_auc": round(auc, 6),
                "direction": "AI" if auc >= 0.5 else "Human",
                "separation": round(abs(auc - 0.5) * 2, 6),
            }
        )
    univariate_frame = pd.DataFrame(univariate).sort_values(
        "separation", ascending=False
    )
    _write_csv(univariate_frame, artifact_dir / "feature_analysis" / "univariate_auc.csv")

    # How different is the train-only selection from the contaminated one?
    # The shipped feature matrix was selected with the whole table, so this
    # comparison quantifies what that contamination actually changed.
    all_columns = [
        column
        for column in data.aligned.columns
        if column != ID_COLUMN
        and pd.api.types.is_numeric_dtype(data.aligned[column])
    ]
    contaminated_columns, contaminated_report = select_features_train_only(
        data.aligned[[ID_COLUMN, *all_columns]],
        max_missing_fraction=float(config["selection"]["max_missing_fraction"]),
        constant_feature_tolerance=float(
            config["selection"]["constant_feature_tolerance"]
        ),
        high_correlation_threshold=float(
            config["selection"]["high_correlation_threshold"]
        ),
        drop_constant=bool(config["selection"]["drop_constant"]),
        drop_high_correlation=bool(config["selection"]["drop_high_correlation"]),
    )
    train_only = set(data.selected_columns)
    contaminated = set(contaminated_columns)
    selection_audit = {
        "train_only_features": len(train_only),
        "whole_dataset_features": len(contaminated),
        "only_in_train_only": sorted(train_only - contaminated),
        "only_in_whole_dataset": sorted(contaminated - train_only),
        "identical": train_only == contaminated,
    }
    _write_json(
        selection_audit, artifact_dir / "feature_analysis" / "selection_audit.json"
    )

    top = univariate_frame.head(25)
    lines = [
        "# Feature analysis",
        "",
        f"- Generated: {_stamp()}",
        f"- Dataset: `{config['dataset']['file']}` "
        f"({data.bundle.audit['rows']} rows, {data.bundle.audit['valid_records']} valid)",
        f"- Feature coverage: {data.coverage['covered_records']}/"
        f"{data.coverage['dataset_records']} "
        f"records, {data.coverage['feature_columns']} candidate columns, "
        f"missing fraction {data.coverage['missing_fraction_in_covered_rows']:.6f}",
        f"- Features used after the train-only selection: **{len(data.selected_columns)}**",
        "",
        "## The four branches",
        "",
        "| Branch | Features | Role |",
        "|---|---:|---|",
        "| Stylometric | 140 | vocabulary, sentence, punctuation, POS, syntactic, readability |",
        "| Writeprints | 301 | bounded char/word/POS n-grams, function words |",
        "| GLTR | 19 | real GPT-2 next-token probability/rank/entropy/perplexity |",
        "| Watermark | 12 | **evidence only** — never a model feature |",
        "",
        "The watermark branch is not part of the feature matrix: it is written to "
        "the metadata table, so an unsupported detection can never become an "
        "artificial negative class.",
        "",
        "## Train-only selection, per family",
        "",
        "The selection rules are the ones the fusion layer uses (missing-value "
        "threshold, constant columns, correlation filtering), but they are "
        "fitted on the **training rows only** and then applied to validation and "
        "test.",
        "",
        _markdown_table(family_frame),
        "",
        f"- Candidates: {data.selection_report['candidate_features']}",
        f"- Selected: {len(data.selected_columns)}",
        f"- Dropped (missing): {len(dropped.get('dropped_missing', []))}",
        f"- Dropped (constant): {len(dropped.get('dropped_constant', []))}",
        f"- Dropped (correlated): {len(dropped.get('dropped_correlated', []))}",
        "",
        "## Univariate signal (training rows only)",
        "",
        "The table below is the ROC-AUC of each selected feature on its own, "
        "computed on the training split. A value of 0.5 means the feature "
        "carries no directional signal; the `separation` column is the distance "
        "from chance, so it ranks the features the same way for both directions.",
        "",
        _markdown_table(top),
        "",
        f"- {len(univariate_frame)} selected features have a computable univariate AUC.",
        "- Univariate AUC ignores interactions and correlations; it is a "
        "description of the feature, not a model.",
        "",
        "## Audit of the shipped selection",
        "",
        "The feature matrix shipped by the fingerprinting pipeline selects its "
        "columns with the whole table it is given, so the selection has seen "
        "the evaluation rows. The table below re-runs the identical rules over "
        "**all** rows and compares the result with the train-only selection "
        "used for these experiments.",
        "",
        f"- Train-only selection: **{selection_audit['train_only_features']}** features",
        f"- Whole-dataset selection (contaminated): "
        f"**{selection_audit['whole_dataset_features']}** features",
        f"- Identical: **{selection_audit['identical']}**",
        f"- Only in the train-only selection: {len(selection_audit['only_in_train_only'])}"
        + (
            f" — {selection_audit['only_in_train_only'][:10]}"
            if selection_audit["only_in_train_only"]
            else ""
        ),
        f"- Only in the whole-dataset selection: {len(selection_audit['only_in_whole_dataset'])}"
        + (
            f" — {selection_audit['only_in_whole_dataset'][:10]}"
            if selection_audit["only_in_whole_dataset"]
            else ""
        ),
        "",
        "The original artifacts are kept unchanged, so the contaminated and the "
        "train-only selections can be compared directly.",
        "",
    ]
    _write_text("\n".join(lines), report_dir / "feature_analysis.md")
    print(f"[det] wrote {report_dir / 'feature_analysis.md'}", flush=True)
    return {"univariate": len(univariate_frame)}


# --------------------------------------------------------------------------- #
# stage: train
# --------------------------------------------------------------------------- #
def stage_train(config: dict[str, Any], data: ExperimentData) -> dict[str, Any]:
    """Tune, train and evaluate every model."""
    model_dir = resolve(config["output"]["model_dir"])
    artifact_dir = resolve(config["output"]["artifact_dir"])
    report_dir = resolve(config["output"]["report_dir"])

    train_frame = data.frame("train")
    valid_frame = data.frame("validation")
    test_frame = data.frame("test")
    y_train = data.labels("train")
    y_valid = data.labels("validation")
    y_test = data.labels("test")
    groups_train = data.groups("train")

    print(
        f"[det] train {len(train_frame)} | validation {len(valid_frame)} | test {len(test_frame)}",
        flush=True,
    )

    preprocessor = fit_preprocessor(
        train_frame,
        data.selected_columns,
        scale=bool(config["preprocessing"].get("scale", True)),
    )
    X_train = preprocessor.transform(train_frame)
    X_valid = preprocessor.transform(valid_frame)
    X_test = preprocessor.transform(test_frame)

    specs = model_specs(int(config["tuning"].get("model_jobs", -1)))
    enabled = {
        name: spec
        for name, spec in specs.items()
        if config["models"].get(name, {}).get("enabled", True)
    }
    print(f"[det] models: {sorted(enabled)}", flush=True)

    search_results: list[dict[str, Any]] = []
    fitted: dict[str, dict[str, Any]] = {}

    for name, spec in enabled.items():
        n_iter = int(config["tuning"]["n_iter"].get(name, 10))
        print(f"[det] tuning {spec.label} (n_iter={n_iter}) ...", flush=True)
        search = run_hyperparameter_search(
            spec,
            X_train,
            y_train,
            groups=groups_train,
            n_splits=int(config["tuning"]["cv_splits"]),
            n_iter=n_iter,
            seed=int(config["tuning"]["seed"]),
            n_jobs=int(config["tuning"]["n_jobs"]),
            verbose=int(config["tuning"].get("verbose", 0)),
        )
        print(
            f"[det]   best cv roc_auc {search.best_score:.4f} "
            f"({search.seconds:.1f}s, {search.n_candidates} candidates)",
            flush=True,
        )
        search_results.append(search.to_dict())
        _write_json(
            search.cv_results, artifact_dir / "tuning" / f"{name}_candidates.json"
        )
        _write_json(
            {k: _jsonable(v) for k, v in search.best_params.items()},
            artifact_dir / "tuning" / f"{name}_best_params.json",
        )

        estimator = fit_final(spec, X_train, y_train, search.best_params)
        if getattr(spec, "probability_strategy", "native") == "margin_calibrated":
            # The margin scores are calibrated on the validation split, which
            # exists exactly for model selection. The test split is untouched.
            estimator.fit_calibrator(X_valid, y_valid)
        fit_seconds = search.seconds
        validation = evaluate_model(
            name, spec.label, estimator, X_valid, y_valid, split="validation"
        )
        print(
            f"[det]   validation acc {validation.metrics['accuracy']:.4f} "
            f"bal.acc {validation.metrics['balanced_accuracy']:.4f} "
            f"roc_auc {validation.metrics['roc_auc']:.4f}",
            flush=True,
        )
        fitted[name] = {
            "spec": spec,
            "estimator": estimator,
            "params": search.best_params,
            "preprocessor": preprocessor,
            "validation": validation,
            "fit_seconds": fit_seconds,
            "search_seconds": search.seconds,
        }

    # Model selection uses the VALIDATION split only.
    best_name = max(
        fitted,
        key=lambda key: (
            fitted[key]["validation"].metrics.get("roc_auc")
            or fitted[key]["validation"].metrics["balanced_accuracy"]
        ),
    )
    print(f"[det] best model by validation ROC-AUC: {best_name}", flush=True)

    # The test split is read here, once, and never influences anything above.
    test_results = []
    predictions: dict[str, pd.DataFrame] = {}
    resources: dict[str, dict[str, Any]] = {}
    for name, entry in fitted.items():
        spec = entry["spec"]
        predict_started = time.perf_counter()
        result = evaluate_model(
            name, spec.label, entry["estimator"], X_test, y_test, split="test"
        )
        predict_seconds = time.perf_counter() - predict_started
        test_results.append(result)
        probabilities = positive_probability(entry["estimator"], X_test)
        predictions[name] = pd.DataFrame(
            {
                "sample_id": test_frame[ID_COLUMN].tolist(),
                "label": y_test,
                "prediction": result.y_pred,
                "ai_probability": probabilities,
            }
        )
        resources[name] = {
            "training_seconds": round(float(entry["fit_seconds"]), 3),
            "search_seconds": round(float(entry["search_seconds"]), 3),
            "prediction_seconds_test": round(float(predict_seconds), 3),
            "model_memory_bytes": _model_size_bytes(entry["estimator"]),
            "model_arrays": _estimator_arrays(entry["estimator"]),
        }
        print(
            f"[det] test {spec.label}: acc {result.metrics['accuracy']:.4f} "
            f"bal.acc {result.metrics['balanced_accuracy']:.4f} "
            f"f1 {result.metrics['f1']:.4f} roc_auc {result.metrics['roc_auc']:.4f}",
            flush=True,
        )

    # Persist every model.
    metadata_common = {
        "dataset_file": config["dataset"]["file"],
        "dataset_rows": data.bundle.audit["rows"],
        "valid_records": data.bundle.audit["valid_records"],
        "training_samples": int(len(train_frame)),
        "validation_samples": int(len(valid_frame)),
        "test_samples": int(len(test_frame)),
        "feature_count": len(data.selected_columns),
        "feature_names": list(data.selected_columns),
        "label_convention": {"human": LABEL_HUMAN, "ai": LABEL_AI},
        "preprocessing": {
            "imputation": "median (train only)",
            "scaling": (
                "standard (train only)"
                if config["preprocessing"].get("scale", True)
                else "none"
            ),
            "dropped_columns": list(preprocessor.dropped_columns),
        },
        "split": {
            "column": "model_split",
            "manifest": config["split"]["manifest"],
            "counts": data.bundle.counts(),
            "selection": "validation ROC-AUC",
        },
        "trained_at": _stamp(),
        "seed": int(config["tuning"]["seed"]),
    }
    for name, entry in fitted.items():
        spec = entry["spec"]
        metadata = {
            **metadata_common,
            "model_type": spec.label,
            "model_name": name,
            "best_hyperparameters": {k: str(v) for k, v in entry["params"].items()},
            "validation_metrics": {
                k: round(float(v), 6) for k, v in entry["validation"].metrics.items()
            },
            "test_metrics": None,
            "resources": resources.get(name, {}),
        }
        bundle = DetectorBundle(
            estimator=entry["estimator"],
            preprocessor=preprocessor,
            feature_names=list(data.selected_columns),
            model_name=name,
            model_label=spec.label,
            metadata=metadata,
        )
        target_dir = model_dir / name
        save_bundle(bundle, target_dir / "model.pkl")
        save_metadata(metadata, target_dir / "metadata.json")
        metadata["resources"]["model_file_bytes"] = (target_dir / "model.pkl").stat().st_size
        save_metadata(metadata, target_dir / "metadata.json")

    best_entry = fitted[best_name]
    best_bundle = DetectorBundle(
        estimator=best_entry["estimator"],
        preprocessor=preprocessor,
        feature_names=list(data.selected_columns),
        model_name=best_name,
        model_label=best_entry["spec"].label,
        metadata={
            **metadata_common,
            "model_type": best_entry["spec"].label,
            "model_name": best_name,
            "best_hyperparameters": {k: str(v) for k, v in best_entry["params"].items()},
            "validation_metrics": {
                k: round(float(v), 6) for k, v in best_entry["validation"].metrics.items()
            },
            "resources": resources.get(best_name, {}),
            "selected_by": "validation ROC-AUC",
        },
    )
    save_bundle(best_bundle, model_dir / "best_model.pkl")
    save_bundle(preprocessor, model_dir / "preprocessing_pipeline.pkl")
    save_metadata(best_bundle.metadata, model_dir / "model_metadata.json")
    best_bundle.metadata["resources"]["model_file_bytes"] = (
        model_dir / "best_model.pkl"
    ).stat().st_size
    save_metadata(best_bundle.metadata, model_dir / "model_metadata.json")

    written = write_results(test_results, artifact_dir)
    comparison = results_frame(test_results)
    for name, figures in resources.items():
        for key, value in figures.items():
            comparison.loc[comparison["model"] == name, key] = value
    _write_csv(comparison, artifact_dir / "model_comparison.csv")
    _write_json(
        {
            "generated_at": _stamp(),
            "best_model": best_name,
            "best_model_label": best_entry["spec"].label,
            "selection_metric": "validation ROC-AUC",
            "search": search_results,
            "resources": resources,
            "test_metrics": [result.to_dict() for result in test_results],
        },
        artifact_dir / "model_comparison.json",
    )

    for name, frame in predictions.items():
        _write_csv(frame, artifact_dir / "predictions" / f"{name}_test.csv")

    # Plots.
    plot_confusion_matrices(test_results, report_dir / "figures" / "confusion_matrices.png")
    plot_roc_curves(test_results, report_dir / "figures" / "roc_curves.png")
    plot_precision_recall_curves(
        test_results, report_dir / "figures" / "precision_recall_curves.png"
    )
    plot_model_comparison(test_results, report_dir / "figures" / "model_comparison.png")

    state = {
        "best_model": best_name,
        "fitted": {name: entry for name, entry in fitted.items()},
        "test_results": test_results,
        "search": search_results,
        "written": written,
    }
    _write_json(
        {
            "generated_at": _stamp(),
            "best_model": best_name,
            "models": sorted(fitted),
            "test_metrics": [result.to_dict() for result in test_results],
            "search": search_results,
        },
        artifact_dir / "train_state.json",
    )
    return state


# --------------------------------------------------------------------------- #
# stage: ablation
# --------------------------------------------------------------------------- #
def stage_ablation(
    config: dict[str, Any], data: ExperimentData, state: dict[str, Any]
) -> dict[str, Any]:
    """Run the nine-experiment feature-family ablation."""
    artifact_dir = resolve(config["output"]["artifact_dir"])
    report_dir = resolve(config["output"]["report_dir"])

    best_name = state["best_model"]
    entry = state["fitted"][best_name]
    spec = entry["spec"]
    params = entry["params"]

    train_frame = data.frame("train")
    test_frame = data.frame("test")
    y_train = data.labels("train")
    y_test = data.labels("test")

    watermark_train = None
    watermark_test = None
    if data.watermark is not None:
        train_ids = set(
            data.bundle.frame.loc[data.bundle.frame["model_split"] == "train", ID_COLUMN]
        )
        test_ids = set(
            data.bundle.frame.loc[data.bundle.frame["model_split"] == "test", ID_COLUMN]
        )
        watermark_train = data.watermark[
            data.watermark[ID_COLUMN].isin(train_ids)
        ].reset_index(drop=True)
        watermark_test = data.watermark[
            data.watermark[ID_COLUMN].isin(test_ids)
        ].reset_index(drop=True)

    sets = build_ablation_sets(
        data.selected_columns,
        train_frame,
        watermark_frame=data.watermark,
        selection_config={
            key: value
            for key, value in config["selection"].items()
            if key
            in (
                "max_missing_fraction",
                "constant_feature_tolerance",
                "high_correlation_threshold",
                "drop_constant",
                "drop_high_correlation",
            )
        },
    )
    rows = run_ablation(
        sets,
        train_frame,
        test_frame,
        y_train,
        y_test,
        spec=spec,
        params=params,
        watermark_train=watermark_train,
        watermark_test=watermark_test,
        calibration_frame=data.frame("validation"),
        calibration_labels=data.labels("validation"),
        scale=bool(config["preprocessing"].get("scale", True)),
    )
    frame = pd.DataFrame(rows)
    _write_csv(frame, artifact_dir / "feature_ablation.csv")
    _write_json(rows, artifact_dir / "feature_ablation.json")

    lines = [
        "# Feature ablation study",
        "",
        f"- Generated: {_stamp()}",
        f"- Dataset: `{config['dataset']['file']}` "
        f"({data.bundle.audit['rows']} rows, {data.bundle.audit['valid_records']} valid)",
        f"- Reference model: **{spec.label}** (chosen by validation ROC-AUC)",
        f"- Split: train {len(data.frame('train'))} / validation {len(data.frame('validation'))} / "
        f"test {len(data.frame('test'))}",
        "- Every feature set is selected with the **training rows only**; the same "
        "split, model and metrics are used throughout.",
        "",
        _markdown_table(frame),
        "",
    ]
    _write_text("\n".join(lines), report_dir / "feature_ablation.md")
    print(f"[det] wrote {report_dir / 'feature_ablation.md'}", flush=True)
    return {"rows": rows}


# --------------------------------------------------------------------------- #
# stage: robustness
# --------------------------------------------------------------------------- #
def stage_robustness(
    config: dict[str, Any], data: ExperimentData, state: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate every trained model per text category."""
    artifact_dir = resolve(config["output"]["artifact_dir"])
    report_dir = resolve(config["output"]["report_dir"])

    best_name = state["best_model"]

    test_data = data.split_data("test")
    test_frame = test_data.frame
    y_test = test_data.labels
    metadata = test_data.metadata

    per_model: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    best_frame = pd.DataFrame()
    for name, entry in state["fitted"].items():
        estimator = entry["estimator"]
        preprocessor = entry.get("preprocessor")
        if preprocessor is None:
            from detection.preprocessing import fit_preprocessor

            preprocessor = fit_preprocessor(
                data.frame("train"),
                data.selected_columns,
                scale=bool(config["preprocessing"].get("scale", True)),
            )
        X_test = preprocessor.transform(test_frame)
        probabilities = positive_probability(estimator, X_test)
        predictions = np.asarray(estimator.predict(X_test)).astype(int)

        categories = evaluate_categories(metadata, y_test, predictions, probabilities)
        per_model[name] = [category.to_dict() for category in categories]
        summaries[name] = summarise_robustness(robustness_table(categories))
        if name == best_name:
            best_frame = robustness_table(categories)

    # The headline table is the winning model; every model's table is kept too.
    _write_csv(best_frame, artifact_dir / "paraphrase_robustness.csv")
    for name, rows in per_model.items():
        _write_csv(
            pd.DataFrame(rows),
            artifact_dir / "paraphrase_robustness" / f"{name}.csv",
        )
    _write_json(
        {
            "generated_at": _stamp(),
            "best_model": best_name,
            "categories": per_model[best_name],
            "summary": summaries[best_name],
            "per_model": {name: summaries[name] for name in summaries},
        },
        artifact_dir / "paraphrase_robustness.json",
    )

    entry = state["fitted"][best_name]
    summary = summaries[best_name]
    frame = best_frame

    lines = [
        "# Paraphrase robustness",
        "",
        f"- Generated: {_stamp()}",
        f"- Dataset: `{config['dataset']['file']}`",
        f"- Model: **{entry['spec'].label}**",
        "- All rows come from `10000.csv`; **no paraphrased text was synthesised**.",
        "",
        "## Per-category performance on the test split",
        "",
        _markdown_table(frame),
        "",
        "## Findings",
        "",
    ]
    for item in summary["categories"]:
        lines.append(
            f"- **{item['category']}** (n={item['n']}): accuracy {item['accuracy']:.4f}, "
            f"FPR {item['false_positive_rate']:.4f}, FNR {item['false_negative_rate']:.4f}"
        )
    if summary.get("degradation_vs_original"):
        lines += ["", "### Degradation relative to the original text of the same authorship", ""]
        for item in summary["degradation_vs_original"]:
            lines.append(
                f"- **{item['category']}** vs {item['reference_category']}: "
                f"accuracy {_delta(item['accuracy_delta'])}, "
                f"FPR {_delta(item['fpr_delta'])}, FNR {_delta(item['fnr_delta'])}"
            )
    lines += [
        "",
        f"- Worst false-positive rate over the human categories: "
        f"{summary.get('worst_human_false_positive_rate')}",
        f"- Worst detection rate over the AI categories: "
        f"{summary.get('worst_ai_detection_rate')}",
        "",
        "Robustness is claimed only where the numbers above support it.",
        "",
        "## All models",
        "",
        "The same breakdown for every trained model, so the robustness of the "
        "winning model can be compared with the alternatives:",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "model": name,
                        "worst_human_fpr": values.get(
                            "worst_human_false_positive_rate"
                        ),
                        "worst_ai_detection_rate": values.get(
                            "worst_ai_detection_rate"
                        ),
                        "robustness_supported": values.get(
                            "robustness_claim_supported"
                        ),
                    }
                    for name, values in summaries.items()
                ]
            )
        ),
        "",
    ]
    _write_text("\n".join(lines), report_dir / "paraphrase_robustness.md")
    print(f"[det] wrote {report_dir / 'paraphrase_robustness.md'}", flush=True)
    return {"summary": summary, "categories": per_model[best_name]}


# --------------------------------------------------------------------------- #
# stage: interpretability
# --------------------------------------------------------------------------- #
def stage_interpretability(
    config: dict[str, Any], data: ExperimentData, state: dict[str, Any]
) -> dict[str, Any]:
    """Explain the winning model and profile its errors."""
    artifact_dir = resolve(config["output"]["artifact_dir"])
    report_dir = resolve(config["output"]["report_dir"])

    best_name = state["best_model"]
    entry = state["fitted"][best_name]
    estimator = entry["estimator"]
    spec = entry["spec"]

    train_frame = data.frame("train")
    test_frame = data.frame("test")
    y_test = data.labels("test")

    from detection.preprocessing import fit_preprocessor

    preprocessor = fit_preprocessor(
        train_frame,
        data.selected_columns,
        scale=bool(config["preprocessing"].get("scale", True)),
    )
    X_test = preprocessor.transform(test_frame)
    predictions = np.asarray(estimator.predict(X_test)).astype(int)

    importances: dict[str, pd.Series] = {}
    coefficients: pd.Series | None = None
    shap_importance: pd.Series | None = None

    if best_name == "logistic_regression":
        coefficients = logistic_coefficients(estimator, data.selected_columns)
        importances["Logistic Regression (|coef|)"] = coefficients.abs().sort_values(
            ascending=False
        )
    elif best_name == "random_forest":
        series = random_forest_importances(estimator, data.selected_columns)
        importances["Random Forest"] = series
    elif best_name == "xgboost":
        series = xgboost_importances(estimator, data.selected_columns)
        importances["XGBoost"] = series

    shap_importance, shap_matrix = shap_values(estimator, X_test, data.selected_columns)
    if not shap_importance.empty:
        importances["SHAP (XGBoost)"] = shap_importance

    for name, series in importances.items():
        _write_csv(
            series.rename("importance").to_frame(),
            artifact_dir
            / "interpretability"
            / f"importance_{name.split()[0].lower()}.csv",
        )

    profile = error_profile(test_frame, y_test, predictions, data.selected_columns)

    top_rows: list[dict[str, Any]] = []
    for name, series in importances.items():
        for rank, (column, value) in enumerate(series.head(15).items(), start=1):
            top_rows.append(
                {"model": name, "rank": rank, "feature": column, "importance": float(value)}
            )
    top_frame = pd.DataFrame(top_rows)
    _write_csv(top_frame, artifact_dir / "interpretability" / "top_features.csv")

    _write_json(
        {
            "generated_at": _stamp(),
            "model": best_name,
            "group_counts": profile["group_counts"],
            "family_summary": {
                name: family_summary(series.head(15))
                for name, series in importances.items()
            },
            "shap_available": bool(shap_importance is not None and not shap_importance.empty),
        },
        artifact_dir / "interpretability" / "summary.json",
    )

    plot_feature_importance(
        importances, report_dir / "figures" / "feature_importance.png"
    )

    lines = [
        "# Interpretability",
        "",
        f"- Generated: {_stamp()}",
        f"- Dataset: `{config['dataset']['file']}`",
        f"- Model: **{spec.label}**",
        "- Feature importance describes *what the model used*. It is not evidence "
        "of causality, and it does not mean the feature caused the text to be "
        "machine-generated.",
        "",
        "## Most influential features",
        "",
    ]
    for name, series in importances.items():
        lines += [f"### {name}", ""]
        table = pd.DataFrame(
            {
                "rank": range(1, min(15, len(series)) + 1),
                "feature": series.head(15).index,
                "importance": series.head(15).to_numpy(),
            }
        )
        lines += [_markdown_table(table), ""]
        counts = family_summary(series.head(min(15, len(series))))
        lines += [
            f"Family mix in the top {min(15, len(series))}: {counts}",
            "",
        ]

    if coefficients is not None:
        lines += [
            "## Logistic regression coefficients (signed)",
            "",
            "A positive coefficient pushes the prediction towards AI, a negative "
            "one towards human.",
            "",
        ]
        table = pd.DataFrame(
            {
                "feature": coefficients.head(15).index,
                "coefficient": coefficients.head(15).to_numpy(),
            }
        )
        lines += [_markdown_table(table), ""]
        lines += [
            "Most human-leaning features:",
            "",
            _markdown_table(
                pd.DataFrame(
                    {
                        "feature": coefficients.tail(15).index,
                        "coefficient": coefficients.tail(15).to_numpy(),
                    }
                )
            ),
            "",
        ]

    lines += [
        "## Error profile",
        "",
        f"- Group sizes on the test split: {profile['group_counts']}",
        "",
        "Features that separate the false positives from the true negatives "
        "(largest shifts first):",
        "",
        _markdown_table(profile["largest_fp_shifts"]),
        "",
        "Features that separate the false negatives from the true positives:",
        "",
        _markdown_table(profile["largest_fn_shifts"]),
        "",
    ]
    _write_text("\n".join(lines), report_dir / "interpretability.md")
    print(f"[det] wrote {report_dir / 'interpretability.md'}", flush=True)
    return {"group_counts": profile["group_counts"]}


# --------------------------------------------------------------------------- #
# stage: report
# --------------------------------------------------------------------------- #
def stage_report(
    config: dict[str, Any], data: ExperimentData, state: dict[str, Any] | None
) -> dict[str, Any]:
    """Write the model-comparison and final-evaluation reports."""
    artifact_dir = resolve(config["output"]["artifact_dir"])
    report_dir = resolve(config["output"]["report_dir"])

    comparison_path = artifact_dir / "model_comparison.csv"
    if not comparison_path.exists():
        return {}
    comparison = pd.read_csv(comparison_path)

    lines = [
        "# Model comparison",
        "",
        f"- Generated: {_stamp()}",
        f"- Dataset: `{config['dataset']['file']}` — "
        f"{data.bundle.audit['rows']} rows, {data.bundle.audit['valid_records']} valid "
        f"({data.bundle.audit['label_human']} human / {data.bundle.audit['label_ai']} AI)",
        f"- Split: train {len(data.frame('train'))} / "
        f"validation {len(data.frame('validation'))} / test {len(data.frame('test'))}",
        f"- Features: **{len(data.selected_columns)}** "
        "(selected on the training rows only)",
        "- The test split was read once, after model selection on validation.",
        "",
        "## Test-set metrics",
        "",
        _markdown_table(comparison),
        "",
        "The dataset is roughly 78% AI, so a detector that always answers "
        "\"AI\" would score about 0.785 accuracy. **Balanced accuracy**, the "
        "false-positive rate on human text and the false-negative rate on AI "
        "text are the numbers that decide whether a model is useful.",
        "",
    ]
    _write_text("\n".join(lines), report_dir / "model_comparison.md")

    final = [
        "# Final evaluation — deterministic AI-text detector",
        "",
        f"- Generated: {_stamp()}",
        f"- Dataset: **`{config['dataset']['file']}`** "
        f"({data.bundle.audit['rows']} rows; {data.bundle.audit['valid_records']} valid records; "
        f"{data.bundle.audit['label_human']} human, {data.bundle.audit['label_ai']} AI)",
        f"- Feature coverage: {data.coverage['covered_records']}/"
        f"{data.coverage['dataset_records']} records, "
        f"{data.coverage['feature_columns']} candidate columns",
        f"- Features used: **{len(data.selected_columns)}** (train-only selection)",
        f"- Split: train {len(data.frame('train'))} / validation {len(data.frame('validation'))} / "
        f"test {len(data.frame('test'))} (leakage-free, family-disjoint)",
        "",
        "## Test-set metrics",
        "",
        _markdown_table(comparison),
        "",
    ]
    if state:
        best = state["best_model"]
        entry = state["fitted"][best]
        best_row = comparison[comparison["model"] == best]
        final += [
            "## Selected model",
            "",
            f"- **{entry['spec'].label}** — chosen by validation ROC-AUC",
            "- Best hyperparameters: "
            f"`{json.dumps({k: str(v) for k, v in entry['params'].items()}, indent=2)}`",
        ]
        if not best_row.empty:
            row = best_row.iloc[0]
            final += [
                f"- Test accuracy {row['accuracy']:.4f}, balanced accuracy "
                f"{row['balanced_accuracy']:.4f}, F1 {row['f1']:.4f}, "
                f"ROC-AUC {row['roc_auc']:.4f}, PR-AUC {row['pr_auc']:.4f}",
                f"- False-positive rate {row['false_positive_rate']:.4f}, "
                f"false-negative rate {row['false_negative_rate']:.4f}",
            ]
        final += [
            "",
            "## Reproduce",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\train_deterministic.py --stage all",
            "```",
            "",
            "## Limitations",
            "",
            "- GPT-2 (2019) is the reference model for the GLTR features; it is "
            "older and smaller than current generators, so absolute "
            "predictability values are reference-relative.",
            "- The watermark family cannot carry authorship signal here: the key "
            "used at generation time is unknown, so its green list is arbitrary "
            "with respect to authorship.",
            "- The writeprints n-gram vocabulary is fit on the **training split "
            "only** (6,978 rows; `fit_scope: train_split_only`), so neither the "
            "validation nor the test text influenced it.",
            "- Five training rows (all Python-code samples) have no "
            "sentence-length histogram because their single spaCy \"sentence\" "
            "exceeds the 75-word top bin; those cells are median-imputed with "
            "statistics fit on the training split.",
            "- No model was trained on the graph features; that is the other "
            "branch of MVGR-AI and is integrated separately.",
            "- Feature importance is descriptive, not causal.",
            "",
        ]
    _write_text("\n".join(final), report_dir / "final_evaluation.md")
    print(f"[det] wrote {report_dir / 'final_evaluation.md'}", flush=True)
    return {}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--config", default=CONFIG_NAME)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    data = prepare(config, require_features=args.stage != "split")

    state: dict[str, Any] = {}
    state_path = resolve(config["output"]["artifact_dir"]) / "train_state.json"
    if args.stage in ("ablation", "robustness", "interpretability", "report"):
        if not state_path.exists():
            print(f"[det] missing {state_path}; run --stage train first", file=sys.stderr)
            return 1
        with open(state_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        state = {"best_model": payload["best_model"], "fitted": _reload_fitted(config, payload)}
        state["test_results"] = []
        state["search"] = payload.get("search", [])

    started = time.perf_counter()
    if args.stage == "split":
        stage_split(config, data)
    if args.stage in ("audit", "all"):
        stage_audit(config, data)
        stage_feature_analysis(config, data)
    if args.stage in ("train", "all"):
        state = stage_train(config, data)
    if args.stage in ("ablation", "all"):
        stage_ablation(config, data, state)
    if args.stage in ("robustness", "all"):
        stage_robustness(config, data, state)
    if args.stage in ("interpretability", "all"):
        stage_interpretability(config, data, state)
    if args.stage in ("report", "all"):
        stage_report(config, data, state)

    elapsed = time.perf_counter() - started
    print(f"[det] stage '{args.stage}' finished in {elapsed:.1f}s", flush=True)
    return 0


def _reload_fitted(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Reload the fitted estimators from disk so stages can run separately."""
    import pickle

    model_dir = resolve(config["output"]["model_dir"])
    artifact_dir = resolve(config["output"]["artifact_dir"])
    specs = model_specs(int(config["tuning"].get("model_jobs", -1)))
    fitted: dict[str, Any] = {}
    for name in payload["models"]:
        path = model_dir / name / "model.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as handle:
            bundle = pickle.load(handle)
        params: dict[str, Any] = {}
        params_path = artifact_dir / "tuning" / f"{name}_best_params.json"
        if params_path.exists():
            with open(params_path, encoding="utf-8") as handle:
                params = json.load(handle)
        fitted[name] = {
            "spec": specs[name],
            "estimator": bundle.estimator,
            "params": params,
            "preprocessor": bundle.preprocessor,
            "validation": None,
        }
    return fitted


if __name__ == "__main__":
    raise SystemExit(main())
