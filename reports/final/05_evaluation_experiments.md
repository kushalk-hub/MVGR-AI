# 14–15. Experiment tracking and evaluation protocol

## 14. Experiment tracking (`shared/tracking.py`, `configs/experiment_config.yaml`)

Tracking is file-based and append-only. Every run is registered **before** any
metric exists, so a metric can never be orphaned from its configuration and
split.

### Registry

```text
experiments/
├── configs/                       # config snapshots, frozen thresholds
├── runs/
│   ├── registry.jsonl             # one line per registered experiment
│   ├── metrics.jsonl              # one line per metric log
│   └── <experiment_id>/experiment.json
├── predictions/                   # contract-valid prediction JSONL
├── checkpoints/                   # frozen rules, GNN checkpoints, bundles
└── logs/events.jsonl
```

### Lifecycle

1. `register_experiment(...)` writes `experiment.json` (validated against
   `experiment_schema.json`) and appends to `registry.jsonl`. The record carries
   the config id, dataset version, split id, graph build id, graph spec version,
   model family, seeds, and git state.
2. `update_status(...)` moves the run through
   `registered → running → completed | failed | invalidated`.
3. `log_metrics(...)` appends metrics to `metrics.jsonl`.

### Reproducibility fields

* `git_state()` records the commit and whether the tree was dirty. When git is
  unavailable it returns `{"commit": "no-git", "dirty": true}` — the run is still
  recorded, but flagged as not commit-pinned. (This repository has no git
  checkout, so all runs currently carry `no-git`.)
* Seeds come from `experiment.seeds = [13, 42, 101]` with `primary_seed = 42`.
* `config_fingerprint` is a content hash of the resolved run config, so two runs
  with the same effective configuration share a `config_id`.

### Model registry (`configs/experiment_config.yaml`)

| Model | Owner | Input | Notes |
|---|---|---|---|
| `tfidf_logistic` | person1 | text | required baseline; vocab fit on train |
| `deterministic_scoring` | person1 | features.v1 | selection/direction/threshold on validation; frozen before test |
| `tabular_control` | person1 | features.v1 | gradient boosting; control, **not** a primary result |
| `gnn` | person2 | graph export | lexical first (GCN), then syntactic, then fusion |

The GNN evaluation order is explicitly `lexical_only → syntactic_only →
multi_view_fusion`, and a fusion claim is only allowed against the best
single-view model on the same `split_id`, `dataset_version`, and seeds.

---

## 15. Evaluation protocol (`configs/evaluation_config.yaml`, `evaluation/`)

Protocol version `eval.v1`. Both owners evaluate through this protocol; there
are no ad-hoc splits.

### Non-negotiables

* `require_approved_manifest: true` — predictions target an approved manifest
  only.
* `join_labels_inside_evaluator_only: true` — the label is joined in
  `evaluation` and nowhere else. Prediction files must not contain labels or any
  target metadata; `evaluate_predictions` fails if they do.
* `threshold_selection.forbid_test_tuning: true` — thresholds are chosen on the
  validation split by balanced accuracy (tie-breaker: higher recall) and frozen
  to `experiments/configs/frozen_thresholds.json` before the test split is
  touched.

### Test-set lock (`evaluation/test_lock.py`)

The test split is not a development split.

* `request_test_access(experiment_id, purpose)` logs the access to
  `experiments/locks/test_access_log.jsonl`.
* `single_use_per_experiment: true` — a second access for the same experiment
  raises `TestLockError` unless `override=True`, and an override is itself
  recorded.
* CLI access requires `--allow-test`.

### Metrics (`evaluation/metrics.py`)

Primary metric: **macro F1**. Reported set: accuracy, balanced accuracy,
precision, recall, F1, ROC-AUC, confusion matrix, Brier score, and expected
calibration error. `FORBIDDEN_PREDICTION_COLUMNS` blocks label/source/task/
family/condition/split columns from ever appearing in a prediction file.

### Uncertainty

Stratified bootstrap, 1,000 resamples, 95% confidence, seed 42, reported for
macro F1, balanced accuracy, and ROC-AUC.

### Stratification

Every reported result is broken down by `source_dataset`, `source_task`,
`text_type`, and `label`. Cells below `minimum_cell_size: 30` are reported as
`insufficient_n` rather than as a number. A pooled number without its strata is
not reportable.

### Controls

* **Length** — report a length-only baseline across the configured bins
  (`0–200`, `200–500`, `500–1000`, `1000–3000`, `3000+`). This matters here:
  the audit shows a label-length confound (human median 222 chars vs AI median
  816 chars), so a model must beat a length-only baseline to earn credit.
* **Class prior** — report the majority-class baseline.
* **Condition confounding** — perfect confounding is detected and reported as
  confounded, never as an effect.

### Comparison (`evaluation/comparison.py`)

`compare_predictions` refuses to compare different `split_id`s or different
sample sets. Paired bootstrap gives the metric delta and a two-sided p-value;
`attach_holm` applies Holm-Bonferroni across a family of comparisons
(`alpha: 0.05`). Effect size is reported alongside significance.

### Error analysis (`evaluation/error_analysis.py`)

`error_frame` joins labels for inspection and tags each row `correct`,
`false_positive`, or `false_negative`. `hard_examples` returns the most
confident errors; `error_distribution` reports error rates per source, task, and
condition. Text is shown only for human review and is never fed back into a
model during evaluation.

### Baselines and availability

`tfidf_logistic` is required. `roberta`, `gptzero`, and `binoculars` are
conditional on access/terms/GPU. An unavailable baseline is reported as
unavailable **with its reason** — never silently omitted and never compared
across a different split.

### Current state

No model has been trained and no test-set evaluation has been run. The split
manifest is not approved (see the [dataset audit findings](08_dataset_audit_findings.md)),
so the training and evaluation stages are correctly blocked.
