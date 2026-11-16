# 12. Deterministic AI-text detection branch (`10000.csv`)

This document describes the **deterministic branch**: the stylometric,
writeprints, GLTR and watermark features turned into a trained, independently
evaluated AI-text detector.

It is a sibling of the graph branch, not a replacement for it. The two are
integrated later into the hybrid MVGR-AI system.

| | |
|---|---|
| Dataset | `10000.csv` (10,000 records) |
| Labels | `0 = Human-authored`, `1 = AI-authored` |
| Features | stylometric + writeprints + GLTR (+ watermark as evidence) |
| Models | Logistic Regression, Random Forest, XGBoost, SVM |
| Compute | CPU only, Windows |
| Code | `detection/`, `scripts/train_deterministic.py` |
| Config | `configs/deterministic_config.yaml` |
| Models | `models/deterministic/` |
| Reports | `reports/deterministic/` |

---

## 1. Scope and separation from the graph branch

The deterministic branch reads **only** the four statistical branches and never
touches graph construction, graph features or the GNN package. Nothing in
`detection/` imports from `graph_engine/` or from the graph-analysis package
`deterministic/` (which measures graph topology and belongs to the other
branch).

The prediction interface (`detection/predictor.py`) needs nothing but a
numeric feature vector and the model files, so it can be dropped into the
hybrid system without pulling the graph branch in.

---

## 2. Dataset

### 2.1 The file

`10000.csv` sits at the repository root (7.7 MB). It is the primary dataset of
this branch and is never modified.

| Column | Role |
|---|---|
| `sample_id` | join key |
| `text` | the document |
| `label` | `0` human / `1` AI |
| `source_dataset`, `source_task`, `domain` | provenance metadata |
| `text_type`, `graph_category`, `paraphrase_status` | paraphrase status |
| `ai_model` | provenance metadata (all `Unknown` in this file) |
| `padben_split`, `split` | the split shipped inside the CSV |
| `group_id`, `source_row_id` | lineage identifiers |
| `split` (column) | the CSV's own assignment |

### 2.2 Composition

* 10,000 rows, **all valid** (non-empty text, parsable label).
* **2,148 human** and **7,852 AI** — an AI/human ratio of 3.66. A detector that
  always answers "AI" already scores ≈0.785 accuracy, so balanced accuracy,
  the false-positive rate and the false-negative rate are the numbers that
  decide whether a model is useful.
* `text_type`: 1,694 original, 3,333 paraphrased, 3,333 deeply paraphrased,
  1,640 unspecified (BiScope sentence pairs).
* `source_dataset`: 7,157 PADBen, 2,843 BiScope.
* No duplicate `sample_id`, no duplicate text (exact or whitespace-normalised).
* `group_id` is the paraphrase family key (`PADBen:<task>_<split>:<row>`):
  9,887 families, 113 of them holding more than one member, and those
  families contain both a human and an AI version of the same source
  document. **A family must never be split across two subsets.**

### 2.3 The split

The `split` column shipped in the CSV does **not** follow the repository's
split policy, and it disagrees with the candidate manifest. The resolved split
is therefore built as:

* every row covered by the candidate manifest
  `man_20ddf272c292efbc8c8c` (PADBen, 7,157 rows) keeps the manifest's
  assignment — the manifest itself is never modified;
* the remaining rows (BiScope, which the source-scoped manifest does not cover)
  are assigned with the repository's own family-disjoint policy
  (`dataset_audit.split_manifest.assign_by_family`), the same seed (42) and
  the same ratios (0.70 / 0.15 / 0.15).

The result is checked before it is used:

| Check | Result |
|---|---|
| Families spanning more than one split | **0** |
| Duplicate texts spanning more than one split | **0** |
| Every row assigned exactly once | yes |

| Split | Rows | Human | AI |
|---|---:|---:|---:|
| train | 6,978 | 1,519 | 5,459 |
| validation | 1,518 | 310 | 1,208 |
| test | 1,504 | 319 | 1,185 |

---

## 3. Leakage control

The leakage rules this branch enforces, and where each one lives:

| Risk | Control | Where |
|---|---|---|
| A paraphrase family in two splits | the split is family-disjoint | `detection/dataset_loader.py` |
| A duplicate text in two splits | checked in the split report | `detection/dataset_loader.py` |
| **Feature selection informed by the test rows** | selection is fitted on **train only** | `detection/feature_alignment.py` |
| **The writeprints vocabulary informed by the test rows** | the vocabulary (which decides *which* n-grams become features) is fitted on **train only** | `scripts/fingerprint_pipeline.py` (`writeprints.vocabulary_scope`) |
| Imputation / scaling fitted on the whole table | fitted on **train only** | `detection/preprocessing.py` |
| Hyperparameters chosen on the test split | tuning uses train-only, group-aware CV; the test split is read once, after the model is chosen on **validation** | `detection/train_models.py`, `scripts/train_deterministic.py` |
| Metadata used as a feature | metadata lives in a separate table and is never in the feature matrix | `fusion/`, `detection/feature_alignment.py` |
| Watermark evidence used as a class signal | the watermark branch is evidence-only (`watermark_as_feature: false`) | `configs/fingerprint_config_10k.yaml` |

### 3.1 The writeprints vocabulary is a learned step too

The n-gram vocabulary is not a fixed list: it is built by document frequency
over the corpus it is fitted on, so the *set of features that exist* depends
on the rows that were used. Fitting it on all 10,000 rows would let the test
rows decide which n-grams become features.

The vocabulary is therefore fitted on the **training rows only**. This needs
the split assignment to exist before the extraction, which is why the split is
resolved first:

```powershell
# 1) resolve the split (dataset + manifest only, no features needed)
.\.venv\Scripts\python.exe scripts\train_deterministic.py --stage split

# 2) extract the features; the writeprints vocabulary is fitted on train only
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py writeprints --config fingerprint_config_10k
```

The scope that was actually used is recorded in the vocabulary artifact
(`fit_scope`) and in the extraction report, so it can be checked rather than
assumed.

### 3.2 The shipped 424-column selection is contaminated

The fused feature matrix produced by the fingerprinting pipeline selects its
columns with `fusion.multiview_feature_fusion.select_features` over the whole
table it is given. For the 500-row pilot that table was the entire pilot, so
the shipped 424-column set was chosen with the evaluation rows included.

`detection/feature_alignment.select_features_train_only` re-runs **exactly the
same rules** — missing-value threshold, constant columns, correlation
filtering — but only on the training rows, and then applies the resulting
column list to validation and test. The existing artifacts are left untouched
so the two can be compared.

For the 10,000-row matrix the two selections happen to agree: both keep the
same **375** features, so no column was added or removed by the train-only
rule. The pipeline still runs the train-only selection as the default, so the
guarantee does not depend on this coincidence (`feature_analysis.md` reports
the comparison every run).

### 3.3 Group-aware cross-validation

`StratifiedGroupKFold` is used for the hyperparameter search, with `group_id`
as the group, so a paraphrase family never appears in both the training and
the validation part of a fold.

---

## 4. Features

The four branches are produced by the existing implementation and are not
re-implemented here:

| Branch | Raw features | Kept after selection | Source |
|---|---:|---:|---|
| Stylometric | 140 | 115 | `fingerprinting/stylometry.py` |
| Writeprints | 301 | 244 | `fingerprinting/writeprints.py` |
| GLTR | 19 | 16 | `gltr/` (real GPT-2 base, offline, CPU) |
| Watermark | 12 (evidence only) | 7 | `watermark/` |

The "raw" column is what each branch emits; the "kept" column is how many
survive the fusion selection (and then the train-only re-selection). Watermark
is never a model feature — it is written to the metadata table — so the 7
columns above are only used by the E4 ablation, not by any trained model.

The 10,000-row extraction writes to `data/processed/fingerprinting_10k/`; the
existing 500-row pilot artifacts in `data/processed/fingerprinting/` are kept
for comparison.

### 4.1 GLTR at 10,000 samples

| Setting | Value |
|---|---|
| Model | `gpt2`, 124,439,808 parameters, vocab 50,257 |
| Revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Storage | `models/gltr/gpt2/` (offline, `local_files_only: true`) |
| Device / dtype | `cpu` / `float32` |
| Windowing | `max_length: 256`, `stride: 128` — every target token scored once |
| Batching | batch 4, dynamically padded, attention-masked, `torch.inference_mode()` |
| Batching order | shortest document first (see below) |

**Batching order.** The 10,000 documents have a very skewed length
distribution (median 26 tokens, 99th percentile 1,130, max 3,333). Batches are
padded to their longest member, so mixing a 26-token document with a 3,333-token
one wastes most of the compute. Processing the documents from shortest to
longest cuts the padded work from ≈2.63 M to ≈1.80 M token slots (1.46×).

This does **not** change any feature value: padding is masked out of the
forward pass and the per-document aggregation is order-invariant. The residual
difference is float32 accumulation noise — measured at 5·10⁻⁶ on the token
probabilities with **zero** rank changes, which is the same magnitude as
re-running the same batch in a different shape.

### 4.2 Watermark without the generation key

The watermark key is read from `MVGR_WATERMARK_KEY` and never stored. The key
used for this run is a research placeholder, so the derived green list is
**arbitrary with respect to authorship**. The watermark family therefore
carries no real authorship signal, and the ablation study (E4) measures that
rather than assuming it.

---

## 5. Models and tuning

| Model | Role | Tuned hyperparameters |
|---|---|---|
| Logistic Regression | linear baseline, interpretable coefficients | `C`, penalty, `class_weight` |
| Random Forest | bagged trees, impurity importances | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| XGBoost | boosted trees, interactions | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda` |
| SVM | margin classifier, linear and RBF kernels | `C`, `kernel`, `gamma` |

* The penalty of the Logistic Regression is tuned as `l1_ratio` (0.0 = L2,
  1.0 = L1) because scikit-learn 1.9 deprecated the `penalty` argument;
  `liblinear` is the solver that supports both.
* The SVM's margin scores are calibrated with Platt scaling **on the
  validation split** instead of using `SVC(probability=True)`, which runs an
  internal five-fold cross-validation on every fit and multiplies the cost of
  a search by five. The probabilities are genuinely calibrated; no decision
  score is ever presented as a probability.
* Tuning: `RandomizedSearchCV` with `StratifiedGroupKFold`, scored by ROC-AUC.
* The best model is chosen on **validation ROC-AUC**; the test split is read
  once, afterwards.

---

## 6. Ablation study

Nine feature sets, one methodology (same split, same reference model, same
metrics, same test rows):

| Key | Feature set |
|---|---|
| E1 | Stylometric only |
| E2 | Writeprints only |
| E3 | GLTR only |
| E4 | Watermark only |
| E5 | Combined selected features |
| E6 | Combined without GLTR |
| E7 | Combined without Writeprints |
| E8 | Combined without Stylometric |
| E9 | Combined without Watermark |

Every set is selected with the training rows only. E9 is identical to E5 by
construction — the watermark branch is evidence and never enters the feature
matrix — which is itself the finding.

---

## 7. Robustness and interpretability

* **Robustness** — the winning model is scored per text category (original /
  paraphrased / deeply paraphrased × human / AI) on the test split. Every row
  comes from `10000.csv`; **no paraphrased text is synthesised**. Robustness
  is claimed only where the numbers support it.
* **Interpretability** — signed Logistic Regression coefficients, Random
  Forest and XGBoost importances, SHAP values from the tree explainer, and an
  error profile that compares the feature means of true positives, true
  negatives, false positives and false negatives.

Feature importance describes what the model *used*. It is not evidence of
causality.

---

## 8. Results (the 10,000-row run)

All numbers below come from `reports/deterministic/` and `outputs/deterministic/`
and use the untouched test split (1,504 rows: 319 human / 1,185 AI).

### 8.1 Model comparison (test set)

| Model | Accuracy | Balanced acc. | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** (selected) | 0.9661 | 0.9372 | 0.9787 | 0.9910 | 0.9975 | 0.1129 | 0.0127 |
| Logistic Regression | 0.9668 | 0.9400 | 0.9791 | 0.9887 | 0.9967 | 0.1066 | 0.0135 |
| SVM (RBF, calibrated) | 0.9594 | 0.9365 | 0.9743 | 0.9896 | 0.9972 | 0.1034 | 0.0236 |
| Random Forest | 0.9481 | 0.9110 | 0.9674 | 0.9826 | 0.9952 | 0.1536 | 0.0245 |

A majority-class baseline would score ≈0.785 accuracy, so balanced accuracy and
the per-class rates are the numbers that matter. XGBoost is selected by
**validation ROC-AUC (0.9923)**; the test split is read once, afterwards. The
best hyperparameters are `n_estimators=600, max_depth=10, learning_rate=0.1,
subsample=1.0, colsample_bytree=0.8, reg_lambda=2.0`.

### 8.2 Ablation (E1–E9, test ROC-AUC)

| Set | Features | Accuracy | ROC-AUC |
|---|---:|---:|---:|
| E1 stylometric only | 115 | 0.8737 | 0.9128 |
| E2 writeprints only | 244 | 0.9176 | 0.9600 |
| E3 GLTR only | 16 | 0.7979 | 0.7484 |
| E4 watermark only | 7 | 0.7713 | 0.6736 |
| E5 combined selected | 375 | 0.9661 | 0.9910 |
| E6 combined without GLTR | 359 | 0.9674 | 0.9913 |
| E7 combined without writeprints | 131 | 0.8843 | 0.9306 |
| E8 combined without stylometric | 260 | 0.9162 | 0.9663 |
| E9 combined without watermark | 375 | 0.9661 | 0.9910 |

Writeprints is the strongest single family and the one whose removal hurts most;
GLTR adds nothing measurable on top of the other two (E6 is within noise of E5);
the watermark-only model is near chance (E4), exactly as expected without the
generation key. E9 equals E5 by construction.

### 8.3 Paraphrase robustness (selected model)

| Category | n | Accuracy | FPR | FNR | Mean AI prob. |
|---|---:|---:|---:|---:|---:|
| Deeply paraphrased AI | 511 | 0.9980 | — | 0.0020 | 0.9970 |
| Paraphrased AI | 331 | 0.9879 | — | 0.0121 | 0.9842 |
| Original AI | 195 | 0.9846 | — | 0.0154 | 0.9835 |
| Unspecified (BiScope) AI | 148 | 0.9527 | — | 0.0473 | 0.9318 |
| Paraphrased Human | 153 | 0.9346 | 0.0654 | — | 0.0921 |
| Unspecified (BiScope) Human | 97 | 0.9175 | 0.0825 | — | 0.1051 |
| Original Human | 69 | 0.7391 | 0.2609 | — | 0.2567 |

Detection of AI text does **not** degrade under paraphrasing (deep
paraphrasing is in fact the easiest AI category here), but the false-positive
rate on human text is highest for short *original* human documents, so
robustness is not claimed unconditionally. No paraphrased text was synthesised;
every row is from `10000.csv`.

### 8.4 Interpretability (selected model)

The highest-gain features are the writeprints POS-bigram and character-unigram
counts, followed by stylometric vocabulary size and punctuation rates and the
GLTR mean log-probability. SHAP (tree explainer, 500 test rows) agrees:
`writeprints__wp_char_unigram_000` dominates, then
`stylometric__vocab_avg_word_length`. The top 15 features span writeprints (6),
stylometric (8) and GLTR (1) — no single family explains the decision. False
positives are longer, more punctuated human documents; false negatives are
short, highly readable AI text. Importance is descriptive, not causal.

---

## 9. Reproduce

```powershell
# 0) install the CPU-only GLTR extra and download the reference model (once)
.\.venv\Scripts\python.exe -m pip install -r requirements\gltr.txt
.\.venv\Scripts\python.exe scripts\download_gltr_model.py

# 1) resolve the leakage-safe split (dataset + manifest only)
.\.venv\Scripts\python.exe scripts\train_deterministic.py --stage split

# 2) extract the features for all 10,000 rows (offline, resumable)
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py text    --config fingerprint_config_10k
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py gltr    --config fingerprint_config_10k
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py fuse    --config fingerprint_config_10k
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py reports --config fingerprint_config_10k

# 3) train, evaluate, ablate, stress-test and explain
.\.venv\Scripts\python.exe scripts\train_deterministic.py --stage all

# 4) validate the extracted features
.\.venv\Scripts\python.exe scripts\validate_fingerprint_features.py --config fingerprint_config_10k
```

Step 1 must come before step 2: the writeprints vocabulary is fitted on the
training rows and needs the split assignment to exist.

Single stages are available too: `--stage split | audit | train | ablation |
robustness | interpretability | report`.

### Using the detector

```python
from detection.predictor import DeterministicDetector

detector = DeterministicDetector.load("models/deterministic/best_model.pkl")
result = detector.predict_from_features(features)
# {"prediction": "AI", "label": 1, "human_probability": 0.18,
#  "ai_probability": 0.82, "model": "XGBoost", ...}
```

`predict_from_text(text)` is available as a convenience; it runs the existing
fingerprinting pipeline (including the offline GPT-2 model) to build the
feature vector.

---

## 10. Layout

```text
detection/
├── dataset_loader.py        load + audit 10000.csv, resolve the leakage-free split
├── feature_alignment.py     align features to sample ids; train-only selection
├── preprocessing.py         train-only fitted imputation and scaling
├── train_models.py          the four classifiers, group-aware tuning
├── evaluate_models.py       metrics, plots, result artifacts
├── feature_ablation.py      the E1-E9 ablation
├── robustness_evaluation.py per-category paraphrase robustness
├── interpretability.py      coefficients, importances, SHAP, error profile
├── predictor.py             the reusable DeterministicDetector
└── text_features.py         raw text -> feature vector (convenience)

models/deterministic/         trained models, best_model.pkl, metadata
reports/deterministic/        the seven reports + figures
outputs/deterministic/        CSV/JSON artifacts and test predictions
```

---

## 11. Limitations

* GPT-2 (2019) is the reference model for the GLTR features. It is smaller and
  older than current generators, so absolute predictability values are
  reference-relative.
* The watermark family cannot carry authorship signal without the key that
  was used at generation time. It is evidence, not a feature.
* The writeprints vocabulary is fitted on the training split, so the n-grams
  that exist are slightly different from a vocabulary fitted on everything —
  that is the price of not letting the test rows shape the feature space.
* The dataset is ~78% AI. Per-category accuracy has to be read together with
  the false-positive rate on human text.
* `domain` and `ai_model` are `Unknown` for most rows, so the model cannot
  learn a domain-specific signature.
* Feature importance is descriptive, not causal.
* Nothing here touches the graph branch; the hybrid MVGR-AI system is a
  separate integration step.
