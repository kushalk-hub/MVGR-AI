# 13. Deterministic branch — final status report (10,000-row run)

This is the closing status report for the **deterministic (statistical)
AI-text detection branch** of MVGR-AI. It records what was built, on which
data, what the measured results are, what was tested, what is limited, and how
to reproduce it.

It is deliberately separate from the graph branch (the teammate's work). The
deterministic branch reads only the four statistical feature families, trains
its own classical models, and needs nothing from the graph package; the two are
integrated later into the hybrid system.

| | |
|---|---|
| Status | **DONE** (deterministic branch only) |
| Dataset | `10000.csv` (10,000 records) — the primary dataset |
| Labels | `0 = Human-authored`, `1 = AI-authored` |
| Features | stylometric + writeprints + GLTR (+ watermark as evidence only) |
| Models | Logistic Regression, Random Forest, XGBoost, SVM (linear + RBF) |
| Best model | **XGBoost** |
| Compute | CPU only, Windows, no GPU |
| Code | `detection/`, `scripts/train_deterministic.py` |
| Config | `configs/deterministic_config.yaml`, `configs/fingerprint_config_10k.yaml` |
| Design doc | [`12_deterministic_branch.md`](12_deterministic_branch.md) |

> **Note on scope.** An earlier 500-row *pilot* was used only to prove the
> pipeline end-to-end (feature extraction, split integrity, fusion, a smoke
> training pass). It is **not** the experiment reported here and it was **never**
> used to train the final models. Every number below comes from the full
> 10,000-row `10000.csv`.

---

## 1. Dataset audit

| Property | Value |
|---|---|
| CSV path | `C:\Users\kusha\OneDrive\Documents\MVGR-AI\10000.csv` |
| SHA-256 | `736a7ea9da1835199a640f68788596185e00931f9496b9d3c7b39eac94257ad9` |
| Rows in the CSV | **10,000** |
| Valid records (non-empty text, parsable label) | **10,000** |
| Invalid records | **0** (none; no row was dropped, substituted or downsampled) |
| Human-authored (`label = 0`) | **2,148** |
| AI-authored (`label = 1`) | **7,852** |
| Imbalance (AI/Human) | 3.6555 |
| Duplicate `sample_id` | 0 |
| Duplicate texts (exact / normalized) | 0 / 0 |

**Composition.** Source: PADBen 7,157 / BiScope 2,843. Text type: original
1,694, paraphrased 3,333, deeply paraphrased 3,333, unspecified (BiScope) 1,640.
Domains: Unknown 7,157, Yelp 1,260, Essay 669, Creative 619, Arxiv 195, Code 100.
`ai_model` is `Unknown` for all 10,000 rows, so no generator identity is used
anywhere. Text length: min 10, median 123, mean 586, max 16,349 characters.

**Paraphrase families.** `group_id` defines 9,887 families; 113 of them have
more than one member (226 rows). Paraphrasing never changes authorship, so a
family is kept whole inside a single split.

The CSV's own `split` column (train 8,569 / validation 865 / test 566) does
**not** follow the repository split policy and disagrees with the candidate
manifest, so it is **not** used. This is documented in
`reports/deterministic/dataset_analysis.md`.

---

## 2. Split (leakage-safe, family-disjoint)

The split is resolved deterministically and never modified:

* the 7,157 PADBen rows reuse the existing candidate manifest
  `data/manifests/candidate/man_20ddf272c292efbc8c8c`;
* the 2,843 BiScope rows are assigned by the repository's own family-disjoint
  policy (`dataset_audit/split_manifest.py::assign_by_family`, seed 42, ratios
  0.70 / 0.15 / 0.15, family = `group_id`).

| Split | Rows |
|---|---:|
| train | **6,978** |
| validation | **1,518** |
| test | **1,504** |

Integrity checks: **0** paraphrase families and **0** duplicate texts span two
splits. The assignment is written to
`outputs/deterministic/split_assignment.csv` and was re-loaded and re-verified
byte-identical (0 mismatches). The manifest itself is untouched.

Test-split composition: 319 human / 1,185 AI.

---

## 3. Feature extraction and coverage

All 10,000 rows were extracted (offline, resumable, checkpointed):

| Family | Raw columns | Source | Notes |
|---|---:|---|---|
| Stylometric | 140 | `fingerprinting/` | vocabulary, sentence, punctuation, POS, syntactic, readability |
| Writeprints | 301 | `fingerprinting/` | bounded char/word/POS n-grams + function words; vocabulary fit on the **training split only** (`fit_scope: train_split_only`, `fit_rows: 6978`) |
| GLTR | 19 | `gltr/` | real GPT-2 (offline, CPU, length-sorted batching; order-invariant) |
| Watermark | 12 | `watermark/` | **evidence only** — never a model feature |

**Coverage: 10,000 / 10,000 records, 0 uncovered, 0 extra rows.** The fused
matrix has **375** columns and a missing-cell fraction of 1.2e-05 (45 cells):
five *training* rows (all Python-code samples whose single spaCy "sentence"
exceeds the 75-word top histogram bin) have no sentence-length histogram; those
cells are median-imputed with statistics fitted on the training split. All other
9,995 records are complete.

Fusion produced **460 candidate** features; the selection dropped 3 (missing), 2
(constant) and 80 (correlated), keeping **375**. The shipped selection was made
with the whole table it was given, so the experiments use a **train-only**
re-selection (`detection/feature_alignment.select_features_train_only`); the
original artifacts are preserved. For this matrix the two selections agree
exactly (375 = 375, identical `True`).

Selected per family: stylometric 115, writeprints 244, GLTR 16 (watermark is not
part of the model matrix).

---

## 4. Models and hyperparameters

Each model was tuned with `RandomizedSearchCV` under **group-aware stratified
3-fold CV** (paraphrase families kept together), scored on ROC-AUC, with
train-only imputation and scaling. Best parameters and validation scores:

| Model | Best hyperparameters | CV ROC-AUC | Validation accuracy | Val. balanced acc. | Val. F1 | Val. ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | `C=0.1`, `l1_ratio=1.0` (L1), `class_weight=None`, `solver=liblinear` | 0.9826 | 0.9697 | 0.9402 | 0.9811 | 0.9853 |
| Random Forest | `n_estimators=200`, `max_depth=12`, `max_features=0.5`, `min_samples_leaf=2`, `min_samples_split=20` | 0.9758 | 0.9519 | 0.9194 | 0.9699 | 0.9839 |
| SVM (RBF, calibrated) | `kernel=rbf`, `C=5.0`, `gamma=0.001` | 0.9819 | 0.9539 | 0.9243 | 0.9711 | 0.9850 |
| **XGBoost** | `n_estimators=600`, `max_depth=10`, `learning_rate=0.1`, `subsample=1.0`, `colsample_bytree=0.8`, `reg_lambda=2.0` | **0.9884** | **0.9717** | **0.9426** | **0.9824** | **0.9923** |

The SVM uses Platt calibration on the validation split (`MarginCalibratedModel`)
rather than `SVC(probability=True)`, so its probabilities are calibrated. The
best model is chosen on **validation ROC-AUC**; the test split is read **once**,
afterwards.

---

## 5. Test-set evaluation (untouched, 1,504 rows)

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** (best) | **0.9661** | **0.9372** | 0.9701 | 0.9873 | 0.9787 | **0.9910** | 0.9975 | 0.1129 | 0.0127 | 0.8871 |
| Logistic Regression | 0.9668 | 0.9400 | 0.9717 | 0.9865 | 0.9791 | 0.9887 | 0.9967 | 0.1066 | 0.0135 | 0.8934 |
| SVM (RBF, calibrated) | 0.9594 | 0.9365 | 0.9716 | 0.9771 | 0.9743 | 0.9896 | 0.9972 | 0.1034 | 0.0236 | 0.8966 |
| Random Forest | 0.9481 | 0.9110 | 0.9637 | 0.9713 | 0.9674 | 0.9826 | 0.9952 | 0.1536 | 0.0245 | 0.8464 |

XGBoost confusion matrix (test): TN 283, FP 36, FN 15, TP 1,170. A
majority-class baseline would score ≈0.785 accuracy, so balanced accuracy and
the per-class rates are the numbers that matter.

**Cost.** Training/search/prediction time and model size (per model):

| Model | Search seconds | Test prediction seconds | Model file bytes | Model memory bytes |
|---|---:|---:|---:|---:|
| Logistic Regression | 228.5 | 0.014 | 29,714 | 3,715 |
| Random Forest | 116.8 | 0.110 | 3,933,104 | 3,911,182 |
| SVM | 806.2 | 2.553 | 6,971,268 | 6,944,781 |
| XGBoost | 160.5 | 0.022 | 1,334,019 | 1,307,932 |

The whole `--stage all` run takes ≈25 minutes on this CPU and is deterministic:
two independent full runs produced **bit-identical** metrics.

---

## 6. Feature ablation (E1–E9, test set)

Every set is selected with the training rows only; the same split, reference
model (XGBoost) and metrics are used throughout.

| Set | Features | Accuracy | Balanced acc. | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 stylometric only | 115 | 0.8737 | 0.7629 | 0.9226 | 0.9128 | 0.4295 | 0.0447 |
| E2 writeprints only | 244 | 0.9176 | 0.8538 | 0.9485 | 0.9600 | 0.2571 | 0.0354 |
| E3 GLTR only | 16 | 0.7979 | 0.6048 | 0.8799 | 0.7484 | 0.7304 | 0.0599 |
| E4 watermark only | 7 | 0.7713 | 0.5364 | 0.8668 | 0.6736 | 0.8715 | 0.0557 |
| E5 combined selected | 375 | 0.9661 | 0.9372 | 0.9787 | 0.9910 | 0.1129 | 0.0127 |
| E6 combined without GLTR | 359 | 0.9674 | 0.9404 | 0.9795 | 0.9913 | 0.1066 | 0.0127 |
| E7 combined without writeprints | 131 | 0.8843 | 0.7742 | 0.9293 | 0.9306 | 0.4169 | 0.0346 |
| E8 combined without stylometric | 260 | 0.9162 | 0.8529 | 0.9477 | 0.9663 | 0.2571 | 0.0371 |
| E9 combined without watermark | 375 | 0.9661 | 0.9372 | 0.9787 | 0.9910 | 0.1129 | 0.0127 |

Findings:

* **Writeprints is the strongest single family** and the one whose removal hurts
  most (E7 ROC-AUC 0.9306 vs E5 0.9910).
* **GLTR adds nothing measurable** on top of the other families (E6 0.9913 vs
  E5 0.9910 — within noise), although GLTR-only (E3) is far above chance.
* The **watermark-only** model is near chance (E4 ROC-AUC 0.6736) because the
  generation-time key is unknown; its green list is arbitrary with respect to
  authorship.
* **E9 = E5 by construction** — the watermark is evidence-only and was never a
  model feature, so removing it cannot change anything.

---

## 7. Paraphrase robustness (selected model, test set)

No paraphrased text was synthesised; every row is from `10000.csv`. Categories
are the dataset's own labels.

| Category | n | Accuracy | FPR | FNR | Mean AI probability |
|---|---:|---:|---:|---:|---:|
| Deeply paraphrased AI | 511 | 0.9980 | — | 0.0020 | 0.9970 |
| Paraphrased AI | 331 | 0.9879 | — | 0.0121 | 0.9842 |
| Original AI | 195 | 0.9846 | — | 0.0154 | 0.9835 |
| Unspecified (BiScope) AI | 148 | 0.9527 | — | 0.0473 | 0.9318 |
| Paraphrased Human | 153 | 0.9346 | 0.0654 | — | 0.0921 |
| Unspecified (BiScope) Human | 97 | 0.9175 | 0.0825 | — | 0.1051 |
| Original Human | 69 | 0.7391 | 0.2609 | — | 0.2567 |

* Detection of AI text does **not** degrade under paraphrasing: deeply
  paraphrased AI is in fact the easiest category (accuracy 0.9980, FNR 0.0020;
  +0.0134 accuracy vs original AI).
* The **weakest group is short original human text** (FPR 0.2609) — the model is
  most likely to mistake a short, clean human document for AI.
* Worst human FPR 0.2609; worst AI detection rate 0.9527.
* `robustness_claim_supported = false`: robustness is **not** claimed
  unconditionally, because the human false-positive rate is high for one
  category. (Under the more forgiving SVM the worst human FPR is 0.1449, which
  would flip the flag to true — the conservative value is reported.)

---

## 8. Interpretability (selected model)

Group sizes on the test split: TP 1,170, TN 283, FP 36, FN 15.

* **Highest XGBoost gain:** `writeprints__wp_pos_bigram_000` (21.27),
  `stylometric__vocab_total_words` (13.12),
  `writeprints__wp_char_unigram_000` (12.64), `stylometric__punct_apostrophe_per_kchar`
  (8.68), `gltr__gltr_mean_log_prob` (8.42).
* **SHAP** (tree explainer, 500 test rows) agrees:
  `writeprints__wp_char_unigram_000` dominates, then
  `stylometric__vocab_avg_word_length`.
* **Top-15 family mix:** writeprints 6, stylometric 8, GLTR 1 — no single family
  explains the decision.
* **False positives** are longer, more heavily punctuated human documents
  (`vocab_total_words` mean 126 vs 43 for true negatives; more commas, wider
  sentence-length range).
* **False negatives** are short, highly readable AI text (mean `vocab_total_words`
  28 vs 90 for true positives; Flesch reading ease 81 vs 39).
* Importance describes **what the model used**, not causality.

---

## 9. Saved artifacts

**Models** — `models/deterministic/`

* `logistic_regression/{model.pkl,metadata.json}`
* `random_forest/{model.pkl,metadata.json}`
* `svm/{model.pkl,metadata.json}`
* `xgboost/{model.pkl,metadata.json}`
* `best_model.pkl` (XGBoost, the reusable bundle)
* `preprocessing_pipeline.pkl` (train-fitted imputation + scaling)
* `model_metadata.json` (dataset file, actual training size, feature
  names/order/count, label convention, preprocessing config, best
  hyperparameters, validation metrics, training timestamp)

**Reports** — `reports/deterministic/` (each names `10000.csv` and its valid
sample count):

* `dataset_analysis.md`, `feature_analysis.md`, `model_comparison.md`,
  `feature_ablation.md`, `paraphrase_robustness.md`, `interpretability.md`,
  `final_evaluation.md`
* figures: confusion matrices, ROC curves, PR curves, feature importance,
  model comparison

**Machine-readable outputs** — `outputs/deterministic/`: `dataset_audit.json`,
`feature_coverage.json`, `feature_selection.json`, `model_comparison.{csv,json}`,
`feature_ablation.{csv,json}`, `paraphrase_robustness.{csv,json}`,
`predictions/*_test.csv`, `tuning/*`, `split_assignment.csv`, `train_state.json`.

---

## 10. Prediction interface

`detection/predictor.py` exposes a reusable, graph-independent API:

```python
from detection.predictor import DeterministicDetector

detector = DeterministicDetector.load("models/deterministic/best_model.pkl")
result = detector.predict_from_features(features)
# {"prediction": "AI", "label": 1, "human_probability": 0.18,
#  "ai_probability": 0.82, "model": "XGBoost", ...}
```

* `predict_from_features` (single row) and `predict_batch` (many rows) accept the
  same feature schema as training and return `prediction`, `label`,
  `human_probability`, `ai_probability` and `model`.
* Only probability-producing / calibrated models are accepted; an uncalibrated
  decision score is never presented as a probability, and the two probabilities
  sum to 1.
* Verified end-to-end against the real saved model: single and batch(200)
  predictions match `outputs/deterministic/predictions/xgboost_test.csv`.

---

## 11. Tests

* `tests/test_detection.py`: **30 unit tests** (no network, no model download):
  dataset audit, family-disjoint split integrity, train-only selection,
  preprocessing, metrics, E1–E9 ablation, paraphrase robustness,
  interpretability (including XGBoost importance alignment), predictor
  round-trip, margin calibration, graph-independence, text features, vocabulary
  scope.
* Full suite: **168 passed, 1 skipped**. The single skip is a pre-existing
  Neo4j-credentials test unrelated to this branch.

---

## 12. Limitations

* GPT-2 (2019) is the GLTR reference model; it is older and smaller than current
  generators, so absolute predictability values are reference-relative.
* The watermark family carries no authorship signal here: the generation-time key
  is unknown, so its green list is arbitrary. E4 is therefore near chance by
  design.
* Five training rows (all Python-code samples) have no sentence-length histogram
  because their single spaCy "sentence" exceeds the 75-word top bin; those cells
  are median-imputed with training-split statistics.
* The writeprints n-gram vocabulary is fit on the training split only
  (`fit_scope: train_split_only`), so neither validation nor test text influenced
  it.
* The dataset is imbalanced (≈78.5% AI); accuracy alone is misleading, so
  balanced accuracy and per-class rates are the reported headline metrics.
* Paraphrase robustness is not claimed unconditionally: the human
  false-positive rate is high for short original human text.
* Feature importance is descriptive, not causal.
* No model was trained on graph features; that is the other branch of MVGR-AI and
  is integrated separately.

---

## 13. Reproduce

```powershell
# 1) resolve the leakage-safe split (dataset + manifest only; no features needed)
.\.venv\Scripts\python.exe scripts\train_deterministic.py --stage split

# 2) extract features for all 10,000 rows (offline, resumable, checkpointed)
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py text --config fingerprint_config_10k
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py gltr --config fingerprint_config_10k
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py fuse --config fingerprint_config_10k

# 3) train, evaluate, ablate, stress-test and explain (≈25 min, CPU, deterministic)
.\.venv\Scripts\python.exe scripts\train_deterministic.py --stage all

# 4) tests
.\.venv\Scripts\python.exe -m pytest
```

---

## 14. Bottom line

On the full 10,000-row `10000.csv` (2,148 human / 7,852 AI), with a
family-disjoint split (train 6,978 / validation 1,518 / test 1,504), 375
train-only-selected features and four tuned classical models, the deterministic
branch selects **XGBoost** and reaches **test accuracy 0.9661, balanced accuracy
0.9372, F1 0.9787, ROC-AUC 0.9910, PR-AUC 0.9975** on 1,504 untouched rows.
Detection of AI text holds up under paraphrasing (deep-paraphrased AI accuracy
0.9980); the main weakness is false positives on short original human text. The
branch is complete, reproducible, tested, and independent of the graph branch.
