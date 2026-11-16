# Final evaluation — deterministic AI-text detector

- Generated: 2026-11-16T14:45:06Z
- Dataset: **`10000.csv`** (10000 rows; 10000 valid records; 2148 human, 7852 AI)
- Feature coverage: 10000/10000 records, 375 candidate columns
- Features used: **375** (train-only selection)
- Split: train 6978 / validation 1518 / test 1504 (leakage-free, family-disjoint)

## Test-set metrics

| model | label | split | accuracy | precision | recall | f1 | balanced_accuracy | true_negative | false_positive | false_negative | true_positive | false_positive_rate | false_negative_rate | specificity | precision_positive | n_samples | roc_auc | pr_auc | training_seconds | search_seconds | prediction_seconds_test | model_memory_bytes | model_arrays | model_file_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | XGBoost | test | 0.9661 | 0.9701 | 0.9873 | 0.9787 | 0.9372 | 283.0000 | 36.0000 | 15.0000 | 1170.0000 | 0.1129 | 0.0127 | 0.8871 | 0.9701 | 1504.0000 | 0.9910 | 0.9975 | 160.5160 | 160.5160 | 0.0210 | 1307932.0000 | 0.0000 | 1334046.0000 |
| svm | Support Vector Machine | test | 0.9594 | 0.9723 | 0.9764 | 0.9743 | 0.9365 | 286.0000 | 33.0000 | 28.0000 | 1157.0000 | 0.1034 | 0.0236 | 0.8966 | 0.9723 | 1504.0000 | 0.9896 | 0.9972 | 806.2180 | 806.2180 | 2.5530 | 6944781.0000 | 16.0000 | 6971268.0000 |
| logistic_regression | Logistic Regression | test | 0.9668 | 0.9717 | 0.9865 | 0.9791 | 0.9400 | 285.0000 | 34.0000 | 16.0000 | 1169.0000 | 0.1066 | 0.0135 | 0.8934 | 0.9717 | 1504.0000 | 0.9887 | 0.9967 | 228.5110 | 228.5110 | 0.0140 | 3715.0000 | 4.0000 | 29714.0000 |
| random_forest | Random Forest | test | 0.9481 | 0.9593 | 0.9755 | 0.9674 | 0.9110 | 270.0000 | 49.0000 | 29.0000 | 1156.0000 | 0.1536 | 0.0245 | 0.8464 | 0.9593 | 1504.0000 | 0.9826 | 0.9952 | 116.8420 | 116.8420 | 0.1100 | 3911182.0000 | 1.0000 | 3933104.0000 |


## Selected model

- **XGBoost** — chosen by validation ROC-AUC
- Best hyperparameters: `{
  "subsample": "1.0",
  "reg_lambda": "2.0",
  "n_estimators": "600",
  "max_depth": "10",
  "learning_rate": "0.1",
  "colsample_bytree": "0.8"
}`
- Test accuracy 0.9661, balanced accuracy 0.9372, F1 0.9787, ROC-AUC 0.9910, PR-AUC 0.9975
- False-positive rate 0.1129, false-negative rate 0.0127

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\train_deterministic.py --stage all
```

## Limitations

- GPT-2 (2019) is the reference model for the GLTR features; it is older and smaller than current generators, so absolute predictability values are reference-relative.
- The watermark family cannot carry authorship signal here: the key used at generation time is unknown, so its green list is arbitrary with respect to authorship.
- The writeprints n-gram vocabulary is fit on the **training split only** (6,978 rows; `fit_scope: train_split_only`), so neither the validation nor the test text influenced it.
- Five training rows (all Python-code samples) have no sentence-length histogram because their single spaCy "sentence" exceeds the 75-word top bin; those cells are median-imputed with statistics fit on the training split.
- No model was trained on the graph features; that is the other branch of MVGR-AI and is integrated separately.
- Feature importance is descriptive, not causal.
