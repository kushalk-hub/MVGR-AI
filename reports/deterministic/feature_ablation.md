# Feature ablation study

- Generated: 2026-11-16T14:45:00Z
- Dataset: `10000.csv` (10000 rows, 10000 valid)
- Reference model: **XGBoost** (chosen by validation ROC-AUC)
- Split: train 6978 / validation 1518 / test 1504
- Every feature set is selected with the **training rows only**; the same split, model and metrics are used throughout.

| key | name | features | status | families | is_combined | note | accuracy | precision | recall | f1 | balanced_accuracy | true_negative | false_positive | false_negative | true_positive | false_positive_rate | false_negative_rate | specificity | precision_positive | n_samples | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | Stylo only | 115 | evaluated | stylometric | False |  | 0.8737 | 0.8920 | 0.9553 | 0.9226 | 0.7629 | 182.0000 | 137.0000 | 53.0000 | 1132.0000 | 0.4295 | 0.0447 | 0.5705 | 0.8920 | 1504.0000 | 0.9128 | 0.9741 |
| E2 | Writeprints only | 244 | evaluated | writeprints | False |  | 0.9176 | 0.9331 | 0.9646 | 0.9485 | 0.8538 | 237.0000 | 82.0000 | 42.0000 | 1143.0000 | 0.2571 | 0.0354 | 0.7429 | 0.9331 | 1504.0000 | 0.9600 | 0.9886 |
| E3 | GLTR only | 16 | evaluated | gltr | False |  | 0.7979 | 0.8270 | 0.9401 | 0.8799 | 0.6048 | 86.0000 | 233.0000 | 71.0000 | 1114.0000 | 0.7304 | 0.0599 | 0.2696 | 0.8270 | 1504.0000 | 0.7484 | 0.9166 |
| E4 | Watermark only | 7 | evaluated | watermark | False | watermark evidence; the key used at generation time is unknown, so the green list is arbitrary with respect to authorship | 0.7713 | 0.8010 | 0.9443 | 0.8668 | 0.5364 | 41.0000 | 278.0000 | 66.0000 | 1119.0000 | 0.8715 | 0.0557 | 0.1285 | 0.8010 | 1504.0000 | 0.6736 | 0.8632 |
| E5 | Combined selected features | 375 | evaluated | stylometric,writeprints,gltr | True |  | 0.9661 | 0.9701 | 0.9873 | 0.9787 | 0.9372 | 283.0000 | 36.0000 | 15.0000 | 1170.0000 | 0.1129 | 0.0127 | 0.8871 | 0.9701 | 1504.0000 | 0.9910 | 0.9975 |
| E6 | Combined features without GLTR | 359 | evaluated | stylometric,writeprints | True | combined set with the gltr family removed | 0.9674 | 0.9718 | 0.9873 | 0.9795 | 0.9404 | 285.0000 | 34.0000 | 15.0000 | 1170.0000 | 0.1066 | 0.0127 | 0.8934 | 0.9718 | 1504.0000 | 0.9913 | 0.9976 |
| E7 | Combined features without Writeprints | 131 | evaluated | stylometric,gltr | True | combined set with the writeprints family removed | 0.8843 | 0.8959 | 0.9654 | 0.9293 | 0.7742 | 186.0000 | 133.0000 | 41.0000 | 1144.0000 | 0.4169 | 0.0346 | 0.5831 | 0.8959 | 1504.0000 | 0.9306 | 0.9799 |
| E8 | Combined features without Stylo | 260 | evaluated | writeprints,gltr | True | combined set with the stylometric family removed | 0.9162 | 0.9330 | 0.9629 | 0.9477 | 0.8529 | 237.0000 | 82.0000 | 44.0000 | 1141.0000 | 0.2571 | 0.0371 | 0.7429 | 0.9330 | 1504.0000 | 0.9663 | 0.9905 |
| E9 | Combined features without Watermark | 375 | evaluated | stylometric,writeprints,gltr | True | the watermark branch is evidence-only and never enters the feature matrix, so this set is identical to E5 | 0.9661 | 0.9701 | 0.9873 | 0.9787 | 0.9372 | 283.0000 | 36.0000 | 15.0000 | 1170.0000 | 0.1129 | 0.0127 | 0.8871 | 0.9701 | 1504.0000 | 0.9910 | 0.9975 |

