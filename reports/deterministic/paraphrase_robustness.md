# Paraphrase robustness

- Generated: 2026-11-16T14:45:03Z
- Dataset: `10000.csv`
- Model: **XGBoost**
- All rows come from `10000.csv`; **no paraphrased text was synthesised**.

## Per-category performance on the test split

| category | text_type | label | n | accuracy | precision | recall | f1 | false_positive_rate | false_negative_rate | mean_ai_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deeply paraphrased AI-generated | deep_paraphrased | 1 | 511 | 0.9980 | 1.0000 | 0.9980 | 0.9990 |  | 0.0020 | 0.9970 |
| Original Human | original | 0 | 69 | 0.7391 | 0.0000 |  |  | 0.2609 |  | 0.2567 |
| Original AI-generated | original | 1 | 195 | 0.9846 | 1.0000 | 0.9846 | 0.9922 |  | 0.0154 | 0.9835 |
| Paraphrased Human | paraphrased | 0 | 153 | 0.9346 | 0.0000 |  |  | 0.0654 |  | 0.0921 |
| Paraphrased AI-generated | paraphrased | 1 | 331 | 0.9879 | 1.0000 | 0.9879 | 0.9939 |  | 0.0121 | 0.9842 |
| Unspecified (BiScope) Human | unknown | 0 | 97 | 0.9175 | 0.0000 |  |  | 0.0825 |  | 0.1051 |
| Unspecified (BiScope) AI-generated | unknown | 1 | 148 | 0.9527 | 1.0000 | 0.9527 | 0.9758 |  | 0.0473 | 0.9318 |


## Findings

- **Deeply paraphrased AI-generated** (n=511): accuracy 0.9980, FPR nan, FNR 0.0020
- **Original Human** (n=69): accuracy 0.7391, FPR 0.2609, FNR nan
- **Original AI-generated** (n=195): accuracy 0.9846, FPR nan, FNR 0.0154
- **Paraphrased Human** (n=153): accuracy 0.9346, FPR 0.0654, FNR nan
- **Paraphrased AI-generated** (n=331): accuracy 0.9879, FPR nan, FNR 0.0121
- **Unspecified (BiScope) Human** (n=97): accuracy 0.9175, FPR 0.0825, FNR nan
- **Unspecified (BiScope) AI-generated** (n=148): accuracy 0.9527, FPR nan, FNR 0.0473

### Degradation relative to the original text of the same authorship

- **Deeply paraphrased AI-generated** vs Original AI-generated: accuracy +0.0134, FPR n/a, FNR -0.0134
- **Paraphrased Human** vs Original Human: accuracy +0.1955, FPR -0.1955, FNR n/a
- **Paraphrased AI-generated** vs Original AI-generated: accuracy +0.0033, FPR n/a, FNR -0.0033
- **Unspecified (BiScope) Human** vs Original Human: accuracy +0.1784, FPR -0.1784, FNR n/a
- **Unspecified (BiScope) AI-generated** vs Original AI-generated: accuracy -0.0319, FPR n/a, FNR +0.0319

- Worst false-positive rate over the human categories: 0.26087
- Worst detection rate over the AI categories: 0.952703

Robustness is claimed only where the numbers above support it.

## All models

The same breakdown for every trained model, so the robustness of the winning model can be compared with the alternatives:

| model | worst_human_fpr | worst_ai_detection_rate | robustness_supported |
| --- | --- | --- | --- |
| logistic_regression | 0.2609 | 0.9459 | False |
| random_forest | 0.3043 | 0.9122 | False |
| svm | 0.1449 | 0.9189 | True |
| xgboost | 0.2609 | 0.9527 | False |

