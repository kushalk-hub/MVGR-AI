# MVGR-AI dataset audit

Dataset: `data\processed\mvgr_ai_combined.parquet`
SHA-256: `9c824a635ed66bc02421c572381910aacb0198209a0e1f8e2d9111728a2eb917`
Rows: **128,709**

## Label semantics

- Observed: {'0': 'human_authored', '1': 'ai_authored'}
- Counts: {'0': 36349, '1': 92360}
- Labels describe authorship only. Paraphrase condition is not a label.

## Sources and splits

| Source | Rows |
|---|---:|
| PADBen | 80,651 |
| BiScope | 48,058 |

| Split | Rows |
|---|---:|
| train | 108,211 |
| validation | 11,861 |
| test | 8,637 |

## Lineage

- Families: 102,530
- Resolution: {'resolved': 80651, 'exact_text_only': 48058}
- Families with both labels: 26179
- Families spanning splits: 0

> BiScope is grouped by exact normalized text only. Paraphrase families
> cannot be reconstructed from the current export, so pooled paraphrase
> claims require a recorded waiver or a source-scoped design.

## Gates

Overall: **FAIL**

| Check | Status | Details |
|---|---|---|
| unique_sample_ids | pass | 128709 unique sample ids |
| text_present | warn | 281 texts shorter than 40 characters |
| label_domain | pass | binary labels 0/1 only |
| family_disjoint | pass | 102530 families, none spanning splits |
| text_hash_disjoint | pass | no normalized text crosses a split |
| split_ratios | warn | deviation > 0.03: train=0.841 (target 0.700), validation=0.092 (target 0.150), test=0.067 (target 0.150) |
| label_coverage | pass | all 7 task/label test cells have >= 30 rows |
| class_imbalance | pass | AI:human ratio 2.54 |
| lineage_resolution | fail | 48058 rows rely on exact-text-only or unresolved lineage in ['BiScope']; paraphrase families may cross splits. Source-scoped claims are possible; pooled claims require an explicit waiver. |
| condition_confounding | warn | 16 confounded cells covering 126492 rows; condition effects cannot be estimated there |

## Confounding

16 condition/label confounding findings.

- `BiScope/BiScope` text_type=`paraphrased`: condition_predicts_label (21,785 rows). Cannot be used as evidence of a paraphrase effect; report as confounded.
- `PADBen/exhaustive-task5` text_type=`deep_paraphrased`: condition_predicts_label (3,739 rows). Cannot be used as evidence of a paraphrase effect; report as confounded.
- `PADBen/sentence-pair-task5` text_type=`deep_paraphrased`: condition_predicts_label (12,492 rows). Cannot be used as evidence of a paraphrase effect; report as confounded.
- `PADBen/sentence-pair-task5` text_type=`original`: condition_predicts_label (3,437 rows). Cannot be used as evidence of a paraphrase effect; report as confounded.
- `BiScope/BiScope` text_type=`original`: label_predicts_condition (4,388 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/exhaustive-task1` text_type=`paraphrased`: label_predicts_condition (3,644 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/exhaustive-task1` text_type=`paraphrased`: label_predicts_condition (3,813 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/exhaustive-task2` text_type=`unknown`: label_predicts_condition (1,047 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/exhaustive-task2` text_type=`unknown`: label_predicts_condition (3,692 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/exhaustive-task5` text_type=`deep_paraphrased`: label_predicts_condition (3,739 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/sentence-pair-task1` text_type=`paraphrased`: label_predicts_condition (12,312 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/sentence-pair-task1` text_type=`paraphrased`: label_predicts_condition (12,418 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/sentence-pair-task2` text_type=`unknown`: label_predicts_condition (11,521 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/sentence-pair-task2` text_type=`unknown`: label_predicts_condition (12,536 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/sentence-pair-task5` text_type=`original`: label_predicts_condition (3,437 rows). Condition-specific label; pooled condition claims are invalid.
- `PADBen/sentence-pair-task5` text_type=`deep_paraphrased`: label_predicts_condition (12,492 rows). Condition-specific label; pooled condition claims are invalid.

## Text statistics

- Empty: 0
- Shorter than 40 chars: 281
- Duplicate normalized texts: 0
- Median length: 137 characters

## Warnings

- No generator metadata is available for any row; model-specific claims are unsupported.
- Some rows have text_type=unknown; condition-specific claims must exclude them.
