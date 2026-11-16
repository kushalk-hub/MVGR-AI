# Feature analysis

- Generated: 2026-11-16T14:20:27Z
- Dataset: `10000.csv` (10000 rows, 10000 valid)
- Feature coverage: 10000/10000 records, 375 candidate columns, missing fraction 0.000012
- Features used after the train-only selection: **375**

## The four branches

| Branch | Features | Role |
|---|---:|---|
| Stylometric | 140 | vocabulary, sentence, punctuation, POS, syntactic, readability |
| Writeprints | 301 | bounded char/word/POS n-grams, function words |
| GLTR | 19 | real GPT-2 next-token probability/rank/entropy/perplexity |
| Watermark | 12 | **evidence only** — never a model feature |

The watermark branch is not part of the feature matrix: it is written to the metadata table, so an unsupported detection can never become an artificial negative class.

## Train-only selection, per family

The selection rules are the ones the fusion layer uses (missing-value threshold, constant columns, correlation filtering), but they are fitted on the **training rows only** and then applied to validation and test.

| family | candidates | selected | dropped | max_missing_fraction | mean_missing_fraction |
| --- | --- | --- | --- | --- | --- |
| stylometric | 115 | 115 | 0 | 0.0004 | 0.0000 |
| writeprints | 244 | 244 | 0 | 0.0000 | 0.0000 |
| gltr | 16 | 16 | 0 | 0.0000 | 0.0000 |


- Candidates: 375
- Selected: 375
- Dropped (missing): 0
- Dropped (constant): 0
- Dropped (correlated): 0

## Univariate signal (training rows only)

The table below is the ROC-AUC of each selected feature on its own, computed on the training split. A value of 0.5 means the feature carries no directional signal; the `separation` column is the distance from chance, so it ranks the features the same way for both directions.

| feature | family | univariate_auc | direction | separation |
| --- | --- | --- | --- | --- |
| writeprints__wp_char_unigram_000 | writeprints | 0.1138 | Human | 0.7724 |
| stylometric__vocab_avg_word_length | stylometric | 0.7316 | AI | 0.4632 |
| writeprints__wp_pos_bigram_000 | writeprints | 0.6993 | AI | 0.3986 |
| stylometric__read_avg_syllables_per_word | stylometric | 0.6966 | AI | 0.3932 |
| stylometric__read_flesch_reading_ease | stylometric | 0.3205 | Human | 0.3591 |
| stylometric__vocab_word_length_hist_1 | stylometric | 0.3318 | Human | 0.3363 |
| writeprints__wp_char_trigram_006 | writeprints | 0.6622 | AI | 0.3244 |
| stylometric__pos_noun_ratio | stylometric | 0.6507 | AI | 0.3015 |
| writeprints__wp_summary_char_entropy | writeprints | 0.6486 | AI | 0.2971 |
| writeprints__wp_char_bigram_032 | writeprints | 0.6485 | AI | 0.2969 |
| writeprints__wp_pos_bigram_013 | writeprints | 0.6446 | AI | 0.2892 |
| stylometric__punct_comma_count | stylometric | 0.6441 | AI | 0.2883 |
| stylometric__vocab_word_length_hist_9 | stylometric | 0.6428 | AI | 0.2857 |
| writeprints__wp_char_bigram_009 | writeprints | 0.3572 | Human | 0.2856 |
| writeprints__wp_char_trigram_009 | writeprints | 0.6420 | AI | 0.2841 |
| writeprints__wp_pos_bigram_001 | writeprints | 0.6376 | AI | 0.2753 |
| stylometric__vocab_stopword_ratio | stylometric | 0.3643 | Human | 0.2714 |
| stylometric__vocab_max_word_length | stylometric | 0.6331 | AI | 0.2663 |
| stylometric__read_flesch_kincaid_grade | stylometric | 0.6331 | AI | 0.2662 |
| stylometric__vocab_word_length_std | stylometric | 0.6327 | AI | 0.2654 |
| stylometric__pos_share_aux | stylometric | 0.3709 | Human | 0.2581 |
| writeprints__wp_punct_pattern_002 | writeprints | 0.6285 | AI | 0.2571 |
| stylometric__pos_adjective_ratio | stylometric | 0.6269 | AI | 0.2537 |
| stylometric__sent_min_length | stylometric | 0.3805 | Human | 0.2390 |
| gltr__gltr_mean_prob | gltr | 0.6145 | AI | 0.2290 |


- 375 selected features have a computable univariate AUC.
- Univariate AUC ignores interactions and correlations; it is a description of the feature, not a model.

## Audit of the shipped selection

The feature matrix shipped by the fingerprinting pipeline selects its columns with the whole table it is given, so the selection has seen the evaluation rows. The table below re-runs the identical rules over **all** rows and compares the result with the train-only selection used for these experiments.

- Train-only selection: **375** features
- Whole-dataset selection (contaminated): **375** features
- Identical: **True**
- Only in the train-only selection: 0
- Only in the whole-dataset selection: 0

The original artifacts are kept unchanged, so the contaminated and the train-only selections can be compared directly.
