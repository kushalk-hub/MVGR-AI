# Interpretability

- Generated: 2026-11-16T14:45:05Z
- Dataset: `10000.csv`
- Model: **XGBoost**
- Feature importance describes *what the model used*. It is not evidence of causality, and it does not mean the feature caused the text to be machine-generated.

## Most influential features

### XGBoost

| rank | feature | importance |
| --- | --- | --- |
| 1 | writeprints__wp_pos_bigram_000 | 21.2726 |
| 2 | stylometric__vocab_total_words | 13.1183 |
| 3 | writeprints__wp_char_unigram_000 | 12.6382 |
| 4 | stylometric__punct_apostrophe_per_kchar | 8.6786 |
| 5 | gltr__gltr_mean_log_prob | 8.4164 |
| 6 | stylometric__punct_parenthesis_count | 8.3391 |
| 7 | writeprints__wp_word_bigram_015 | 8.3146 |
| 8 | stylometric__read_avg_syllables_per_word | 7.0601 |
| 9 | writeprints__wp_word_bigram_027 | 6.8234 |
| 10 | writeprints__wp_pos_bigram_013 | 6.2942 |
| 11 | stylometric__vocab_avg_word_length | 6.2886 |
| 12 | writeprints__wp_fw_she | 5.3999 |
| 13 | stylometric__sent_length_hist_8 | 5.3567 |
| 14 | stylometric__syn_passive_count | 5.3104 |
| 15 | stylometric__punct_comma_count | 5.0987 |


Family mix in the top 15: {'writeprints': 6, 'stylometric': 8, 'gltr': 1}

### SHAP (XGBoost)

| rank | feature | importance |
| --- | --- | --- |
| 1 | writeprints__wp_char_unigram_000 | 5.9757 |
| 2 | stylometric__vocab_avg_word_length | 1.5339 |
| 3 | writeprints__wp_punct_pattern_002 | 0.3771 |
| 4 | stylometric__vocab_word_length_std | 0.2735 |
| 5 | writeprints__wp_pos_bigram_013 | 0.2553 |
| 6 | writeprints__wp_punct_pattern_001 | 0.2482 |
| 7 | stylometric__punct_density | 0.2322 |
| 8 | stylometric__punct_comma_per_kchar | 0.2254 |
| 9 | stylometric__punct_period_per_kchar | 0.2076 |
| 10 | stylometric__pos_share_aux | 0.1971 |
| 11 | stylometric__vocab_richness_herdan | 0.1889 |
| 12 | writeprints__wp_char_bigram_000 | 0.1582 |
| 13 | stylometric__vocab_type_token_ratio | 0.1575 |
| 14 | gltr__gltr_entropy_mean | 0.1427 |
| 15 | writeprints__wp_char_trigram_015 | 0.1421 |


Family mix in the top 15: {'writeprints': 6, 'stylometric': 8, 'gltr': 1}

## Error profile

- Group sizes on the test split: {'true_positive': 1170, 'true_negative': 283, 'false_positive': 36, 'false_negative': 15}

Features that separate the false positives from the true negatives (largest shifts first):

| feature | false_positive_mean | true_negative_mean | fp_minus_tn |
| --- | --- | --- | --- |
| gltr__gltr_max_rank | 9317.4722 | 6813.2721 | 2504.2001 |
| stylometric__vocab_total_words | 126.3333 | 42.5406 | 83.7927 |
| stylometric__vocab_unique_words | 73.5278 | 28.5159 | 45.0119 |
| gltr__gltr_perplexity | 101.3979 | 143.4040 | -42.0061 |
| stylometric__punct_total_count | 25.8056 | 7.0177 | 18.7879 |
| stylometric__sent_length_range | 12.7222 | 2.8693 | 9.8530 |
| stylometric__syn_subordinate_count | 10.6111 | 2.7067 | 7.9044 |
| gltr__gltr_median_rank | 8.4028 | 15.5194 | -7.1167 |
| stylometric__syn_distinct_relations | 19.6389 | 13.5088 | 6.1301 |
| stylometric__punct_comma_count | 7.8056 | 2.3463 | 5.4593 |
| stylometric__read_flesch_reading_ease | 51.2096 | 56.5077 | -5.2980 |
| stylometric__sent_min_length | 11.4722 | 16.6608 | -5.1886 |
| stylometric__sent_count | 7.5278 | 2.4382 | 5.0896 |
| stylometric__sent_max_length | 24.1944 | 19.5300 | 4.6644 |
| stylometric__syn_distance_max | 20.1389 | 16.4382 | 3.7007 |


Features that separate the false negatives from the true positives:

| feature | false_negative_mean | true_positive_mean | fn_minus_tp |
| --- | --- | --- | --- |
| gltr__gltr_max_rank | 7185.4000 | 7451.5812 | -266.1812 |
| stylometric__vocab_total_words | 27.6667 | 90.0564 | -62.3897 |
| gltr__gltr_perplexity | 156.7754 | 109.6605 | 47.1150 |
| stylometric__read_flesch_reading_ease | 80.9647 | 39.0128 | 41.9519 |
| gltr__gltr_median_rank | 45.4000 | 13.7803 | 31.6197 |
| stylometric__vocab_unique_words | 23.9333 | 55.1650 | -31.2316 |
| stylometric__punct_total_count | 3.8000 | 15.8291 | -12.0291 |
| stylometric__punct_comma_per_kchar | 2.8305 | 12.1544 | -9.3239 |
| stylometric__read_flesch_kincaid_grade | 5.2257 | 11.9162 | -6.6904 |
| gltr__gltr_mean_rank | 432.0169 | 438.5553 | -6.5384 |
| stylometric__punct_comma_count | 0.6667 | 6.7470 | -6.0803 |
| stylometric__punct_period_per_kchar | 16.1599 | 10.5292 | 5.6307 |
| stylometric__syn_subordinate_count | 2.2667 | 6.3530 | -4.0863 |
| stylometric__sent_max_length | 16.3333 | 20.2709 | -3.9376 |
| stylometric__syn_distance_max | 13.4000 | 17.0333 | -3.6333 |

