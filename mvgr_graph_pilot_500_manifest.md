# 500-row Neo4j Graph Pilot Manifest

- Input: `final_data/mvgr_graph_pilot_10000.csv` (read only).
- Input SHA-256: 736a7ea9da1835199a640f68788596185e00931f9496b9d3c7b39eac94257ad9.
- Output: `final_data/mvgr_graph_pilot_500.csv`.
- Output rows: 500; output SHA-256: d78dd004dd90ef2a3d1e6479b896b9382985497d94b94b0b832aa25e5c7f5a1a.
- Deterministic seed: 20261114.
- Source columns, labels, task names, text types, and split values are preserved. This sample adds no further fields.
- Proportional stratification: `graph_category`, then `source_dataset`, `source_task`, `label`, and existing `split`.
- This is an engineering/storage pilot, not a new evaluation split or a change to the full 10,000-row graph sample.

## Category and label counts

| graph_category | Rows | label 0 | label 1 |
|---|---:|---:|---:|
| original | 167 | 58 | 109 |
| paraphrased | 167 | 49 | 118 |
| deep_paraphrased | 166 | 0 | 166 |

## Source counts

| source_dataset | Rows |
|---|---:|
| PADBen | 357 |
| BiScope | 143 |

## Proportional sampling strata

| graph_category | source / task / label / split | Available in 10k sample | Selected |
|---|---|---:|---:|
| original | BiScope / BiScope / 0 / test | 40 | 2 |
| original | BiScope / BiScope / 0 / train | 175 | 9 |
| original | BiScope / BiScope / 0 / validation | 35 | 2 |
| original | BiScope / BiScope / 1 / test | 184 | 9 |
| original | BiScope / BiScope / 1 / train | 876 | 44 |
| original | BiScope / BiScope / 1 / validation | 188 | 9 |
| original | PADBen / exhaustive-task2 / 0 / train | 57 | 3 |
| original | PADBen / exhaustive-task2 / 0 / validation | 3 | 0 |
| original | PADBen / exhaustive-task2 / 1 / train | 189 | 10 |
| original | PADBen / exhaustive-task2 / 1 / validation | 21 | 1 |
| original | PADBen / sentence-pair-task2 / 0 / test | 3 | 0 |
| original | PADBen / sentence-pair-task2 / 0 / train | 641 | 32 |
| original | PADBen / sentence-pair-task2 / 0 / validation | 12 | 1 |
| original | PADBen / sentence-pair-task2 / 1 / test | 20 | 1 |
| original | PADBen / sentence-pair-task2 / 1 / train | 650 | 33 |
| original | PADBen / sentence-pair-task2 / 1 / validation | 44 | 2 |
| original | PADBen / sentence-pair-task5 / 0 / train | 189 | 9 |
| original | PADBen / sentence-pair-task5 / 0 / validation | 7 | 0 |
| paraphrased | BiScope / BiScope / 1 / test | 202 | 10 |
| paraphrased | BiScope / BiScope / 1 / train | 948 | 48 |
| paraphrased | BiScope / BiScope / 1 / validation | 195 | 10 |
| paraphrased | PADBen / exhaustive-task1 / 0 / train | 203 | 10 |
| paraphrased | PADBen / exhaustive-task1 / 0 / validation | 22 | 1 |
| paraphrased | PADBen / exhaustive-task1 / 1 / train | 212 | 11 |
| paraphrased | PADBen / exhaustive-task1 / 1 / validation | 23 | 1 |
| paraphrased | PADBen / sentence-pair-task1 / 0 / test | 22 | 1 |
| paraphrased | PADBen / sentence-pair-task1 / 0 / train | 695 | 35 |
| paraphrased | PADBen / sentence-pair-task1 / 0 / validation | 44 | 2 |
| paraphrased | PADBen / sentence-pair-task1 / 1 / test | 21 | 1 |
| paraphrased | PADBen / sentence-pair-task1 / 1 / train | 701 | 35 |
| paraphrased | PADBen / sentence-pair-task1 / 1 / validation | 45 | 2 |
| deep_paraphrased | PADBen / exhaustive-task5 / 1 / train | 695 | 34 |
| deep_paraphrased | PADBen / exhaustive-task5 / 1 / validation | 73 | 4 |
| deep_paraphrased | PADBen / sentence-pair-task5 / 1 / test | 74 | 4 |
| deep_paraphrased | PADBen / sentence-pair-task5 / 1 / train | 2338 | 116 |
| deep_paraphrased | PADBen / sentence-pair-task5 / 1 / validation | 153 | 8 |

## Superseded preliminary graph size estimate

The following original one-view regex estimate is retained as sampling history only. It is not an estimate for the current spaCy two-view loader. The loader computes exact spaCy token and edge projections from these 500 rows before connecting to Aura, then applies the 180,000-node / 360,000-relationship safety caps. Use the loader's printed pre-load counts and verify actual Aura counts after import.

- Tokenizer: Unicode regex `\w+|[^\w\s]`; one node per token occurrence.
- Edges: undirected co-occurrence pairs within a radius of two positions, stored once in canonical position order.
- Token occurrences: 52,176; co-occurrence edges: 102,852.
- Estimated Neo4j nodes: 53,176 (sample + text-graph + token-occurrence nodes).
- Estimated Neo4j relationships: 155,528 (sample-to-graph + graph-to-token + co-occurrence).
- These are pre-load counts; verify the actual AuraDB counts after import.

No graph records are created by this sampler.
