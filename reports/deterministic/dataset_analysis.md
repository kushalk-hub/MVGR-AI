# Dataset analysis — `10000.csv`

- Generated: 2026-11-16T14:20:22Z
- Dataset file: `10000.csv`
- Resolved path: `C:\Users\kusha\OneDrive\Documents\MVGR-AI\10000.csv`
- SHA-256: `736a7ea9da1835199a640f68788596185e00931f9496b9d3c7b39eac94257ad9`
- Rows in the CSV: **10000**
- Valid records (non-empty text, parsable label): **10000**
- Invalid records: **0** — none

## Label distribution (`0 = Human`, `1 = AI`)

- Human: **2148**
- AI: **7852**
- AI/Human imbalance ratio: 3.6555

## Composition

- Source dataset: {'PADBen': 7157, 'BiScope': 2843}
- Text type: {'deep_paraphrased': 3333, 'paraphrased': 3333, 'original': 1694, 'unknown': 1640}
- Paraphrase status: {'True': 6666, 'False': 1694, '': 1640}
- Graph category: {'original': 3334, 'deep_paraphrased': 3333, 'paraphrased': 3333}
- Domain: {'Unknown': 7157, 'Yelp': 1260, 'Essay': 669, 'Creative': 619, 'Arxiv': 195, 'Code': 100}
- AI model: {'Unknown': 10000}

## Data quality

- Duplicate `sample_id`: 0
- Duplicate texts (exact): 0
- Duplicate texts (whitespace-normalised): 0
- Text length (chars): min 10, median 123.0, max 16349

### Missing values per column (empty strings)

| column | empty |
| --- | --- |
| sample_id | 0 |
| text | 0 |
| label | 0 |
| source_dataset | 0 |
| source_task | 0 |
| text_type | 0 |
| paraphrase_status | 1640 |
| domain | 0 |
| ai_model | 0 |
| padben_split | 2843 |
| split | 0 |
| group_id | 0 |
| source_row_id | 0 |
| graph_category | 0 |


## Paraphrase families

- Family key: `group_id`
- Unique families: 9887
- Families with more than one member: 113 (226 rows)
- Families containing both a human and an AI version: 113

## Split

The split column shipped in the CSV disagrees with the repository's candidate manifest, so the resolved split is built as follows: the `man_20ddf272c292efbc8c8c` manifest is reused for every row it covers (PADBen), and the remaining rows (BiScope, which the source-scoped manifest does not cover) are assigned with the repository's own family-disjoint policy, the same seed and the same ratios.

- Resolved counts: {'train': 6978, 'validation': 1518, 'test': 1504}
- Families spanning more than one split: 0
- Duplicate texts spanning more than one split: 0
- Leakage-free: **True**

### Split composition

| split | label | n |
| --- | --- | --- |
| train | 1 | 5459 |
| train | 0 | 1519 |
| validation | 1 | 1208 |
| validation | 0 | 310 |
| test | 1 | 1185 |
| test | 0 | 319 |


### Split composition by source dataset

| split | source | n |
| --- | --- | --- |
| train | PADBen | 4987 |
| train | BiScope | 1991 |
| validation | PADBen | 1092 |
| validation | BiScope | 426 |
| test | PADBen | 1078 |
| test | BiScope | 426 |


### Split composition by text type

| split | text_type | n |
| --- | --- | --- |
| train | paraphrased | 2323 |
| train | deep_paraphrased | 2310 |
| train | original | 1204 |
| train | unknown | 1141 |
| validation | paraphrased | 526 |
| validation | deep_paraphrased | 512 |
| validation | unknown | 254 |
| validation | original | 226 |
| test | deep_paraphrased | 511 |
| test | paraphrased | 484 |
| test | original | 264 |
| test | unknown | 245 |


The split shipped inside the CSV (`split` column) was also checked: 0 families span more than one of its splits. It is group-safe too, but it does not follow the repository's split policy, so it is not used for these experiments.
