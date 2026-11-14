# 11. Offline fingerprinting, GLTR and watermark verification

This document covers the four additional offline analysis techniques added on
top of the completed graph pipeline, plus the feature-fusion layer that joins
them:

1. Text fingerprinting (stylometric analysis) — `fingerprinting/stylometry.py`
   and its `*_features.py` modules.
2. Writeprints-style sparse stylometry — `fingerprinting/writeprints.py`.
3. GLTR-style token probability/rank analysis — `gltr/`.
4. Statistical watermark verification — `watermark/statistics.py`,
   `watermark/unigram.py`, `watermark/topic.py`.
5. Multi-view feature fusion — `fusion/multiview_feature_fusion.py`.

Everything runs **offline**: no API calls, no cloud services, no telemetry, no
automatic model downloads. The existing graph pipeline, the immutable datasets,
Neo4j Aura, the provenance digests and the candidate/approved manifests are not
touched.

> Scope: this is the **500-sample engineering pilot**. It performs feature
> extraction only. No model is trained, no formal split is generated, and the
> test set is never accessed (no approved manifest exists).

## 1. Module A — stylometric text fingerprinting

`fingerprinting/stylometry.py` is the orchestrator; each family is a separate,
independently testable module. It reuses the shared tokenizer/sentence splitter
and the local spaCy model (`en_core_web_sm`); when spaCy is absent the POS,
syntactic and readability-syllable features degrade to `NaN` (never imputed).

| Family | Module | Features |
|---|---|---:|
| Vocabulary | `vocabulary_features.py` | 26 |
| Sentence | `sentence_features.py` | 19 |
| Punctuation | `punctuation_features.py` | 28 |
| POS | `pos_features.py` | 32 |
| Syntactic | `syntactic_features.py` | 30 |
| Readability | `readability_features.py` | 5 |
| **Total** | | **140** |

* Vocabulary: word/unique counts, TTR, root TTR (Guiraud), corrected TTR,
  hapax ratio, MSTTR, average word length, word-length histogram, stopword
  ratio, lexical diversity.
* Sentence: count, mean/median/min/max, variance/standard deviation, a fixed
  sentence-length histogram, long- and short-sentence ratios.
* Punctuation: per-mark frequencies, punctuation density and a punctuation
  character histogram.
* POS: nine coarse ratios (noun, verb, adjective, adverb, pronoun, determiner,
  preposition, conjunction, …) and the full POS tag distribution.
* Syntactic: dependency-relation distribution (top-15 + `other`), mean
  dependency distance, relation diversity/entropy, parse-tree depth, passive
  and subordinate-clause frequencies (from spaCy dependency labels).
* Readability: Flesch Reading Ease, Flesch–Kincaid grade, average syllables per
  word and average words per sentence — computed from a built-in syllable
  counter, so `textstat`/`nltk` are not required.

## 2. Module B — Writeprints-style sparse features

`fingerprinting/writeprints.py` emits a **bounded** sparse feature block. The
vocabulary is selected by document frequency with a deterministic name
tie-break, so the feature order is reproducible; sparse vectors are written as a
CSR `.npz` alongside the dense CSV.

| Group | Vocab / count |
|---|---:|
| Character unigrams | 30 |
| Character bigrams | 40 |
| Character trigrams | 40 |
| Word unigrams | 60 |
| Word bigrams | 30 |
| POS unigrams | 17 |
| POS bigrams | 24 |
| Punctuation patterns | 16 |
| Function-word frequencies | 40 |
| Summary (sentence length + vocab diversity) | 4 |
| **Total** | **301** |

Normalization is `frequency` (configurable: `raw_count | frequency | tfidf |
l2`). The vocabulary limit is configurable and deliberately small to avoid
uncontrolled expansion.

> This is a **Writeprints-inspired** implementation. It is **not** a claim that
> the original proprietary or published Writeprints methodology has been
> reproduced exactly.

## 3. Module C — GLTR-style token analysis

`gltr/` implements the full pipeline (`local_model.py`, `token_probability.py`,
`token_rank.py`, `entropy_analysis.py`, `document_features.py`, `pipeline.py`)
using a **locally stored causal language model**. It applies correct causal
shifting for next-token probabilities, aligns the tokenizer, excludes padding
and special tokens, and runs on CPU with batching and a configurable sequence
length.

**Status in this environment: AVAILABLE and EXECUTED** on all 500 pilot
samples.

| Setting | Value |
|---|---|
| Reference model | `gpt2` (124M parameters) |
| Pinned revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Storage | `models/gltr/gpt2/` (≈478 MB; `config.json` + `model.safetensors` + tokenizer + `MODEL_INFO.json`) |
| Parameters / vocab | 124,439,808 / 50,257 |
| Device / dtype | `cpu` / `float32` (no CUDA; `cuda_available: false`) |
| Sequence length | `max_length: 256` tokens, `stride: 128` (sliding window) |
| Batch size | 4 (dynamically padded, attention-masked) |
| Runtime | `torch 2.15.0+cpu`, `transformers 5.23.0` |
| Download policy | `allow_download: false`, `local_files_only: true` |

The 19 document features are unchanged (exact names and ordering preserved):

```
gltr_token_count, gltr_mean_prob, gltr_median_prob, gltr_std_prob,
gltr_min_prob, gltr_max_prob, gltr_mean_log_prob, gltr_perplexity,
gltr_mean_rank, gltr_median_rank, gltr_mean_log_rank, gltr_max_rank,
gltr_frac_rank_le_{10,100,1000}, gltr_frac_rank_gt_1000,
gltr_entropy_{mean,median,std}
```

How the values are computed (all real GPT-2 inference, never synthetic):

* For token position `t` the model's logits at position `t-1` predict it — the
  causal shift is explicit, so no token is scored using its own future.
* Probabilities are the softmax of the shifted logits gathered at the realised
  next token; ranks are `1 + #{vocabulary items with a strictly higher logit}`
  against the **full 50,257-token** vocabulary; entropy is measured over the
  shifted distribution; perplexity is `exp(-mean log p)`.
* Documents longer than 256 tokens are split into overlapping windows and each
  target token is counted exactly once (never double-counted, never truncated
  away); empty and single-token texts yield `gltr_token_count = 0` with the
  remaining features left empty (`NaN`), never zero.

`estimate-gltr` sizes a run before executing it. GLTR-style predictability is a
**statistical signal, not proof of authorship**.


## 4. Module D — statistical watermark verification

`watermark/` is a modular verification framework. It never hardcodes or prints a
secret key; the key is read from the environment variable `MVGR_WATERMARK_KEY`.

* **Unigram green-list detector** (`unigram.py`, Kirchenbauer-style): gamma
  green-list fraction, configurable conditioning window, z-score and a
  normal/binomial p-value; a detection requires the z-score **and** p-value to
  clear their thresholds.
* **Topic-conditioned detector** (`topic.py`): interface only. It returns
  `unsupported` until a known generation + verification protocol is configured.
* **Statistics** (`statistics.py`): z-score, normal survival function, exact
  binomial p-value.

Per-sample evidence columns: `watermark_status`, `watermark_provider`,
`watermark_detected`, `watermark_confidence`, `watermark_z_score`,
`watermark_green_count`, `watermark_green_proportion`,
`watermark_expected_proportion`, `watermark_p_value`,
`watermark_watermarked_token_count`, `watermark_topic_status`,
`watermark_details` (8 numeric feature columns).

Statuses are `detected | not_detected | unsupported | inconclusive`. With no key
configured, all 500 pilot rows are `unsupported` — the framework **never** turns
"unsupported" into a negative (human) class. Absence of a watermark is never
evidence of human authorship. Incompatible/unknown schemes return
`UNSUPPORTED_OR_UNKNOWN_WATERMARK`.

## 5. Module E — multi-view feature fusion

`fusion/multiview_feature_fusion.py` combines six branches while keeping
identifiers, metadata and labels separate:

| Branch | Source | Available |
|---|---|---|
| `lexical_graph` | `data/processed/graph_features.csv` (`glex_*`) | yes |
| `syntactic_graph` | `data/processed/graph_features.csv` (`gsyn_*`) | yes |
| `stylometric` | `stylometric_features.csv` | yes |
| `writeprints` | `writeprints_features.csv` | yes |
| `gltr` | `gltr_features.csv` | yes (16 of 19 selected) |
| `watermark` | `watermark_results.csv` | evidence only |

* Sample IDs and graph IDs are preserved; metadata and category labels live in a
  **separate** table (`fused_multiview_metadata.csv`) and are never model inputs.
* Branch columns are namespaced `"<branch>__<feature>"`; missing branches are
  handled without error (a branch with no values contributes 0 columns).
* Feature selection (configurable): drop columns above the missing threshold,
  constant columns, and one of each highly-correlated pair.
* A versioned schema (`fingerprint_feature_schema.json`) records the feature set
  version, fusion version, id column, null policy, branch availability,
  experiment declarations and the leakage note.
* The watermark branch is auxiliary and is **not** a model feature
  (`watermark_as_feature: false`), so an unsupported detection cannot become an
  artificial negative class.

**Pilot result:** 650 candidate columns → **424 selected** (30 dropped missing,
26 constant, 170 correlated). GLTR contributes 16 columns; 3 GLTR columns
(`gltr_token_count`, `gltr_mean_log_rank`, `gltr_frac_rank_gt_1000`) are dropped
as correlated redundancies of retained features.

## 6. Experiments (prepared, not trained)

Declared in `configs/fingerprint_config.yaml`; nothing is trained or split.

| Experiment | Branches | Columns |
|---|---|---:|
| E1 | lexical graph | 95 |
| E2 | syntactic graph | 95 |
| E3 | text fingerprinting | 140 |
| E4 | writeprints | 301 |
| E5 | GLTR-style | 19 |
| E6 | lexical + syntactic graphs | 190 |
| E7 | graphs + fingerprinting | 330 |
| E8 | graphs + fingerprinting + GLTR | 349 |
| E9 | all supported branches (+ optional watermark evidence) | 650 |

## 7. Outputs

`data/processed/fingerprinting/`:

| File | Rows × cols |
|---|---|
| `stylometric_features.csv` | 500 × 141 |
| `writeprints_features.csv` / `.npz` | 500 × 302 |
| `writeprints_vocabulary.json` | — |
| `gltr_features.csv` | 500 × 20 (real GPT-2 values) |
| `gltr_features.checkpoint.csv` (+ `.meta.json`) | resumable GLTR checkpoint |
| `watermark_results.csv` | 500 × 13 |
| `fused_multiview_features.csv` | 500 × 425 |
| `fused_multiview_metadata.csv` | 500 × 13 |
| `fingerprint_feature_schema.json` | — |

`outputs/fingerprint_audit/`: `fingerprint_extraction_report.{json,md}`,
`feature_comparison_report.md`, `offline_execution_report.md`,
`fingerprint_validation_report.{json,md}`.

Existing graph feature artifacts are never overwritten; the new outputs are
versioned separately.

## 8. Validation

`scripts/validate_fingerprint_features.py` — read-only, **41/41 checks PASS**:
artifact presence, row counts, unique sample IDs, sample-set alignment, schema
keys and column list, leakage (identifiers/metadata/labels never features,
metadata preserved separately), finite/non-negative/range value checks,
missing-value thresholds, watermark status validity, the
"no-false-negative-class" rule, and GLTR availability consistency.

## 9. Missing values

Undefined features are emitted as `NaN` and never imputed.

| Branch | Features | Overall missing | Max column missing |
|---|---:|---:|---:|
| stylometric | 140 | 0.015 | 0.736 |
| writeprints | 301 | 0.000 | 0.000 |
| gltr | 19 | 0.000 | 0.000 |
| watermark | 8 (feature cols) | 1.000 | 1.000 |
| fused | 424 | 0.000 | 0.000 |

The GLTR branch is fully populated: all 19 features are present and finite for
all 500 samples (0 processing errors). The stylometric maximum is the MSTTR /
very-short-text family, undefined for texts shorter than one 50-word segment.
The watermark feature columns are entirely missing because no watermark key is
configured (all rows are `unsupported`). Fused columns have no missing values
because selection drops columns above the 30% missing threshold.

## 10. Execution and resources

Full pilot run (`fingerprint_pipeline.py all --no-resume`, offline):
**500/500 samples**, 140 + 301 + 19 + 12 branch features, fused to 424;
**1261.4 s** wall time, single process, CPU-only.

GLTR-only measurement (500 samples, dedicated benchmark):

| Metric | Value |
|---|---|
| Wall time | 1017.7 s (≈17 min); ≈2.0 s/sample |
| Peak working set (RSS) | 1320.2 MB |
| Model load | ≈21 s (478 MB from `models/gltr/gpt2/`) |
| Forwards | 1 per window batch (dynamically padded, batch 4) |
| Token counts | min 8, max 2320, mean 110.1 per document |
| Samples with features | 500 / 500 |
| Processing errors | 0 |

The pipeline's own `tracemalloc` peak (887 MB) tracks Python allocations only
and excludes the native PyTorch model tensors, so the ≈1.32 GB working-set
figure above is the number to use for memory planning. The model is loaded
exactly once per run; the analyzer holds a single copy.

## 11. Offline guarantees

`offline_execution_report.md` records: `network required: false`, `auto model
download: false`, `gltr allow_download: false`, `gltr local_files_only: true`,
`gltr execution: executed`, `transformers installed: true`, `torch installed:
true`, `spacy installed: true`, `cuda available: false`, `watermark key
configured: false`. spaCy and the Hugging Face cache are used locally only.

The model is downloaded **once** by `scripts/download_gltr_model.py` (the only
code path that touches the network for GLTR). Every experiment then sets
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, loads from `models/gltr/gpt2/`
with `local_files_only=True`, and never contacts the Hub. The full 500-sample
`all --no-resume` run above was executed with both offline flags set and made no
network calls.

## 12. Limitations

* Stylometric and writeprints features describe **style**, not authorship.
* The writeprints vocabulary is fit on the pilot corpus for extraction only; a
  real experiment must fit it on the train split to avoid leakage.
* GLTR is executed with the GPT-2 base model as a **reference**, not as an
  authorship detector: high predictability is not proof of AI authorship and low
  predictability is not proof of human authorship. GPT-2 (2019) is smaller and
  older than the models that generate modern text, so absolute probabilities are
  reference-relative.
* Per-token features are averaged over a document, so very short texts (min 8
  scored tokens here) are noisier than long ones.
* Watermark verification is only meaningful for the exact scheme and key used at
  generation time; absence of detection is not evidence of human authorship.
* The pilot is an engineering sample; no formal split exists and no model was
  trained or evaluated.

## 13. Setup, download and commands

**Install the CPU-only GLTR extra** (one time). The `+cpu` PyTorch wheel comes
from the PyTorch CPU index, so no CUDA runtime is pulled in:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\gltr.txt
```

This installs `torch==2.15.0+cpu`, `transformers==5.23.0`,
`tokenizers>=0.23.1,<0.24.0`, `safetensors>=0.8.0`.

**Download the reference model** (one time, network required). This is the only
step that touches the network:

```powershell
# downloads gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e -> models\gltr\gpt2
.\.venv\Scripts\python.exe scripts\download_gltr_model.py
# re-download over an existing copy:
.\.venv\Scripts\python.exe scripts\download_gltr_model.py --force
```

**Run offline** (no network). Every experiment loads the model from
`models\gltr\gpt2\` with `local_files_only=True`; the flags below make an
accidental Hub call fail fast:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

# whole pipeline (stylometric + writeprints + gltr + watermark + fusion + reports)
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py all
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py all --no-resume   # ignore the GLTR checkpoint

# individual branches
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py stylometric
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py writeprints
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py watermark
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py gltr
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py gltr --limit 50
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py fuse
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py reports

# size the GLTR job before running it
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py estimate-gltr

# validation
.\.venv\Scripts\python.exe scripts\validate_fingerprint_features.py
```

Long GLTR runs checkpoint to
`data\processed\fingerprinting\gltr_features.checkpoint.csv` (plus a
`.meta.json` keyed to model/revision/max_length/stride/feature set). A re-run
resumes by default; `--no-resume` reprocesses everything.

