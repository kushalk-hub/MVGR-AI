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
using a **locally cached causal language model**. It applies correct causal
shifting for next-token probabilities, aligns the tokenizer, excludes padding
and special tokens, and supports CPU (and GPU when present) with batching and a
configurable sequence length.

**Status in this environment: UNAVAILABLE.** `transformers` (and `torch`) are
not installed, and `allow_download` defaults to `false`, so the module reports
itself unavailable and emits an all-`NaN` placeholder table of the **19**
document features rather than downloading anything:

```
gltr_token_count, gltr_mean_prob, gltr_median_prob, gltr_std_prob,
gltr_min_prob, gltr_max_prob, gltr_mean_log_prob, gltr_perplexity,
gltr_mean_rank, gltr_median_rank, gltr_mean_log_rank, gltr_max_rank,
gltr_frac_rank_le_{10,100,1000}, gltr_frac_rank_gt_1000,
gltr_entropy_{mean,median,std}
```

Run `estimate-gltr` before any real run to size the job (token count, batches,
estimated time). GLTR-style predictability is a **statistical signal, not proof
of authorship**.

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
| `gltr` | `gltr_features.csv` | no (unavailable) |
| `watermark` | `watermark_results.csv` | evidence only |

* Sample IDs and graph IDs are preserved; metadata and category labels live in a
  **separate** table (`fused_multiview_metadata.csv`) and are never model inputs.
* Branch columns are namespaced `"<branch>__<feature>"`; missing branches are
  handled without error (the GLTR branch contributes 0 columns).
* Feature selection (configurable): drop columns above the missing threshold,
  constant columns, and one of each highly-correlated pair.
* A versioned schema (`fingerprint_feature_schema.json`) records the feature set
  version, fusion version, id column, null policy, branch availability,
  experiment declarations and the leakage note.
* The watermark branch is auxiliary and is **not** a model feature
  (`watermark_as_feature: false`), so an unsupported detection cannot become an
  artificial negative class.

**Pilot result:** 631 candidate columns → **408 selected** (30 dropped missing,
26 constant, 167 correlated).

## 6. Experiments (prepared, not trained)

Declared in `configs/fingerprint_config.yaml`; nothing is trained or split.

| Experiment | Branches | Columns |
|---|---|---:|
| E1 | lexical graph | 95 |
| E2 | syntactic graph | 95 |
| E3 | text fingerprinting | 140 |
| E4 | writeprints | 301 |
| E5 | GLTR-style | 0 (unavailable) |
| E6 | lexical + syntactic graphs | 190 |
| E7 | graphs + fingerprinting | 330 |
| E8 | graphs + fingerprinting + GLTR | 330 |
| E9 | all supported branches (+ optional watermark evidence) | 631 |

## 7. Outputs

`data/processed/fingerprinting/`:

| File | Rows × cols |
|---|---|
| `stylometric_features.csv` | 500 × 141 |
| `writeprints_features.csv` / `.npz` | 500 × 302 |
| `writeprints_vocabulary.json` | — |
| `gltr_features.csv` | 500 × 20 (unavailable/NaN) |
| `watermark_results.csv` | 500 × 13 |
| `fused_multiview_features.csv` | 500 × 409 |
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
| gltr | 19 | 1.000 | 1.000 |
| watermark | 8 (feature cols) | 1.000 | 1.000 |
| fused | 408 | 0.000 | 0.000 |

The GLTR column is entirely missing because the model is unavailable. The
stylometric maximum is the MSTTR / very-short-text family, undefined for texts
shorter than one 50-word segment. Fused columns have no missing values because
selection drops columns above the 30% missing threshold.

## 10. Execution and resources

Full pilot run (`fingerprint_pipeline.py all`): **500/500 samples**, 140 + 301 +
19 + 12 branch features, fused to 408; **≈100.8 s**, **peak 141.8 MB** RSS,
single process, CPU-only.

## 11. Offline guarantees

`offline_execution_report.md` records: `network required: false`, `auto model
download: false`, `gltr allow_download: false`, `transformers/torch installed:
false`, `spacy installed: true`, `watermark key configured: false`. spaCy and
the Hugging Face cache are used locally only.

## 12. Limitations

* Stylometric and writeprints features describe **style**, not authorship.
* The writeprints vocabulary is fit on the pilot corpus for extraction only; a
  real experiment must fit it on the train split to avoid leakage.
* GLTR is implemented but **not executed** here (no local causal LM); it is
  never downloaded automatically.
* Watermark verification is only meaningful for the exact scheme and key used at
  generation time; absence of detection is not evidence of human authorship.
* The pilot is an engineering sample; no formal split exists and no model was
  trained or evaluated.

## 13. Commands

```powershell
# whole pipeline (stylometric + writeprints + gltr + watermark + fusion + reports)
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py all

# individual branches
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py stylometric
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py writeprints
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py watermark
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py gltr          # reports unavailable without a local model
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py fuse
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py reports

# size the GLTR job before running it
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py estimate-gltr

# validation
.\.venv\Scripts\python.exe scripts\validate_fingerprint_features.py
```
