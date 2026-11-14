# MVGR-AI — completed work status

_Last updated: 2026-11-14. Every number below was produced by running the code in
this repository, not copied from a plan._

This file is a single snapshot of **what is actually finished** versus **what is
blocked or not started**. The authoritative design docs live in
[`reports/final/`](reports/final/README.md).

---

## Snapshot

| Item | Value |
|---|---|
| Test suite | **132 passed, 2 skipped** (`python -m pytest`) |
| Dataset (canonical) | `data/processed/mvgr_ai_combined.parquet`, **128,709 rows**, SHA-256 `9c824a63…b917`, version `mvgr-ai/2026-11-13` |
| Provenance | byte-identical to immutable `datasets/processed/` (`match: true`) |
| `features.v1` | **77** deterministic features |
| Neo4j pilot (live) | **500 samples, 1,000 graphs, 102,822 token nodes, 254,028 relationships** |
| Graph feature extraction | **complete** — 500/500 samples, 1,000/1,000 views, 147.6 s |
| Offline fingerprinting | **complete** — 500/500 samples, stylometric 140 + writeprints 301 + GLTR 19 (unavailable) + watermark 12 → fused 408, ~100.8 s |
| Split manifest | **not approved** (3 candidates exist) |
| Models trained | **none** |
| Frontend | **not implemented** (design doc only) |

---

## 1. Repository and scaffolding — DONE

* Repository restructured into a single shared tree; legacy `person1/`/`person2/`
  folders removed and all imports rewritten.
* `shared/paths.py` central path/config loading (no module hard-codes a working
  directory); `configs/*.yaml` for dataset, graph, experiment, evaluation, and
  graph-feature configuration.
* `mvgr.py` CLI with subcommands: `audit`, `lineage`, `split-manifest`,
  `approve-manifest`, `verify-manifest`, `graph-build`, `features`, `baseline`,
  `deterministic`, `neo4j`, `serve`, `contracts`.
* `.env.example` documents credentials by name only; no secrets in configs.

## 2. Contracts layer — DONE

`shared/contracts/` — six JSON Schemas with fail-closed validators, content-
addressed IDs, and a single version source of truth:

* Schemas: graph record, prediction, split manifest, family index, experiment,
  analysis report.
* Validators raise `ContractError` on any violation (an invalid artifact is an
  error, never a warning); `validate_analysis_report` adds the honesty rules.
* Content-addressed IDs: `cfg_`, `gb_`, `split_`, `man_`, `exp_`, `rep_`.
* Versions: `graph_schema 1.0.0`, `prediction_schema 1.0.0`,
  `manifest_schema 1.0.0`, `family_index_schema 1.0.0`, `experiment_schema 1.0.0`,
  `report_schema 1.0.0`, `split.v1`, `tokenizer.v1`, `features.v1`.

## 3. Dataset audit — DONE (gates run on the real corpus)

`dataset_audit/` runs end-to-end on the 128,709-row corpus:

* Inventory, provenance digest verification, label mapping, family lineage,
  leakage and confounding gates.
* Findings: 2.54 AI:human imbalance; sources PADBen 80,651 / BiScope 48,058;
  102,530 families (pair-level, PADBen only); no duplicate normalized texts;
  length confound (human median 222 vs AI 816 chars).
* Gate status **FAIL** on `lineage_resolution` (BiScope has no paraphrase-family
  link — a data limitation, not a code defect). 16 confounded cells recorded.
* Three candidate split manifests generated (see §11); none approved.
* Outputs in `reports/dataset_audit/` and `outputs/`.

## 4. Deterministic features and scoring — DONE

* `deterministic/` implements **77 `features.v1` features** across independently
  testable families (clustering, degree distribution, centrality, density,
  shortest path, community, connectivity, …).
* Validation-selected, frozen scoring rule (`deterministic/scoring.py`); the
  null-undefined policy and guards are shared with the graph pipeline.

## 5. Baseline — DONE

`baseline/tfidf_logistic.py` — the required TF-IDF + logistic reference model
with test-split locking.

## 6. Fingerprinting and watermark — DONE

* `fingerprinting/` — SHA-256, n-gram Jaccard, MinHash/LSH reference index
  (`sha256.py`, `ngram.py`, `minhash.py`, `index.py`).
* `watermark/verifier.py` — provider-scoped verification that defaults to
  `unsupported`; absence of a watermark is never reported as human authorship.

## 7. Graph engine — DONE

* `graph_engine/graph_builder/` — lexical and syntactic builders.
* `graph_engine/graph_export/` — versioned JSONL export with build manifests.
* `graph_engine/neo4j/` — connection, schema, idempotent loader, queries.
* `graph_engine/gnn/` — dataset/model/train/predict package (PyTorch + PyG are an
  optional extra; **not installed** in the reference environment).

## 8. Neo4j pilot — LOADED and VERIFIED

The 500-sample mixed pilot is live on Neo4j Aura under build
`mvgr_pilot500_20261114_spacy_multiview_r1`:

* 500 samples, 1,000 graphs (500 `lexical_cooccurrence`, 500 `syntactic_dependency`),
  102,822 token nodes, 104,322 total nodes, 254,028 relationships.
* Edge counts: `CO_OCCURS` 101,322, `DEPENDS_ON` 48,884; `HAS_GRAPH` 1,000,
  `HAS_TOKEN` 102,822.
* 500/500 samples have exactly two views; 0 cross-graph edges; constraints
  `sample_id_unique`, `graph_id_unique`, `token_node_id_unique`.
* Read-only verification: **24 pass, 0 warn, 0 fail** →
  `outputs/graph_audit/neo4j_verification_report.{json,md}`.
* **Note:** the pilot input SHA-256 is unchanged (`d78dd004…7f5a1a`); no graph
  records were modified.

## 9. Graph feature pipeline (Phases A–F) — DONE

Scripts: `scripts/verify_neo4j_graphs.py`, `scripts/extract_graph_features.py`,
`scripts/prepare_ml_dataset.py`, `scripts/validate_graph_features.py`,
`configs/graph_feature_config.yaml`, `tests/test_graph_features.py`.

Full extraction run (`--no-resume`): **500/500 samples, 1,000/1,000 views,
102,822 nodes, 147.6 s, 0 failed batches.**

| Output | Rows × cols | Size |
|---|---|---|
| `data/processed/graph_features.csv` | 1,000 × 115 (95 features) | 1.15 MB |
| `data/processed/sample_multiview_features.csv` | 500 × 53 (40 features) | 0.29 MB |
| `data/processed/node_features.csv` | 102,822 × 22 (14 features) | 29.1 MB |
| `data/processed/feature_schema.json` | — | 66.9 KB |
| `data/processed/ml_ready_features.csv` | 500 × 13 | 0.10 MB |
| `data/processed/ml_ready_metadata.csv` | 500 × 12 | 0.06 MB |
| `data/processed/ml_dataset_report.json` | — | 8.1 KB |
| `outputs/graph_audit/feature_extraction_report.{json,md}` | — | 16.6 KB |

* Feature families: structural, lexical, syntactic, and cross-view multi-view
  (per-view aggregates, deltas, cosine/Jensen–Shannon similarities).
* Validation (Phase F): **50 pass, 0 warn, 0 fail** — schema, coverage,
  duplicates, missing/invalid values, two-view association, aggregation
  consistency, leakage, and a live-DB cross-check that matches exactly.
* Missing values are all by design: view-specific nulls on the other view's rows,
  plus source-missing metadata (`padben_split`, `paraphrase_status`) and empty
  `ent_type` for tokens with no entity. No predictive column is entirely null.
* Leakage: `graph_view_split_consistency`, `group_split_purity`,
  `no_identifier_features`, `no_metadata_features`, `label_not_a_feature`,
  `one_row_per_sample`, `feature_metadata_alignment` — all pass
  (`leakage_passed: true`).
* ML-ready: 40 candidate features → **13 selected** (0 missing, 12 constant,
  15 correlated).
* Fixed defect this stage: `DEPENDS_ON` edges carry no `distance` property, so
  `dependency_distance_mean/max` were 100% null; the extractor now derives the
  head–child distance from token positions.
* `graph_embeddings.npy` intentionally **not produced** (no validated embedding
  method).

## 10. Offline fingerprinting, GLTR, watermark and fusion — DONE

Read-only extension of the pilot; feature extraction only (no training, no
formal split, no test-set access). New versioned outputs under
`data/processed/fingerprinting/`; existing graph artifacts are never
overwritten.

* **Module A — stylometric fingerprinting** (`fingerprinting/stylometry.py` +
  `vocabulary_/sentence_/punctuation_/pos_/syntactic_/readability_features.py`):
  **140** features (vocab 26, sentence 19, punctuation 28, POS 32, syntactic 30,
  readability 5). Reuses local spaCy (`en_core_web_sm`); POS/syntactic/
  readability-syllable features degrade to `NaN` without it.
* **Module B — writeprints-style** (`fingerprinting/writeprints.py`): **301**
  bounded sparse features (char/word/POS n-grams, punctuation patterns,
  function words, summary), written as CSV + CSR `.npz`. Frequency
  normalization; configurable vocabulary limits.
* **Module C — GLTR-style** (`gltr/`): full local causal-LM pipeline
  (`local_model.py`, `token_probability.py`, `token_rank.py`,
  `entropy_analysis.py`, `document_features.py`, `pipeline.py`), **19**
  document features. **Unavailable in this environment** (`transformers`/`torch`
  not installed; `allow_download: false`), so it emits an all-`NaN` placeholder
  and never downloads a model. `estimate-gltr` sizes a real run first.
* **Module D — watermark verification** (`watermark/statistics.py`,
  `unigram.py`, `topic.py`): Kirchenbauer-style unigram green-list detector
  (gamma 0.25, z- and p-value thresholds) + topic-conditioned interface
  (unsupported without a known protocol). **12** evidence columns; key read from
  `MVGR_WATERMARK_KEY`, never stored or printed. With no key, all 500 rows are
  `unsupported`; absence of detection is never reported as human authorship.
* **Module E — fusion** (`fusion/multiview_feature_fusion.py`): joins
  lexical/syntactic graph branches with stylometric/writeprints/GLTR and keeps
  watermark as auxiliary evidence. Identifiers/metadata/labels stay in a
  separate table. Pilot: **631 candidates → 408 selected** (30 missing, 26
  constant, 167 correlated).
* **Config/versions**: `configs/fingerprint_config.yaml`; new version constants
  in `shared/contracts/versions.py` (`fingerprint.v1`, `stylometric.v1`,
  `writeprints.v1`, `gltr.v1`, `watermark.v1`, `fusion.v1`). `pyproject.toml`
  gains the optional `gltr` extra and `gltr*`/`fusion*` packages.
* **Full pilot run**: 500/500 samples, ~100.8 s, peak 141.8 MB.
* **Validation** (`scripts/validate_fingerprint_features.py`): **PASS 41/41**
  (schema, coverage, duplicates, sample alignment, leakage, value ranges,
  missing-value thresholds, watermark status validity, GLTR availability
  consistency).
* Reports in `outputs/fingerprint_audit/`; design write-up in
  [`reports/final/11_fingerprint_gltr_watermark.md`](reports/final/11_fingerprint_gltr_watermark.md).

## 11. Backend — DONE

`backend/` FastAPI app: `GET /health`, `GET /v1/contracts/versions`,
`POST /v1/analyze`. Report assembly always returns a contract-valid report and
degrades honestly (`classification.status = "unavailable"` when no model bundle
is present). Optional `api` extra; routes are covered by a test that skips when
FastAPI is absent.

## 12. Tests — DONE (132 pass, 2 skipped)

| File | Tests | Covers |
|---|---:|---|
| `tests/test_dataset.py` | 14 | provenance, lineage, gates, split assignment, manifest schema |
| `tests/test_deterministic.py` | 17 | `features.v1`, families, scoring, guards |
| `tests/test_graph.py` | 11 | builders, graph contract, build identity, export |
| `tests/test_graph_features.py` | 22 | feature math, selection gates, leakage, verifier detection, live-DB |
| `tests/test_fingerprinting.py` | 17 | stylometric families, empty/short/long text, POS/syntactic degradation, schema |
| `tests/test_writeprints.py` | 10 | n-gram vocab bounds, normalization, sparse/dense consistency |
| `tests/test_gltr.py` | 14 | tokenizer alignment, causal shift, missing-model handling (skips without a local LM) |
| `tests/test_watermark.py` | 11 | green-list statistics, key handling, unsupported schemes, no-false-negative rule |
| `tests/test_feature_fusion.py` | 12 | branch join, identifier/metadata separation, selection, schema |
| `tests/test_integration.py` | 6 | end-to-end graph → features → score → evaluate |
| **Total** | **134** | **132 passed, 2 skipped** |

The two skips are the GLTR local-model test (no causal LM installed) and the
FastAPI integration test (optional `api` extra absent).

## 13. Documentation — DONE

`reports/final/` is the authoritative set:

| # | Document |
|---|---|
| 00 | System architecture |
| 01 | Offline and online pipelines |
| 02 | Graph storage and views (Neo4j) |
| 03 | Analysis methods (deterministic, GNN, fingerprint, watermark) |
| 04 | Backend, contracts and API |
| 05 | Evaluation and experiments |
| 06 | Roadmap, testing and risks |
| 07 | Responsibility matrix |
| 08 | Dataset audit findings |
| 09 | Graph feature pipeline (Phases A–F) |
| 10 | Frontend (design and contract) |
| 11 | Offline fingerprinting, GLTR and watermark verification |

## 14. Frontend — NOT IMPLEMENTED (design ready)

* `frontend/README.md` is a placeholder; no UI code exists.
* A complete design/interface spec is written in
  [`reports/final/10_frontend.md`](reports/final/10_frontend.md) (screens,
  honesty rendering rules, API integration, testing, milestones).

---

## Blocked / not done

| Item | State | Reason |
|---|---|---|
| Approved split manifest | **Blocked** | Human decision required. 3 candidates exist; none approved. |
| Formal train/validation/test split | **Not generated** | Gated on the approved manifest. |
| Model training (deterministic/GBM) | **Not started** | Requires an approved split. |
| GNN training | **Not started** | Requires approved split **and** PyTorch + PyG (not installed). |
| Test-set evaluation | **Not run** | Test lock held until the manifest is approved. |
| GLTR execution | **Not run** | No local causal LM; `transformers`/`torch` not installed. Module reports unavailable and never auto-downloads. |
| Fingerprint/GLTR/watermark features as model inputs | **Not started** | Extraction only; a real experiment is gated on the approved split (and must fit writeprints vocab on train). |
| Frontend implementation | **Not started** | Roadmap Phase 8. |
| `graph_embeddings.npy` | **Skipped** | No validated embedding method implemented. |

### Candidate manifests (none approved)

| Manifest | Scope | train/val/test | Gate |
|---|---|---|---|
| `man_f3aea517be133c1421c1` | pooled (regenerate) | 90,099 / 19,306 / 19,304 | fail `lineage_resolution` |
| `man_a4b3d451ff3129f4f95a` | pooled (existing) | 108,211 / 11,861 / 8,637 | fail `lineage_resolution` |
| `man_20ddf272c292efbc8c8c` | source-scoped (PADBen) | 56,457 / 12,098 / 12,096 | **pass** (awaiting approval) |

To unblock training, approve the PADBen source-scoped manifest (or record a
`WAIVER:` for the pooled scope):

```powershell
python mvgr.py approve-manifest --manifest data/manifests/candidate/man_20ddf272c292efbc8c8c/manifest.json `
  --approver "<name>" --statement "PADBen source-scoped split approved; no cross-source claims."
```

---

## How to verify everything

```powershell
# tests
.\.venv\Scripts\python.exe -m pytest

# dataset audit + lineage
python mvgr.py audit
python mvgr.py lineage

# live Neo4j verification (read-only)
.\.venv\Scripts\python.exe scripts\verify_neo4j_graphs.py

# graph feature extraction + validation + ML prep
.\.venv\Scripts\python.exe scripts\extract_graph_features.py
.\.venv\Scripts\python.exe scripts\validate_graph_features.py
.\.venv\Scripts\python.exe scripts\prepare_ml_dataset.py

# offline fingerprinting / writeprints / GLTR / watermark / fusion (pilot)
.\.venv\Scripts\python.exe scripts\fingerprint_pipeline.py all
.\.venv\Scripts\python.exe scripts\validate_fingerprint_features.py

# contract smoke test
python mvgr.py contracts --validate
```

## Guiding rules (still in force)

1. The dataset is the source of truth; nothing invents records, labels or
   paraphrase links.
2. Labels, source, task, family, condition and split are never graph node
   features or model inputs.
3. No approved manifest, no training and no test-set access.
4. A missing watermark or missing model is never evidence of human authorship.
5. Similarity is evidence of textual overlap, never of common authorship.
