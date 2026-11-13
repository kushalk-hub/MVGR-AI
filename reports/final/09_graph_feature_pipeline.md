# 9. Graph feature extraction and ML-ready preparation (Phases A–F)

This document describes the read-only verification (Phase A), graph feature
extraction (Phases B/C), leakage policy (Phase D) and ML-ready dataset
preparation (Phase E) that run against the **already-loaded pilot build** in
Neo4j Aura. Nothing in this pipeline re-uploads, mutates or deletes graph
records.

## Live schema (discovered, never assumed)

The pilot was loaded with a purpose-built loader, so the live schema differs
from the contract loader in `graph_engine/neo4j/loader.py`:

```
(:Sample)-[:HAS_GRAPH]->(:TextGraph)-[:HAS_TOKEN]->(:TokenOccurrence)
(:TokenOccurrence)-[:CO_OCCURS]->(:TokenOccurrence)     # lexical_cooccurrence
(:TokenOccurrence)-[:DEPENDS_ON]->(:TokenOccurrence)    # syntactic_dependency
```

Every label, relationship type, property key and categorical vocabulary is
discovered from the database (`db.labels()`, `db.relationshipTypes()`,
`keys(n)`, distinct property values) before use.

## Phase A — verification (`scripts/verify_neo4j_graphs.py`)

Read-only checks comparing live statistics against the pilot manifest, plus
internal-consistency checks:

* node/relationship totals equal the sum of the per-label / per-type counts;
* 500 samples, 1,000 graphs (500 per view), 102,822 token nodes;
* every sample has exactly two views; no duplicate view pairs;
* no orphan tokens, no graphs without tokens, no dangling graphs;
* no structural edge crosses a `graph_id` boundary;
* stored `token_count`/`edge_count` match actual neighbourhood sizes;
* only the three expected labels and four expected relationship types exist.

Reports: `outputs/graph_audit/neo4j_verification_report.{json,md}`.

## Phases B/C — feature extraction (`scripts/extract_graph_features.py`)

Samples are processed in batches (25 per chunk). Each chunk issues bounded,
batched queries — never one query per graph — and produces:

* `data/processed/graph_features.csv` — one row per graph view,
* `data/processed/sample_multiview_features.csv` — one row per sample,
* `data/processed/node_features.csv` — one row per token node,
* `data/processed/feature_schema.json` — column roles and definitions.

### Transport note (important)

The Aura endpoint over the current network corrupts TLS records once a single
Bolt result exceeds roughly 100 KB (`SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC`).
`fetch_size` does not help, and the failure is deterministic on result size, not
connection state. The extractor therefore pages the token and edge queries with
`SKIP`/`LIMIT` (`token_page_rows`, `edge_page_rows`) and retries each query —
including result *consumption* — on a fresh connection.

### Feature families

* **structural** — size, density, degree statistics and histogram, clustering,
  transitivity, triangles, components, guarded shortest-path/diameter;
* **lexical** — type/token ratio, surface/lemma length, stopword and punctuation
  ratios, POS entropy, per-POS ratios, entity-type counts;
* **syntactic** — dependency-edge counts, relation diversity/entropy, in/out
  degree maxima, roots, head–child distance, tree depth, top-K relation counts;
* **multi-view** — per-view aggregates plus cross-view deltas and similarities
  (cosine and Jensen–Shannon over the degree histograms).

Undefined values follow the `features.v1` convention: emitted as empty cells
(NaN), never imputed.

### Dependency-distance correction

The pilot loader stores an edge `distance` only on `CO_OCCURS` edges; `DEPENDS_ON`
edges carry no distance property. The extractor originally read `r.distance` for
both edge types, so `dependency_distance_mean`/`dependency_distance_max` were
always null. The extractor now recovers the head–child distance from the token
`position` of the endpoints when the property is absent, so the two syntactic
features are defined for the syntactic view (and remain null for the lexical
view, as intended).

### Interpretation note

The master prompt refers to "original-view / paraphrased-view" multi-view
features. The pilot's two real graph views are `lexical_cooccurrence` and
`syntactic_dependency`; the multi-view features are therefore computed across
those two views.

## Phase D — leakage policy

Encoded in `configs/graph_feature_config.yaml` and enforced in code and tests:

* both views of a sample always share one split (split is a sample property);
* related samples (`group_id`) never cross splits where lineage exists;
* identifiers (`sample_id`, `graph_id`, `build_id`, `source_row_id`,
  `group_id`) are never predictive features;
* metadata/target columns (`label`, `source_dataset`, `source_task`,
  `text_type`, `graph_category`, `paraphrase_status`, `domain`, `ai_model`,
  `padben_split`, `split`) are preserved for joins/stratification but are never
  model inputs;
* the target label is carried only in the metadata table, never the feature
  matrix.

## Phase E — ML-ready preparation (`scripts/prepare_ml_dataset.py`)

Consumes the extractor's tables and writes:

* `data/processed/ml_ready_features.csv` — numeric feature matrix, one row per
  sample;
* `data/processed/ml_ready_metadata.csv` — identifiers, target and provenance;
* `data/processed/ml_dataset_report.json` — the full preparation report.

Steps: missing-value statistics → drop features above the configured missing
fraction → drop constant/near-constant features → drop one of each highly
correlated pair → encode the target into the metadata table → run the leakage
checks.

### Formal split gate

A formal train/validation/test split requires an **approved** manifest
(`data/manifests/split_manifest.approved.json`). No such manifest exists in the
repository (only candidates, none approved), so formal split generation is
**not performed**. The pilot's preserved `split` column is carried through for
pipeline validation only and is explicitly labelled non-formal.

## Phase F — configuration and validation

`configs/graph_feature_config.yaml` references Neo4j credentials by
environment-variable *name* only (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`,
`NEO4J_DATABASE`) and never stores a value.

`scripts/validate_graph_features.py` is a read-only validator for the four
feature artifacts. It checks sample coverage, schema conformance (column names
and order), duplicates, missing values, invalid values (infinities, negative
counts, out-of-range ratios, non-binary flags), two-view association,
per-sample/per-view aggregation consistency and leakage roles, and — when the
database is reachable — cross-checks the feature totals against the live Neo4j
counts. It writes `outputs/graph_audit/feature_extraction_report.{json,md}`.

Missing-value policy: the per-view `graph_features` table legitimately holds
nulls for view-specific features on the other view's rows, so the missing-value
gate applies to the ML-facing `sample_multiview_features` table (every
predictive column must stay within the configured missing fraction) plus a
"no predictive column is entirely null" check on every table.

## Run statistics and validation results

Full extraction run (`python scripts/extract_graph_features.py --no-resume`):

| Metric | Value |
|---|---:|
| Samples processed | 500 / 500 |
| Graph views processed | 1,000 / 1,000 (500 `lexical_cooccurrence`, 500 `syntactic_dependency`) |
| Token-node rows | 102,822 |
| Wall-clock time | 147.6 s |
| Failed batches | 0 |

Output artifacts:

| Artifact | Rows × columns | Size |
|---|---|---:|
| `data/processed/graph_features.csv` | 1,000 × 115 | 1.15 MB |
| `data/processed/sample_multiview_features.csv` | 500 × 53 | 0.29 MB |
| `data/processed/node_features.csv` | 102,822 × 22 | 29.1 MB |
| `data/processed/feature_schema.json` | — | 66.9 KB |
| `data/processed/ml_ready_features.csv` | 500 × 13 | 0.10 MB |
| `data/processed/ml_ready_metadata.csv` | 500 × 12 | 0.06 MB |
| `outputs/graph_audit/neo4j_verification_report.{json,md}` | — | 7.1 KB |
| `outputs/graph_audit/feature_extraction_report.{json,md}` | — | 16.6 KB |

Phase A verification: **24 pass, 0 warn, 0 fail**. Phase F validation
(`python scripts/validate_graph_features.py`): **50 pass, 0 warn, 0 fail**.

### Missing values

| Table | Missing cells | Share | Cause |
|---|---:|---:|---|
| `graph_features` | 13,952 / 115,000 | 12.1% | 27 syntactic-only columns are null on the 500 lexical rows (13,500 cells); source-missing metadata (`padben_split` 286, `paraphrase_status` 166) |
| `sample_multiview_features` | 226 / 26,500 | 0.85% | source-missing metadata (`padben_split` 143, `paraphrase_status` 83) |
| `node_features` | 96,410 / 2,262,084 | 4.3% | `ent_type` is empty for tokens with no named entity |

No predictive feature column is entirely null; the two views' features are
complete for their own rows. Undefined values are emitted as empty cells and
never imputed.

### Invalid values

None. Across all three tables there are no infinities, no negative counts, no
out-of-range ratios and no non-binary flags (checked by the validator).

### Live database cross-check

Read-only counts: samples 500, graphs 1,000, tokens 102,822, `HAS_GRAPH` 1,000,
`HAS_TOKEN` 102,822, `CO_OCCURS` 101,322, `DEPENDS_ON` 48,884. These match the
feature totals exactly.

### ML-ready preparation

40 candidate sample-level features → 13 selected (0 dropped for missing, 12
constant, 15 highly correlated). All leakage checks pass; the formal split is
**not** generated.

### Tests

`pytest`: **69 passed, 1 skipped**; `tests/test_graph_features.py`: **22 passed**
(21 pure unit tests plus one live-database test).

## What is intentionally not produced

* `data/processed/graph_embeddings.npy` — no embedding method is implemented and
  validated in this stage, so embeddings are skipped rather than fabricated.
* No model is trained and no test-set metric is computed; the 500-sample mixed
  pilot is for pipeline validation only and is not an approved training set.
