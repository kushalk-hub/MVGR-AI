# 13. Responsibility matrix

Ownership is documented here rather than encoded as top-level `person1/`/
`person2/` folders. This keeps shared code (`shared/`) in a neutral home and lets
both contributors work in one tree while still having one accountable owner per
module.

Legend: **P1** = owner A (audit, features, evidence, evaluation); **P2** =
owner B (graphs, storage, GNN). **Both** = shared ownership, change needs review
from each side.

---

## Module ownership

| Area | Module / file | Owner |
|---|---|---|
| Contracts | `shared/contracts/*` (schemas, validators, ids, versions) | Both |
| Preprocessing | `shared/preprocessing/text.py` | Both |
| Validation | `shared/validation/` | Both |
| Tracking | `shared/tracking.py` | Both |
| Paths/config | `shared/paths.py`, `configs/*` | Split (see below) |
| Audit | `dataset_audit/inventory.py` | P1 |
| Audit | `dataset_audit/lineage.py` | P1 |
| Audit | `dataset_audit/leakage_audit.py` | P1 |
| Audit | `dataset_audit/label_mapping.py` | P1 |
| Audit | `dataset_audit/split_manifest.py` | P1 |
| Features | `deterministic/feature_spec.py` | P1 |
| Features | `deterministic/feature_extractor.py` + family modules | P1 |
| Features | `deterministic/scoring.py`, `pipeline.py` | P1 |
| Baseline | `baseline/tfidf_logistic.py` | P1 |
| Fingerprints | `fingerprinting/*` | P1 |
| Watermark | `watermark/verifier.py` | P1 |
| Evaluation | `evaluation/*` | P1 |
| Graphs | `graph_engine/graph_builder/*` | P2 |
| Export | `graph_engine/graph_export/export.py` | P2 |
| Storage | `graph_engine/neo4j/*` | P2 |
| Models | `graph_engine/gnn/*` | P2 |
| Backend | `backend/*` | Both |
| Frontend | `frontend/` | Both (future) |
| Docs | `reports/final/*` | Both |
| Tests | `tests/*` | Both |

## Config ownership

| Config | Owner | Bump rule |
|---|---|---|
| `configs/dataset_config.yaml` | P1 | version bump + `docs/DECISIONS.md` entry |
| `configs/graph_config.yaml` | P2 | `graph_spec_version` bump on any semantic change |
| `configs/experiment_config.yaml` | Both | review from both contributors |
| `configs/evaluation_config.yaml` | P1 | `protocol_version` bump |

## Interface contracts (the handshake between owners)

| Interface | Producer | Consumer | Contract |
|---|---|---|---|
| Graph export JSONL | P2 | P1 (`deterministic`), P2 (`gnn`, Neo4j) | `graph_schema.json` |
| `features.v1` table | P1 | P1 (scoring), backend | feature registry + fixed column order |
| Frozen scoring rule | P1 | backend bundle | `deterministic.scoring` rule hash |
| Prediction JSONL | P1 & P2 | P1 (`evaluation`) | `prediction_schema.json` |
| Split manifest | P1 | P2 (graph build), everyone (training) | `manifest_schema.json` |
| Family index | P1 | P1 (split), P2 (awareness) | `family_index_schema.json` |
| Analysis report | Both | frontend | `analysis_report_schema.json` |

## Deliverable ownership

| # | Deliverable | Lead |
|---|---|---|
| 1 | System architecture | Both |
| 2 | Offline + online pipelines | Both |
| 3 | Neo4j graph storage | P2 |
| 4 | Graph views | P2 |
| 5 | Deterministic analysis | P1 |
| 6 | GNN | P2 |
| 7 | Fingerprinting | P1 |
| 8 | Watermark | P1 |
| 9 | Backend | Both |
| 10 | Contracts | Both |
| 11 | API | Both |
| 12 | Repository structure | Both |
| 13 | Responsibility matrix | Both |
| 14 | Experiment tracking | Both |
| 15 | Evaluation protocol | P1 |
| 16 | Roadmap | Both |
| 17 | Testing strategy | Both |
| 18 | Risks and mitigations | Both |
| 19 | Data flow | Both |
| 20 | Dataset audit findings | P1 |

## Collaboration rules

1. **Contracts first.** A change to a schema or version is agreed before the
   code that depends on it, and lands in the same change as the version bump.
2. **No cross-boundary edits.** P2 does not edit `deterministic/` or `evaluation/`
   without P1 review, and vice versa. Shared modules (`shared/`, `backend/`) need
   both.
3. **Target metadata is a hard wall.** Neither owner may add label/source/task/
   family/condition/split to a graph record, a feature table, or a prediction
   file.
4. **No approval, no training.** Neither owner trains or evaluates against a
   manifest that is not approved.
5. **Everything is an artifact.** Any result worth reporting is registered,
   versioned, and reproducible from its `config_id` and `split_id`.
