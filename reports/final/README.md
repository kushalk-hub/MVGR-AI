# MVGR-AI architecture and deliverable documentation

This folder is the authoritative, reviewable documentation set for MVGR-AI. It
is written against the real audit of the 128,709-row corpus, not against
assumptions about the datasets.

## Deliverable index

| # | Deliverable | Document |
|---|---|---|
| 1 | System architecture | [`00_architecture.md`](00_architecture.md) |
| 2 | Offline and online pipelines | [`01_pipelines.md`](01_pipelines.md) |
| 3 | Neo4j graph storage | [`02_graph_and_neo4j.md`](02_graph_and_neo4j.md) |
| 4 | Graph views | [`02_graph_and_neo4j.md`](02_graph_and_neo4j.md) |
| 5 | Deterministic analysis | [`03_analysis_methods.md`](03_analysis_methods.md) |
| 6 | GNN | [`03_analysis_methods.md`](03_analysis_methods.md) |
| 7 | Fingerprinting | [`03_analysis_methods.md`](03_analysis_methods.md) |
| 8 | Watermark | [`03_analysis_methods.md`](03_analysis_methods.md) |
| 9 | Backend | [`04_backend_api_contracts.md`](04_backend_api_contracts.md) |
| 10 | Contracts | [`04_backend_api_contracts.md`](04_backend_api_contracts.md) |
| 11 | API | [`04_backend_api_contracts.md`](04_backend_api_contracts.md) |
| 12 | Repository structure | [`00_architecture.md`](00_architecture.md) |
| 13 | Responsibility matrix | [`07_responsibility_matrix.md`](07_responsibility_matrix.md) |
| 14 | Experiment tracking | [`05_evaluation_experiments.md`](05_evaluation_experiments.md) |
| 15 | Evaluation protocol | [`05_evaluation_experiments.md`](05_evaluation_experiments.md) |
| 16 | Roadmap | [`06_roadmap_testing_risks.md`](06_roadmap_testing_risks.md) |
| 17 | Testing strategy | [`06_roadmap_testing_risks.md`](06_roadmap_testing_risks.md) |
| 18 | Risks and mitigations | [`06_roadmap_testing_risks.md`](06_roadmap_testing_risks.md) |
| 19 | Data flow | [`01_pipelines.md`](01_pipelines.md) |
| 20 | Dataset audit findings | [`08_dataset_audit_findings.md`](08_dataset_audit_findings.md) |
| 21 | Graph feature pipeline (Phases A–F) | [`09_graph_feature_pipeline.md`](09_graph_feature_pipeline.md) |
| 22 | Frontend (design and contract) | [`10_frontend.md`](10_frontend.md) |
| 23 | Offline fingerprinting, GLTR and watermark verification | [`11_fingerprint_gltr_watermark.md`](11_fingerprint_gltr_watermark.md) |

## Status

* Architecture, contracts, audit, deterministic features, baselines,
  fingerprinting, watermark, graph builders, export, Neo4j loader, backend, and
  tests are implemented and pass.
* No model has been trained and no test-set evaluation has been run. The split
  manifest is **not** approved; the pooled candidate fails the
  `lineage_resolution` gate by design and requires an explicit waiver.
* The GNN package requires the optional `gnn` extra (PyTorch + PyG), which is
  not installed in the reference environment.
* The frontend is a documented placeholder; no UI code is written this phase.
* Offline fingerprinting, writeprints, GLTR, watermark verification and feature
  fusion are implemented and tested (see
  [`11_fingerprint_gltr_watermark.md`](11_fingerprint_gltr_watermark.md)). The
  full 500-sample pilot extraction ran and its validator passes 41/41. GLTR is
  implemented but reports **unavailable** because no local causal LM is
  installed; it never downloads a model automatically. No model was trained and
  no formal split was generated.
