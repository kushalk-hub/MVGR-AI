# 2. Offline and online pipelines

## Offline pipeline

The offline pipeline runs once per dataset version and per approved split. It is
the only place where training and evaluation happen.

```mermaid
flowchart LR
    A[datasets/processed] -->|verify SHA-256| B[data/processed]
    B --> C[dataset_audit.inventory]
    C --> D[dataset_audit.lineage]
    D --> E[data/derived/family_index.jsonl]
    E --> F[dataset_audit.split_manifest]
    F --> G{gate status}
    G -- pass --> H[approve-manifest]
    G -- fail + WAIVER --> H
    H --> I[data/manifests/split_manifest.approved.json]

    I --> J[graph_engine builders]
    J --> K[graph exports<br/>gb_.../lexical.jsonl + manifest]
    K --> L[neo4j loader]
    K --> M[deterministic.feature_extractor]
    K --> N[gnn dataset/train]
    B --> O[baseline tfidf_logistic]
    M --> P[frozen scoring rule]
    P --> Q[evaluation]
    N --> Q
    O --> Q
    Q --> R[reports/evaluation + experiments]
```

### Stage order and gates

1. **Provenance** — the contract copy is verified against the upstream digest.
2. **Audit** — inventory, label mapping, lineage, leakage, and confounding.
3. **Family index** — one `family_id` per sample; PADBen is pair-level,
   BiScope is exact-normalized-text only.
4. **Candidate manifest** — family-disjoint assignment with checksummed
   `.ids.txt` sidecars.
5. **Approval** — a failing gate needs a statement beginning with `WAIVER:`.
6. **Graph build** — only from an approved manifest; the export stage rejects
   frames that contain target metadata.
7. **Features** — `features.v1` extraction from the graph export.
8. **Models** — frozen deterministic rule, TF-IDF baseline, and (optional) GNN.
9. **Evaluation** — labels are joined only here, stratified by source/task/
   condition, with bootstrap uncertainty.

## Online pipeline

The online pipeline answers a single text with an evidence report. It reuses the
offline builder and feature registry so a score means the same thing in both
pipelines.

```mermaid
sequenceDiagram
    participant U as Client
    participant API as backend /v1/analyze
    participant G as graph_engine
    participant D as deterministic
    participant F as fingerprinting
    participant W as watermark
    U->>API: POST {text}
    API->>G: build lexical graph
    G-->>API: graph record
    API->>D: extract features.v1
    D-->>API: feature vector
    API->>API: apply frozen rule (if a bundle exists)
    API->>F: reference index lookup (if configured)
    API->>W: watermark verification
    W-->>API: detected / not_detected / unsupported / inconclusive
    API-->>U: analysis_report (contract-validated)
```

### Degradation policy

| Missing piece | Reported as | Never reported as |
|---|---|---|
| No scoring rule | `classification.status = "unavailable"` | "human" |
| No reference index | `fingerprint.status = "no_reference_index"` | "no reuse found" |
| No watermark provider | `watermark.status = "unsupported"` | "human" |
| Graph dependency error | `graph_analysis.status = "unavailable"` | a fabricated score |

## Data flow: what may cross a boundary

```mermaid
flowchart LR
    T[text] --> GR[graph record]
    GR --> FE[feature vector]
    FE --> SC[score]
    LB[label + source + task + condition + split] -.->|joined only here| EV[evaluation]
    SC --> EV
    EV --> MT[metrics]
```

The dashed edge is the only place target metadata may meet a prediction. Graph
records and feature tables are checked to contain no target-metadata columns.
