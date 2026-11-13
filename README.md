# MVGR-AI

Multi-view graph analysis of human, AI, paraphrased, and deeply paraphrased
text. MVGR-AI turns each document into a graph, extracts deterministic graph
features, and combines them with fingerprint and watermark evidence into a
single, honest evidence report. It never treats the absence of a watermark or
the absence of a model as proof of human authorship.

The dataset is the source of truth. Nothing here invents records, labels, or
paraphrase links, and no artifact may leak labels, source, task, family,
condition, or split into graph structure or model features.

---

## Repository layout

```text
MVGR-AI/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── pyproject.toml                # additive packaging/CLI entry point
├── mvgr.py                       # additive CLI: audit, split, build, serve, ...
│
├── configs/
│   ├── dataset_config.yaml
│   ├── graph_config.yaml
│   ├── experiment_config.yaml
│   └── evaluation_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/                # byte-identical contract copy of the source
│   ├── derived/                  # family index, caches
│   └── manifests/                # candidate + approved split manifests
│
├── dataset_audit/
│   ├── inventory.py
│   ├── label_mapping.py
│   ├── lineage.py
│   └── leakage_audit.py
│
├── deterministic/
│   ├── feature_extractor.py
│   ├── clustering.py
│   ├── degree_distribution.py
│   ├── centrality.py
│   ├── graph_density.py
│   ├── shortest_path.py
│   ├── community.py
│   ├── connectivity.py
│   └── scoring.py
│
├── baseline/
│   └── tfidf_logistic.py
│
├── fingerprinting/
│   ├── sha256.py
│   ├── ngram.py
│   └── minhash.py
│
├── watermark/
│   └── verifier.py
│
├── graph_engine/
│   ├── graph_builder/
│   │   ├── lexical_graph.py
│   │   └── syntactic_graph.py
│   ├── neo4j/
│   │   ├── connection.py
│   │   ├── schema.py
│   │   ├── loader.py
│   │   └── queries.py
│   ├── graph_export/
│   │   └── export.py
│   └── gnn/
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       └── predict.py
│
├── shared/
│   ├── contracts/
│   │   ├── graph_schema.json
│   │   └── prediction_schema.json
│   ├── preprocessing/
│   └── validation/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   └── schemas/
│
├── frontend/                     # placeholder this phase (no UI code yet)
│   └── README.md
│
├── evaluation/
│   ├── metrics.py
│   ├── comparison.py
│   └── error_analysis.py
│
├── experiments/
│   ├── configs/
│   ├── predictions/
│   ├── checkpoints/
│   └── logs/
│
├── reports/
│   ├── dataset_audit/
│   ├── graph_audit/
│   ├── evaluation/
│   └── final/
│
└── tests/
    ├── test_dataset.py
    ├── test_graph.py
    ├── test_deterministic.py
    └── test_integration.py
```

Two artifacts are intentionally outside this tree: `datasets/` (the immutable
upstream download) and `dataset_preparation.py` + `scripts/` (the legacy
generator that produced it). The audit verifies that `data/processed/` is a
byte-identical copy of `datasets/processed/` and fails closed if it is not.

---

## Two pipelines

```mermaid
flowchart LR
    subgraph Offline["Offline pipeline"]
        A[Immutable dataset<br/>datasets/processed] --> B[dataset_audit]
        B --> C[Family lineage<br/>data/derived/family_index.jsonl]
        C --> D[Candidate split manifest]
        D --> E{Approved manifest?}
        E -- yes --> F[graph_engine builders]
        F --> G[Graph exports<br/>JSONL + manifests]
        G --> H[Neo4j load]
        G --> I[deterministic features<br/>features.v1]
        I --> J[Frozen scoring rule]
        G --> K[GNN]
        G --> L[TF-IDF baseline]
        J --> M[Shared evaluation]
        K --> M
        L --> M
        M --> N[reports/evaluation]
    end

    subgraph Online["Online pipeline"]
        O[Text] --> P[Lexical graph]
        P --> Q[features.v1]
        O --> R[Fingerprint lookup]
        O --> S[Watermark verification]
        Q --> T[Evidence aggregation]
        R --> T
        S --> T
        T --> U[analysis_report]
        U --> V[React frontend - future]
    end
```

The offline pipeline is the only path allowed to train or evaluate. The online
pipeline reuses the *same* graph builder, tokenizer, and feature registry as
offline so that a score means the same thing in both.

---

## Quickstart

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[graph,baseline,dev]"      # networkx + scikit-learn + pytest
```

Run the audit (fails closed until the split manifest is approved):

```powershell
python mvgr.py audit
python mvgr.py lineage
```

Build candidate split manifests:

```powershell
python mvgr.py split-manifest --mode regenerate --scope pooled
python mvgr.py split-manifest --mode existing   --scope pooled
python mvgr.py split-manifest --mode regenerate --scope source_scoped --sources PADBen
```

Build graphs and features, then score:

```powershell
python mvgr.py graph-build --view lexical --split train
python mvgr.py features
python mvgr.py deterministic
```

Serve the online API (requires the `api` extra):

```powershell
python mvgr.py serve
```

Run the tests:

```powershell
python -m pytest
```

---

## Dataset sources

* BiScope — <https://huggingface.co/datasets/HanxiGuo/BiScope_Data>
* PADBen — <https://huggingface.co/datasets/JonathanZha/PADBen>

Review each dataset card and license before use or redistribution. Do not
redistribute the merged dataset unless both source licenses permit it.
