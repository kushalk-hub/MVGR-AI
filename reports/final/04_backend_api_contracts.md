# 9–11. Backend, contracts and API

This document covers the backend service (9), the shared contract layer (10), and
the HTTP API (11). The three are one design: the API's only job is to assemble a
contract-valid report from evidence that actually exists.

---

## 10. Contracts (`shared/contracts/`)

Contracts are the interface between every module and every owner. Each boundary
is a JSON Schema with a fail-closed validator: an artifact that does not validate
is an error, never a warning.

| Contract | Schema file | Validator | Used by |
|---|---|---|---|
| Graph export record | `graph_schema.json` | `validate_graph_record` | `graph_engine/graph_export`, `deterministic`, `gnn` |
| Prediction record | `prediction_schema.json` | `validate` | `deterministic`, `baseline`, `gnn` |
| Split manifest | `manifest_schema.json` | `validate_manifest` | `dataset_audit/split_manifest` |
| Family index | `family_index_schema.json` | `validate` | `dataset_audit/lineage` |
| Experiment record | `experiment_schema.json` | `validate` | `shared/tracking` |
| Analysis report | `analysis_report_schema.json` | `validate_analysis_report` | `backend` |

### Versions (`shared/contracts/versions.py`)

Single source of truth. A version is bumped in the same change that alters the
schema or semantics, and an existing version is never edited once artifacts with
that version exist.

```text
graph_schema 1.0.0   graph_spec 1.0.0   export jsonl+json-manifest
prediction_schema 1.0.0   manifest_schema 1.0.0   family_index_schema 1.0.0
experiment_schema 1.0.0   report_schema 1.0.0
split.v1   tokenizer.v1   features.v1
```

`mvgr.py contracts --validate` loads all six schemas and prints every version;
this is the cheap smoke test that the contract layer is intact.

### Content-addressed IDs (`shared/contracts/ids.py`)

Every artifact is identified by a SHA-256 over a canonical JSON serialization of
its defining payload — never a timestamp or random value.

| Prefix | Artifact |
|---|---|
| `cfg_` | resolved run configuration |
| `gb_` | graph build (dataset version + split + spec + view config + sorted sample ids) |
| `split_` | split assignment set |
| `man_` | split manifest |
| `exp_` | experiment |
| `rep_` | online analysis report |

Because IDs are content-addressed, a re-run with identical inputs produces an
identical ID, and sample ordering cannot change a build's identity.

### Fail-closed validation (`shared/contracts/validators.py`)

`validate(name, payload)` raises `ContractError` on any schema violation and is
the only path schemas are checked through. `validate_analysis_report` adds the
project-specific honesty rules on top of the JSON Schema (see below). `utc_now`,
`write_json`, and `write_jsonl` are the shared serialization helpers.

---

## 9. Backend (`backend/`)

### Structure

```text
backend/
├── main.py                 FastAPI app; mounts routers; optional CORS
├── routes/
│   ├── health.py           GET /health
│   ├── analyze.py          POST /v1/analyze
│   └── contracts.py        GET /v1/contracts/versions
├── schemas/
│   └── requests.py         Pydantic request/response models
└── services/
    ├── analysis.py         report assembly (the whole online pipeline)
    └── model_bundle.py     frozen-rule + reference-index loading
```

### Model bundle (`services/model_bundle.py`)

A bundle is a directory (default `experiments/checkpoints/current`, overridable
with `MVGR_API_MODEL_BUNDLE`) containing:

* `scoring_rule.json` — a frozen `deterministic.scoring` rule;
* `reference_index.npz` — optional, **train-only** fingerprint index.

`load_bundle` reports exactly what loaded and why anything else did not. It
never substitutes a fabricated score for a missing piece.

### Report assembly (`services/analysis.py`)

`analyze(text, ...)` runs the online pipeline and returns one contract-valid
report:

1. normalize and hash the text (`normalize_text`, `sha256_text`);
2. build the lexical graph and extract `features.v1`;
3. apply the frozen rule if a bundle is present, else mark classification
   `unavailable`;
4. look up the reference index if present, else mark fingerprint
   `no_reference_index`;
5. verify the watermark (default `unsupported`);
6. assemble `evidence[]`, `limitations[]`, and `versions{}`.

Every block carries an explicit `status` and a human-readable `reason`. A
missing model, index, or provider degrades the report; it never turns into a
positive authorship claim. The `BASE_LIMITATIONS` list always includes
"Absence of a watermark is not evidence of human authorship."

### Health

`health()` reports liveness plus bundle availability (`model_available`,
`loaded`, `reason`) so an operator can see at a glance whether the service is
serving scores or honest "unavailable" reports.

---

## 11. API

### `GET /health`

```json
{ "status": "ok", "model_bundle": "...", "model_available": false,
  "loaded": [], "reason": "No scoring rule at ..." }
```

### `GET /v1/contracts/versions`

Returns `{"contracts": { ... }}` — the full `CONTRACT_VERSIONS` map. This is how
a client pins itself to the contract versions it was built against.

### `POST /v1/analyze`

Request (`AnalyzeRequest`):

```json
{ "text": "...", "include_graph": true, "include_fingerprint": true }
```

* `text` — 1 to 50,000 characters (Pydantic), additionally capped by
  `MVGR_API_MAX_TEXT_CHARS` (default 50,000) → HTTP 413 if exceeded.
* Blank text → HTTP 422.

Response — an `analysis_report` validated by `validate_analysis_report`:

```json
{
  "report_id": "rep_...",
  "report_schema_version": "1.0.0",
  "created_at": "2026-11-13T...Z",
  "input": { "characters": 123, "words": 21, "text_sha256": "..." },
  "classification": { "status": "available|unavailable", "label": "ai|human",
                      "p_ai": 0.83, "calibrated": false, "model_version": "...",
                      "reason": "..." },
  "graph_analysis": { "status": "available", "views": { "lexical": { ... } } },
  "fingerprint": { "status": "available|no_reference_index",
                   "exact_sha256": "...", "candidates": [ ... ] },
  "watermark": { "status": "detected|not_detected|unsupported|inconclusive", ... },
  "evidence": [ { "source": "...", "status": "...", "statement": "...", ... } ],
  "limitations": [ "..." ],
  "versions": { "report_schema_version": "1.0.0",
                "graph_spec_version": "1.0.0", "model_version": "...",
                "feature_set_version": "features.v1" }
}
```

### Honesty rules enforced by the contract

`validate_analysis_report` rejects a report that:

* pairs `watermark.status = "not_detected"` with `classification.label = "human"`;
* omits the `limitations` list;
* emits an evidence block without an explicit status.

These are structural guarantees, not documentation promises: the API cannot
return a dishonest report even if a caller asks it to.

### Status vocabulary

| Block | Statuses |
|---|---|
| classification | `available`, `unavailable` |
| graph_analysis | `available`, `unavailable` |
| fingerprint | `available`, `no_reference_index`, `unavailable` |
| watermark | `detected`, `not_detected`, `unsupported`, `inconclusive` |

### Environment variables

| Variable | Purpose |
|---|---|
| `MVGR_API_MODEL_BUNDLE` | Bundle directory (default `experiments/checkpoints/current`) |
| `MVGR_API_MAX_TEXT_CHARS` | Request size cap (default 50,000) |
| `MVGR_API_CORS_ORIGINS` | Comma-separated allowed origins (CORS off if unset) |
| `MVGR_WATERMARK_PROVIDER` / `MVGR_WATERMARK_KEY` | Watermark provider config (unset → `unsupported`) |

### Serving

```powershell
pip install -e ".[api]"
python mvgr.py serve
```

FastAPI and uvicorn are an optional extra. In the reference environment they are
not installed, so the routes are covered by a test that skips on import and the
report assembly is exercised by calling `backend.services.analysis` directly.
