# 10. Frontend

This document is the single reference for the MVGR-AI web frontend. The frontend
is **not implemented yet** — `frontend/` is a reserved placeholder — so this is a
design and interface specification, not a description of running code. It exists
so that whoever builds the UI renders the backend's evidence report faithfully
instead of inventing scores or hiding uncertainty.

## Status at a glance

| Item | State |
|---|---|
| Implementation | **Not started** (placeholder only) |
| Owner | Both contributors (see [responsibility matrix](07_responsibility_matrix.md)) |
| Contract it consumes | [`shared/contracts/analysis_report_schema.json`](../../shared/contracts/analysis_report_schema.json) |
| Backend it talks to | `backend/` — `POST /v1/analyze` |
| Roadmap phase | Phase 8, gated behind an approved split manifest and API hardening ([roadmap](06_roadmap_testing_risks.md)) |
| What exists today | `frontend/README.md` (placeholder note) |

## 1. Purpose and audience

The frontend is a **thin, evidence-first client** for the online analysis API. A
user submits text; the client displays the returned `analysis_report` verbatim.

Primary audiences:

* **Reviewers / researchers** who need to see every piece of evidence and every
  caveat behind a result.
* **Operators** who need to confirm which model and contract versions produced a
  report.

The frontend is explicitly **not** an authorship oracle. It does not decide, and
it does not hide the fact that a decision may be impossible.

## 2. Design principles

These mirror the project-wide rules in [system architecture](00_architecture.md):

1. **Render the contract, don't reinterpret it.** The report shape is fixed by
   `analysis_report_schema.json`. The UI is a view over that JSON, not a second
   source of truth.
2. **No client-side scoring.** All classification, graph analysis, fingerprinting
   and watermark checks happen server-side so the model and contract versions stay
   auditable. The client never re-scores, re-ranks, or thresholds.
3. **Honesty is a UI requirement.** A missing model, index or watermark is shown
   as *unavailable* / *unsupported*, never converted into a positive claim. This
   is the same fail-closed rule the API enforces.
4. **Evidence before verdict.** The evidence blocks and the limitations list are
   first-class, not buried behind a summary.
5. **Every block carries a status.** No evidence block is ever shown without its
   explicit `status` and, where present, its `reason`.
6. **Version pinning is visible.** The report's `versions{}` block is surfaced so a
   result can always be tied to a model and schema version.

## 3. Where it sits

```text
user text ──► frontend (React) ──► POST /v1/analyze ──► backend report assembly
                     ▲                                         │
                     └────────── analysis_report (JSON) ◄──────┘
```

The frontend is a leaf consumer: it calls the API and renders the response. It
has no access to the dataset, the graph store, or the model artifacts.

## 4. The contract it renders

Every screen is driven by one response object, validated server-side by
`validate_analysis_report`. The top-level shape:

```jsonc
{
  "report_id": "rep_...",
  "report_schema_version": "1.0.0",
  "created_at": "2026-11-13T...Z",
  "input":         { "characters": 123, "words": 21, "text_sha256": "..." },
  "classification":{ "status": "available|unavailable", "label": "ai|human|uncertain",
                     "p_ai": 0.83, "confidence": 0.7, "calibrated": false,
                     "model_version": "...", "reason": "..." },
  "graph_analysis":{ "status": "available|partial|unavailable",
                     "views": { "lexical": { "n_nodes": 42, "...": 0 } }, "reason": "..." },
  "fingerprint":   { "status": "available|no_reference_index|unavailable",
                     "exact_sha256": "...", "normalized_sha256": "...",
                     "candidates": [ { "reference_id": "...", "method": "exact",
                                       "score": 0.98 } ], "reason": "..." },
  "watermark":     { "status": "detected|not_detected|unsupported|inconclusive",
                     "provider": "...", "confidence": 0.0, "details": "...",
                     "assumptions": [ "..." ] },
  "evidence":      [ { "source": "classification|graph|fingerprint|watermark",
                       "status": "supports_ai|supports_human|neutral|unsupported",
                       "statement": "...", "strength": 0.6 } ],
  "limitations":   [ "..." ],
  "versions":      { "report_schema_version": "1.0.0", "graph_spec_version": "1.0.0",
                     "model_version": "...", "feature_set_version": "features.v1" }
}
```

Notes that matter for the UI:

* `classification.status = "unavailable"` means there is **no score** — the
  `label`, `p_ai` and `confidence` fields may be absent.
* `fingerprint.candidates[].score` is a **similarity to a reference**, never an
  authorship probability.
* `watermark.status = "unsupported"` means the environment cannot check — not
  that a watermark is absent.
* `evidence[]` and `limitations[]` are always present (the schema requires them).

## 5. Rendering rules (the honesty table)

The client must translate statuses into honest language. These mappings are
normative — a pull request that violates them should be rejected.

| Report value | UI must say | UI must **not** say |
|---|---|---|
| `classification.status = "unavailable"` | "No score available" + `reason` | "Human" / "AI" |
| `classification.label = "uncertain"` | "Inconclusive" | "Human" |
| `watermark.status = "unsupported"` | "No evidence either way (unsupported)" | "Human" / "No watermark" |
| `watermark.status = "not_detected"` | "No watermark detected" + "absence is not proof of human authorship" | "Human" |
| `fingerprint.status = "no_reference_index"` | "No reference index available" | "No match → human" |
| `fingerprint.candidates[].score` | "similarity to reference `<id>`" | "authorship" / "plagiarism" |
| `graph_analysis.status = "partial"` | show the views present + a partial badge | "Complete" |
| any evidence `status = "unsupported"` | "Unsupported" chip | omit the block |
| `limitations[]` | always visible (see §7) | collapsed by default |

Rule of thumb: **absence of evidence is displayed as absence of evidence.**

## 6. Planned screens

1. **Analyze** — the entry point.
   * Textarea (paste) plus a file drop (`.txt`); character counter against the
     server cap.
   * Options: `include_graph`, `include_fingerprint`.
   * Submit → loading state → report. Errors (413 / 422 / network) are shown
     inline, not as a generic toast.
2. **Report summary** — a compact, honest header: classification status, the
   overall evidence tally (`supports_ai` / `supports_human` / `neutral` /
   `unsupported` counts), and a prominent **Limitations** link.
3. **Evidence** — one card per `evidence[]` entry, plus one per analysis block:
   * *Classification* — status, label, `p_ai`, `calibrated`, `model_version`.
   * *Graph analysis* — the `views` numbers as a small metrics table; an
     interactive node/edge canvas is a later milestone.
   * *Fingerprint* — `exact_sha256`, `normalized_sha256`, and a candidate table
     (`reference_id`, `method`, `score`) labelled "similarity".
   * *Watermark* — status, provider, confidence, details, assumptions.
4. **Limitations** — the `limitations[]` list, always visible on the report page
   (not a hidden tooltip). At minimum it includes "Absence of a watermark is not
   evidence of human authorship."
5. **Versions / provenance** — a small footer or panel showing `report_id`,
   `created_at`, and every entry in `versions{}`.
6. **Contracts** — an optional diagnostic view over `GET /v1/contracts/versions`
   so a client build can show the versions it is pinned to.

## 7. Limitations and honesty UI

The `limitations[]` array is a required part of the report and must be shown on
the same screen as the result, above the fold. It is never collapsed by default
and never truncated without a visible "show all" affordance.

The UI must also carry a persistent, short disclaimer, e.g.:

> Results are statistical evidence, not proof of authorship. A missing watermark
> or a missing model is not evidence that a text is human-written.

## 8. API integration

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + model-bundle availability; drives a "scoring available?" banner |
| `GET` | `/v1/contracts/versions` | Contract versions this backend serves |
| `POST` | `/v1/analyze` | Submit text, receive an `analysis_report` |

Request body (`AnalyzeRequest`):

```json
{ "text": "...", "include_graph": true, "include_fingerprint": true }
```

Error handling:

| Condition | HTTP | UI behaviour |
|---|---|---|
| Blank text | `422` | inline validation message; no request sent if the client checks first |
| Text over the cap | `413` | show the cap (`MVGR_API_MAX_TEXT_CHARS`, default 50,000) and the current length |
| Backend unreachable | — | "Service unavailable" state; never a fabricated result |

CORS is configured server-side from `MVGR_API_CORS_ORIGINS` (comma-separated
origins; CORS is off when unset). The dev setup must add the frontend origin to
that variable.

Environment variables that affect the client contract:

| Variable | Effect |
|---|---|
| `MVGR_API_CORS_ORIGINS` | Which origins may call the API |
| `MVGR_API_MAX_TEXT_CHARS` | Client-side character cap to mirror |

The client should also read the API base URL from a single config value
(suggested `VITE_API_BASE_URL`) rather than hard-coding it.

## 9. State and data flow

```text
idle ──submit──► loading ──200──► report ──new submit──► loading
                    │
                    ├─ 413/422 ─► validation error (stay on Analyze)
                    └─ network ─► service-unavailable error (retry)
```

* The report is immutable once received; the UI never mutates it.
* Optionally keep a session-only history of `report_id`s in memory. No report is
  persisted client-side by default (it contains the user's submitted text hash).
* No caching of scoring: every submit is a fresh server call.

## 10. Suggested stack and layout

The stack is a suggestion, not a contract:

* **React + TypeScript + Vite**.
* A small **typed API client** whose types are generated from
  `analysis_report_schema.json` (or hand-mirrored and kept in sync by a test).
* No client-side model calls or re-scoring.

Proposed layout (to be created when implementation starts):

```text
frontend/
├── README.md                 # placeholder (this phase)
├── package.json
├── vite.config.ts
└── src/
    ├── main.tsx
    ├── api/
    │   ├── client.ts         # fetch wrapper, base URL, error mapping
    │   └── types.ts          # generated from analysis_report_schema.json
    ├── components/
    │   ├── AnalyzeForm.tsx
    │   ├── StatusChip.tsx    # status → honest label (the §5 table)
    │   ├── ClassificationCard.tsx
    │   ├── GraphCard.tsx
    │   ├── FingerprintCard.tsx
    │   ├── WatermarkCard.tsx
    │   ├── EvidenceList.tsx
    │   ├── Limitations.tsx   # always visible
    │   └── VersionsFooter.tsx
    └── pages/
        ├── AnalyzePage.tsx
        └── ReportPage.tsx
```

A single `StatusChip` (or equivalent) should own the §5 mapping so the honesty
rules live in exactly one place.

## 11. Accessibility and UX requirements

* Text contrast and focus states meet WCAG 2.1 AA; status is never conveyed by
  colour alone (always a text label too).
* The results page is keyboard navigable; the evidence list has a logical heading
  order.
* Long hashes and IDs are shown in a monospace font with copy-to-clipboard.
* Loading, empty, partial and error states are all designed — not just the happy
  path.
* The character counter warns before the server cap, not only after a 413.

## 12. Testing strategy

The frontend must not depend on the full corpus or any heavy backend dependency.
Planned tests:

| Test | What it asserts |
|---|---|
| Contract fixture rendering | A stored `analysis_report` fixture renders every block without crashing |
| Honesty rules | `unavailable`/`unsupported`/`no_reference_index` render as "no score / no evidence", never as "human" |
| Limitations always visible | The limitations list is present on the report page |
| Error mapping | 413 and 422 produce the correct inline messages |
| Type drift | Generated types still match `analysis_report_schema.json` (fails if the contract changes) |

Fixtures should be produced by the backend (`backend.services.analysis`) so the
frontend is tested against real, contract-valid reports rather than hand-written
ones.

## 13. Milestones

| # | Milestone | Exit condition |
|---|---|---|
| M0 | Placeholder | `frontend/README.md` only (current) |
| M1 | Analyze + report skeleton | Submit text, render classification + limitations + versions |
| M2 | Full evidence cards | Graph, fingerprint, watermark cards with §5 status mapping |
| M3 | Hardening | Error states, accessibility pass, contract-fixture tests |
| M4 | Graph canvas | Interactive node/edge view over `graph_analysis.views` (later) |

M1–M3 depend on the API being hardened (Phase 7 in the [roadmap](06_roadmap_testing_risks.md)),
which in turn follows an approved split manifest.

## 14. Non-goals

* No client-side classification, graph analysis, fingerprinting or watermarking.
* No training, no model download, no dataset access from the browser.
* No user accounts, persistence or sharing in the first implementation.
* No "verdict" summary that collapses the evidence into a single human/AI
  pronouncement.

## 15. Open questions

* **History:** is a session-only report history in scope, or strictly one report
  at a time?
* **Graph canvas:** is the interactive node/edge view required for the first
  release, or is the metrics table enough?
* **Hosting:** same-origin reverse proxy vs. a separate origin with CORS — this
  determines the `MVGR_API_CORS_ORIGINS` setup.
* **Type generation:** generate from the JSON Schema in CI, or hand-mirror with a
  drift test?

## 16. References

* [System architecture](00_architecture.md) — where the frontend sits.
* [Backend, contracts and API](04_backend_api_contracts.md) — the endpoints and
  honesty rules it renders.
* [Roadmap, testing and risks](06_roadmap_testing_risks.md) — Phase 8.
* [Responsibility matrix](07_responsibility_matrix.md) — ownership.
* [`shared/contracts/analysis_report_schema.json`](../../shared/contracts/analysis_report_schema.json)
  — the report contract.
* `frontend/README.md` — the placeholder note in the code tree.
