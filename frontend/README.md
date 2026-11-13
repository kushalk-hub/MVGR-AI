# MVGR-AI frontend (placeholder)

The web UI is **not implemented in this phase**. This directory reserves the
location for the future React application that consumes the online API.

> The full design and interface specification lives in
> [`reports/final/10_frontend.md`](../reports/final/10_frontend.md). This file is
> only the short in-tree placeholder note.

## What it will do

The frontend is a thin, evidence-first client. It submits text to
`POST /v1/analyze` and renders the returned `analysis_report` verbatim. It does
not re-score, re-rank, or hide evidence.

Planned views:

1. **Analyze** — paste or upload text, submit, and view the report.
2. **Evidence** — per-block cards for classification, graph analysis,
   fingerprint candidates, and watermark status, each with its explicit status.
3. **Graph view** — render the returned `graph_analysis.views` numbers; an
   interactive node/edge view is a later milestone.
4. **Limitations** — always visible, never collapsible by default.

## Contract

The frontend must render the exact `shared/contracts/analysis_report_schema.json`
shape and must honour these rules:

* `classification.status = "unavailable"` renders as "no score", never as
  "human".
* `watermark.status = "unsupported"` or `"not_detected"` renders as "no evidence
  either way", never as "human".
* `fingerprint` scores render as "similarity to a reference", never as
  "authorship".
* The `limitations` list is always shown.

## Suggested stack (when implementation starts)

* React + TypeScript + Vite
* A small typed API client generated from the report contract
* No client-side model calls; all scoring happens server-side so the model
  version and contract version stay auditable
