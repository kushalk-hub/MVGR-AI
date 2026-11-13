"""FastAPI application for the online MVGR-AI pipeline.

Endpoints
---------
GET  /health                     liveness plus model-bundle availability
GET  /v1/contracts/versions      every contract version served by this build
POST /v1/analyze                 contract-valid analysis report

The API is deliberately honest: without an approved model bundle it returns a
valid report with ``classification.status = "unavailable"`` instead of a guess.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import ROUTERS

app = FastAPI(
    title="MVGR-AI analysis API",
    version="0.1.0",
    description=(
        "Evidence-reporting API for AI-text analysis. Reports never claim that a "
        "missing watermark or a missing model proves human authorship."
    ),
)

_origins = [
    origin.strip()
    for origin in os.environ.get("MVGR_API_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

for router in ROUTERS:
    app.include_router(router)
