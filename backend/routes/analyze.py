"""Analysis endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from backend.schemas import AnalyzeRequest
from backend.services import analyze

router = APIRouter(prefix="/v1", tags=["analysis"])


def max_chars() -> int:
    try:
        return int(os.environ.get("MVGR_API_MAX_TEXT_CHARS", "50000"))
    except ValueError:
        return 50_000


@router.post("/analyze")
def post_analyze(request: AnalyzeRequest) -> dict:
    if len(request.text) > max_chars():
        raise HTTPException(
            status_code=413,
            detail=f"text exceeds MVGR_API_MAX_TEXT_CHARS={max_chars()}",
        )
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")
    return analyze(
        request.text,
        include_graph=request.include_graph,
        include_fingerprint=request.include_fingerprint,
    )


__all__ = ["max_chars", "router"]
