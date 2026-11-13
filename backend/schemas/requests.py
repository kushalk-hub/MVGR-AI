"""Request models for the backend API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    include_graph: bool = True
    include_fingerprint: bool = True


class HealthResponse(BaseModel):
    status: str
    model_bundle: str
    model_available: bool


__all__ = ["AnalyzeRequest", "HealthResponse"]
