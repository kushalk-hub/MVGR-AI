"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import HealthResponse
from backend.services import health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> dict:
    return health()
