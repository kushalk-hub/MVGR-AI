"""Contract-version endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from shared.contracts.versions import CONTRACT_VERSIONS

router = APIRouter(prefix="/v1/contracts", tags=["contracts"])


@router.get("/versions")
def get_contract_versions() -> dict:
    return {"contracts": CONTRACT_VERSIONS}


__all__ = ["router"]
