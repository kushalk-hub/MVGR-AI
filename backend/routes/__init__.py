"""HTTP routes for the MVGR-AI API."""

from backend.routes.analyze import router as analyze_router
from backend.routes.contracts import router as contracts_router
from backend.routes.health import router as health_router

ROUTERS = (health_router, contracts_router, analyze_router)

__all__ = ["ROUTERS", "analyze_router", "contracts_router", "health_router"]
