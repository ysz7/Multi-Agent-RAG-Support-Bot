"""Liveness and configuration surface."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_settings_dep
from app.api.schemas import HealthResponse
from app.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings_dep)) -> HealthResponse:
    """Report configuration and backend reachability.

    Deliberately reports *which* backends are configured, never any secret.
    """
    checks: dict[str, str] = {}

    try:
        from psycopg import AsyncConnection

        connection = await AsyncConnection.connect(settings.database_url, connect_timeout=3)
        await connection.close()
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"

    checks["langfuse"] = "configured" if settings.langfuse_enabled else "disabled"

    return HealthResponse(
        status="ok" if checks.get("database") == "ok" else "degraded",
        auth_mode=settings.auth_mode,
        llm_provider=settings.llm_provider,
        vector_store=settings.vector_store,
        database=checks.get("database", "unknown"),
        checks=checks,
    )
