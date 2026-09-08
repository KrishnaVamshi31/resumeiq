"""Health, readiness and metrics endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text as sql_text

from app.config import Settings, get_settings
from app.core.llm.client import get_llm_client
from app.core.skills.taxonomy import load_taxonomy
from app.db.session import session_scope
from app.logging_conf import get_logger
from app.observability import metrics
from app.schemas.analysis import HealthResponse

logger = get_logger(__name__)

router = APIRouter(tags=["ops"])

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Cheap liveness check. Never touches the database or the network."""
    return HealthResponse(
        status="ok", version=VERSION, environment=settings.env, checks={"process": "ok"}
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness(
    settings: Annotated[Settings, Depends(get_settings)], response: Response
) -> HealthResponse:
    """Verifies the dependencies a request actually needs.

    The LLM is reported but never fails readiness: the service is fully
    functional without it, so an outage at the provider must not take this
    service out of the load balancer.
    """
    checks: dict[str, str] = {}

    try:
        with session_scope() as session:
            session.execute(sql_text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.error("readiness_database_failed", extra={"detail": str(exc)})
        checks["database"] = "error"

    taxonomy = load_taxonomy()
    checks["taxonomy"] = "ok" if len(taxonomy) > 0 else "error"

    client = get_llm_client(settings)
    if not settings.llm_enabled:
        checks["llm"] = "disabled"
    elif not settings.anthropic_api_key:
        checks["llm"] = "unconfigured"
    elif not client.available:
        checks["llm"] = "circuit_open"
    else:
        checks["llm"] = "ok"

    hard_failure = any(v == "error" for v in checks.values())
    if hard_failure:
        response.status_code = 503

    return HealthResponse(
        status="degraded" if hard_failure else "ok",
        version=VERSION,
        environment=settings.env,
        checks=checks,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
    responses={200: {"content": {"text/plain": {}}}},
)
async def prometheus_metrics() -> Response:
    return Response(content=metrics.render_prometheus(), media_type="text/plain; version=0.0.4")


@router.get("/metrics/json", summary="Metrics as JSON")
async def json_metrics() -> dict[str, Any]:
    return metrics.snapshot()
