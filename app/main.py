"""FastAPI application factory and process lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api import routes_analysis, routes_health, routes_jobs
from app.config import get_settings
from app.core.llm.client import close_llm_client
from app.core.skills.taxonomy import load_taxonomy
from app.db.session import dispose_db, init_db
from app.errors import register_exception_handlers
from app.logging_conf import configure_logging, get_logger
from app.observability import RequestContextMiddleware

logger = get_logger(__name__)

DESCRIPTION = """\
**ResumeIQ** scores a resume on four dimensions - ATS compatibility, content
quality, structure and job match - and returns every signal that produced the
score, ranked by how many points fixing it would recover.

Scoring is fully deterministic: the same input always produces the same numbers,
with no model call involved. When an Anthropic API key is configured, a coaching
layer adds narrative feedback and concrete rewrites on top; that layer can never
alter a score, and the service degrades cleanly without it.
"""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    init_db()
    taxonomy = load_taxonomy()  # warm the cache so the first request is not slow
    logger.info(
        "startup",
        extra={
            "env": settings.env,
            "skills": len(taxonomy),
            "llm": "enabled" if settings.llm_available else "disabled",
            "model": settings.llm_model if settings.llm_available else None,
        },
    )
    try:
        yield
    finally:
        await close_llm_client()
        dispose_db()
        logger.info("shutdown")


def create_app() -> FastAPI:
    """Build the application. Used by uvicorn, the tests and the CLI."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title="ResumeIQ - AI Resume Intelligence & Job Matching",
        description=DESCRIPTION,
        version=routes_health.VERSION,
        lifespan=lifespan,
        # Interactive docs are useful in development and a needless surface in
        # production, where the OpenAPI JSON alone is enough for clients.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )

    register_exception_handlers(app)

    app.include_router(routes_health.router)
    app.include_router(routes_analysis.router)
    app.include_router(routes_jobs.router)

    return app


app = create_app()
