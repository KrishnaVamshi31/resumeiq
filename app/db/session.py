"""Database engine, session factory and the repository functions.

Sessions are synchronous. Async endpoints call these through
`starlette.concurrency.run_in_threadpool` so a slow write never blocks the event
loop, which keeps the door open for swapping SQLite for Postgres without
rewriting the call sites.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import AnalysisRecord, Base
from app.logging_conf import get_logger

logger = get_logger(__name__)

_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def _build_engine():  # type: ignore[no-untyped-def]
    settings = get_settings()
    url = settings.database_url
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite needs the parent directory to exist, and sessions are handed
        # between threadpool workers.
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db() -> None:
    """Create the engine and schema. Called once on startup."""
    global _engine, _SessionFactory
    if _engine is None:
        _engine = _build_engine()
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(_engine)
        logger.info("database_ready", extra={"url": get_settings().database_url})


def dispose_db() -> None:
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionFactory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    if _SessionFactory is None:
        init_db()
    assert _SessionFactory is not None
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------
# Repository functions
# --------------------------------------------------------------------------


def save_analysis(record: AnalysisRecord) -> AnalysisRecord:
    with session_scope() as session:
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def get_analysis(analysis_id: str) -> AnalysisRecord | None:
    with session_scope() as session:
        return session.get(AnalysisRecord, analysis_id)


def list_analyses(limit: int = 25, offset: int = 0) -> tuple[list[AnalysisRecord], int]:
    """Most recent first, with the total count for pagination."""
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(AnalysisRecord)) or 0
        rows = list(
            session.scalars(
                select(AnalysisRecord)
                .order_by(AnalysisRecord.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
        )
        return rows, total


def delete_analysis(analysis_id: str) -> bool:
    with session_scope() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            return False
        session.delete(record)
        return True
