"""Persistence models.

Only what is genuinely worth keeping: the analysis envelope for history and
trend views, plus the columns you would actually want to filter or aggregate
on. The full response is stored as JSON so the API contract can evolve without
a migration for every new field.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class AnalysisRecord(Base):
    """One completed analysis."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    filename: Mapped[str | None] = mapped_column(String(255))
    source_format: Mapped[str] = mapped_column(String(16))
    #: SHA-256 of the uploaded bytes: lets the UI spot re-uploads of the same file.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    overall_score: Mapped[float] = mapped_column(Float)
    ats_score: Mapped[float] = mapped_column(Float)
    content_score: Mapped[float] = mapped_column(Float)
    structure_score: Mapped[float] = mapped_column(Float)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    job_title: Mapped[str | None] = mapped_column(String(255))
    company: Mapped[str | None] = mapped_column(String(255))

    word_count: Mapped[int] = mapped_column(Integer, default=0)
    skill_count: Mapped[int] = mapped_column(Integer, default=0)
    llm_used: Mapped[bool] = mapped_column(default=False)

    #: The complete API response, so history replays exactly what was shown.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Redacted resume text, retained only to support re-analysis against new jobs.
    resume_text: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_analyses_created_score", "created_at", "overall_score"),)
