"""Shared test configuration and fixtures.

Environment variables are set at import time, before anything under `app` is
imported, because `get_settings()` is cached for the life of the process.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="resumeiq-tests-")
os.environ["RESUMEIQ_DATABASE_URL"] = f"sqlite:///{Path(_TMP_DIR).as_posix()}/test.db"
os.environ["RESUMEIQ_ENV"] = "development"
os.environ["RESUMEIQ_LOG_FORMAT"] = "console"
os.environ["RESUMEIQ_LOG_LEVEL"] = "WARNING"
os.environ["RESUMEIQ_LLM_ENABLED"] = "false"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["RESUMEIQ_RATE_LIMIT_REQUESTS"] = "10000"

from fastapi.testclient import TestClient  # noqa: E402

from app.core.extraction.base import ExtractedDocument, PageStats, SourceFormat  # noqa: E402
from app.core.parsing.parser import parse_resume  # noqa: E402
from app.core.scoring.aggregate import AnalysisResult, analyze  # noqa: E402
from app.main import create_app  # noqa: E402
from app.observability import metrics  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_document(text: str, fmt: SourceFormat = SourceFormat.TXT, **kwargs) -> ExtractedDocument:
    """Build an ExtractedDocument without going through a real file."""
    return ExtractedDocument(
        text=text,
        source_format=fmt,
        page_count=kwargs.pop("page_count", max(1, round(len(text.split()) / 500) or 1)),
        pages=kwargs.pop(
            "pages",
            [
                PageStats(
                    index=0,
                    char_count=len(text),
                    line_count=len(text.splitlines()),
                    has_extractable_text=True,
                )
            ],
        ),
        **kwargs,
    )


@pytest.fixture(scope="session")
def strong_resume_text() -> str:
    return read_fixture("strong_resume.txt")


@pytest.fixture(scope="session")
def weak_resume_text() -> str:
    return read_fixture("weak_resume.txt")


@pytest.fixture(scope="session")
def job_text() -> str:
    return read_fixture("backend_job.txt")


@pytest.fixture
def strong_resume(strong_resume_text: str):
    return parse_resume(strong_resume_text)


@pytest.fixture
def strong_analysis(strong_resume_text: str) -> AnalysisResult:
    return analyze(make_document(strong_resume_text))


@pytest.fixture
def matched_analysis(strong_resume_text: str, job_text: str) -> AnalysisResult:
    return analyze(make_document(strong_resume_text), job_text=job_text)


@pytest.fixture
def client() -> Iterator[TestClient]:
    metrics.reset()
    with TestClient(create_app()) as test_client:
        yield test_client
