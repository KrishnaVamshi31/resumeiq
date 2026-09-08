"""Job description endpoints.

Exposed separately so a client can inspect how a posting was interpreted -
which skills were read as required versus preferred - before running an
analysis against it. Parsing mistakes are far easier to spot here than inside a
composite score.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import rate_limit
from app.core.parsing.jobdesc import parse_job
from app.errors import EmptyDocument
from app.schemas.analysis import JobParseResponse, job_out

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


class JobParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30_000, description="Raw job posting text.")
    title: str | None = None
    company: str | None = None


@router.post(
    "/parse",
    response_model=JobParseResponse,
    summary="Parse a job posting into structured requirements",
    dependencies=[Depends(rate_limit)],
)
async def parse_job_posting(
    payload: Annotated[JobParseRequest, Body()],
) -> JobParseResponse:
    if not payload.text.strip():
        raise EmptyDocument("`text` is empty.")

    job = await run_in_threadpool(
        parse_job, payload.text, title=payload.title, company=payload.company
    )
    return JobParseResponse(job=job_out(job), responsibilities=job.responsibilities)
