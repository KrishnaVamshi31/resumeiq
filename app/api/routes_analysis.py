"""Analysis endpoints - the core of the public API."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import rate_limit
from app.config import Settings, get_settings
from app.core.extraction.base import ExtractedDocument, PageStats, SourceFormat
from app.core.extraction.detector import extract
from app.core.llm.service import FeedbackOutcome, generate_feedback
from app.core.scoring.aggregate import AnalysisResult, analyze
from app.core.scoring.base import DimensionId
from app.db.models import AnalysisRecord
from app.db.session import delete_analysis, get_analysis, list_analyses, save_analysis
from app.errors import EmptyDocument, ResourceNotFound
from app.logging_conf import get_logger
from app.schemas.analysis import (
    AnalysisListResponse,
    AnalysisResponse,
    AnalysisSummary,
    to_response,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/analyses", tags=["analysis"])

MAX_JOB_DESCRIPTION_CHARS = 30_000


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyse an uploaded resume",
    dependencies=[Depends(rate_limit)],
)
async def create_analysis(
    file: Annotated[UploadFile, File(description="Resume as PDF, DOCX or TXT.")],
    settings: Annotated[Settings, Depends(get_settings)],
    job_description: Annotated[str | None, Form()] = None,
    job_title: Annotated[str | None, Form()] = None,
    company: Annotated[str | None, Form()] = None,
    use_ai: Annotated[bool, Form()] = True,
) -> AnalysisResponse:
    """Score a resume, optionally against a job description.

    The deterministic analysis always runs. AI coaching is layered on top when
    it is enabled and reachable; if it is not, the response still carries every
    score and recommendation and says why the narrative is missing.
    """
    payload = await file.read()
    if not payload:
        raise EmptyDocument("The uploaded file is empty.")

    document = await run_in_threadpool(
        extract,
        payload,
        file.filename,
        max_bytes=settings.max_upload_bytes,
        max_chars=settings.max_resume_chars,
    )
    return await _run_pipeline(
        document=document,
        filename=file.filename,
        content_hash=hashlib.sha256(payload).hexdigest(),
        job_description=job_description,
        job_title=job_title,
        company=company,
        use_ai=use_ai,
        settings=settings,
    )


@router.post(
    "/text",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyse resume text directly",
    dependencies=[Depends(rate_limit)],
)
async def create_analysis_from_text(
    resume_text: Annotated[str, Form(description="Plain resume text.")],
    settings: Annotated[Settings, Depends(get_settings)],
    job_description: Annotated[str | None, Form()] = None,
    job_title: Annotated[str | None, Form()] = None,
    company: Annotated[str | None, Form()] = None,
    use_ai: Annotated[bool, Form()] = True,
) -> AnalysisResponse:
    """Same pipeline without a file upload - convenient for integrations.

    Layout-based ATS checks (columns, tables, header regions) cannot fire on raw
    text, so those signals score neutrally; this is documented in the response
    warnings rather than silently assumed.
    """
    text = (resume_text or "").strip()
    if not text:
        raise EmptyDocument("`resume_text` is empty.")
    text = text[: settings.max_resume_chars]

    document = ExtractedDocument(
        text=text,
        source_format=SourceFormat.TXT,
        page_count=max(1, round(len(text.split()) / 500) or 1),
        pages=[
            PageStats(
                index=0,
                char_count=len(text),
                line_count=len(text.splitlines()),
                has_extractable_text=True,
            )
        ],
        warnings=[
            "Submitted as raw text: layout-dependent ATS checks (columns, tables, "
            "headers) could not be evaluated."
        ],
    )
    return await _run_pipeline(
        document=document,
        filename=None,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        job_description=job_description,
        job_title=job_title,
        company=company,
        use_ai=use_ai,
        settings=settings,
    )


@router.get("", response_model=AnalysisListResponse, summary="List previous analyses")
async def list_previous(
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AnalysisListResponse:
    rows, total = await run_in_threadpool(list_analyses, limit, offset)
    return AnalysisListResponse(
        items=[
            AnalysisSummary(
                id=row.id,
                created_at=row.created_at,
                filename=row.filename,
                overall_score=row.overall_score,
                job_title=row.job_title,
                company=row.company,
                llm_used=row.llm_used,
            )
            for row in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse, summary="Fetch a stored analysis")
async def fetch_analysis(analysis_id: str) -> AnalysisResponse:
    record = await run_in_threadpool(get_analysis, analysis_id)
    if record is None:
        raise ResourceNotFound(f"No analysis with id {analysis_id!r}.")
    # The stored payload is the exact response that was returned originally.
    return AnalysisResponse.model_validate(record.payload)


@router.delete(
    "/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a stored analysis"
)
async def remove_analysis(analysis_id: str) -> None:
    deleted = await run_in_threadpool(delete_analysis, analysis_id)
    if not deleted:
        raise ResourceNotFound(f"No analysis with id {analysis_id!r}.")


# --------------------------------------------------------------------------


async def _run_pipeline(
    *,
    document: ExtractedDocument,
    filename: str | None,
    content_hash: str,
    job_description: str | None,
    job_title: str | None,
    company: str | None,
    use_ai: bool,
    settings: Settings,
) -> AnalysisResponse:
    """Deterministic analysis, then optional coaching, then persistence."""
    job_text = (job_description or "").strip()[:MAX_JOB_DESCRIPTION_CHARS] or None

    result: AnalysisResult = await run_in_threadpool(
        analyze, document, job_text=job_text, job_title=job_title, company=company
    )

    if use_ai:
        feedback = await generate_feedback(result)
    else:
        feedback = FeedbackOutcome(skipped_reason="AI coaching was not requested.")

    analysis_id = uuid.uuid4().hex
    created_at = datetime.now(UTC)

    record = AnalysisRecord(
        id=analysis_id,
        created_at=created_at,
        filename=filename,
        source_format=document.source_format.value,
        content_hash=content_hash,
        overall_score=result.overall_score,
        ats_score=result.dimensions[DimensionId.ATS].score,
        content_score=result.dimensions[DimensionId.CONTENT].score,
        structure_score=result.dimensions[DimensionId.STRUCTURE].score,
        match_score=(
            result.dimensions[DimensionId.MATCH].score
            if DimensionId.MATCH in result.dimensions
            else None
        ),
        job_title=(result.job.title if result.job else job_title),
        company=company,
        word_count=document.word_count,
        skill_count=len(result.skills),
        llm_used=feedback.ok,
        resume_text=document.text,
    )

    response = to_response(
        result,
        analysis_id=analysis_id,
        created_at=created_at,
        filename=filename,
        feedback=feedback,
    )
    record.payload = response.model_dump(mode="json")

    try:
        await run_in_threadpool(save_analysis, record)
    except Exception:
        # Persistence is a convenience, not a correctness requirement: never
        # fail an analysis the caller is waiting for because the store is down.
        logger.exception("analysis_persist_failed", extra={"analysis_id": analysis_id})

    return response
