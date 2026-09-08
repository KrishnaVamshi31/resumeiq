"""The coaching layer: deterministic analysis in, validated narrative out.

Everything here is optional by design. If no API key is configured, the model
refuses, the provider is down, or the response fails validation, the service
returns `None` and the caller still has a complete deterministic analysis. The
narrative is an enhancement, never a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings, get_settings
from app.core.llm.client import AnthropicClient, LLMError, LLMRefusal, get_llm_client
from app.core.llm.guards import (
    Redactor,
    find_fabricated_metrics,
    new_nonce,
    sanitise,
    strip_placeholders,
)
from app.core.llm.prompts import (
    FEEDBACK_SCHEMA,
    MAX_BULLET_REWRITES,
    MAX_PRIORITY_FIXES,
    SYSTEM_PROMPT,
    build_user_message,
)
from app.core.scoring.aggregate import AnalysisResult
from app.logging_conf import get_logger
from app.observability import metrics

logger = get_logger(__name__)

MAX_FIELD_CHARS = 1200
MAX_LIST_ITEMS = 8


class PriorityFix(BaseModel):
    target: str = Field(max_length=200)
    problem: str = Field(max_length=MAX_FIELD_CHARS)
    fix: str = Field(max_length=MAX_FIELD_CHARS)
    why_it_matters: str = Field(max_length=MAX_FIELD_CHARS)


class BulletRewrite(BaseModel):
    original: str = Field(max_length=MAX_FIELD_CHARS)
    improved: str = Field(max_length=MAX_FIELD_CHARS)
    rationale: str = Field(max_length=MAX_FIELD_CHARS)


class FeedbackPayload(BaseModel):
    """Schema the model output must satisfy before we will show it."""

    headline: str = Field(max_length=400)
    strengths: list[str] = Field(default_factory=list)
    priority_fixes: list[PriorityFix] = Field(default_factory=list)
    bullet_rewrites: list[BulletRewrite] = Field(default_factory=list)
    summary_rewrite: str = Field(default="", max_length=MAX_FIELD_CHARS)
    keyword_guidance: list[str] = Field(default_factory=list)
    interview_risks: list[str] = Field(default_factory=list)
    integrity_notes: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class LLMFeedback:
    """Validated coaching output plus the provenance a reviewer needs."""

    payload: FeedbackPayload
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    latency_ms: float
    request_id: str | None = None
    #: Rewrites dropped because they asserted numbers not present in the resume.
    dropped_rewrites: list[str] = field(default_factory=list)
    #: Injection attempts detected in the submitted documents.
    injection_findings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeedbackOutcome:
    """Either feedback, or the reason there is none. Never an exception."""

    feedback: LLMFeedback | None = None
    skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.feedback is not None


async def generate_feedback(
    analysis: AnalysisResult,
    *,
    client: AnthropicClient | None = None,
    settings: Settings | None = None,
) -> FeedbackOutcome:
    """Produce the narrative coaching layer, degrading gracefully on any failure."""
    settings = settings or get_settings()
    client = client or get_llm_client(settings)

    if not settings.llm_enabled:
        return FeedbackOutcome(skipped_reason="AI coaching is disabled by configuration.")
    if not settings.anthropic_api_key:
        return FeedbackOutcome(
            skipped_reason=(
                "AI coaching is unavailable: no ANTHROPIC_API_KEY is configured. "
                "All scores and recommendations below are unaffected."
            )
        )
    if not client.available:
        return FeedbackOutcome(
            skipped_reason="AI coaching is temporarily unavailable (provider errors); "
            "deterministic analysis is unaffected."
        )

    nonce = new_nonce()
    redactor = Redactor(analysis.resume.contact, enabled=settings.llm_redact_pii)

    resume_text = sanitise(redactor.redact(analysis.resume.raw_text), nonce)
    job_text = (
        sanitise(analysis.job.raw_text, nonce) if analysis.job is not None else None
    )

    user_message = build_user_message(analysis, resume_text, job_text, nonce)

    try:
        response = await client.structured(
            system=SYSTEM_PROMPT,
            user_message=user_message,
            schema=FEEDBACK_SCHEMA,
        )
    except LLMRefusal as exc:
        logger.warning("llm_refusal", extra={"detail": str(exc)})
        return FeedbackOutcome(
            skipped_reason="The model declined to analyse this document. "
            "Deterministic analysis is unaffected."
        )
    except LLMError as exc:
        logger.warning("llm_unavailable", extra={"detail": str(exc), "retryable": exc.retryable})
        metrics.increment("resumeiq_llm_failure_total", {"retryable": str(exc.retryable)})
        return FeedbackOutcome(
            skipped_reason=f"AI coaching could not be generated ({exc}). "
            "Deterministic analysis is unaffected."
        )

    try:
        payload = FeedbackPayload.model_validate(response.data)
    except ValidationError as exc:
        logger.warning("llm_schema_violation", extra={"errors": exc.error_count()})
        metrics.increment("resumeiq_llm_schema_violation_total")
        return FeedbackOutcome(
            skipped_reason="The AI response did not match the expected format and was discarded."
        )

    payload, dropped = _post_validate(payload, analysis, redactor)

    findings = [f"{f.kind}: {f.excerpt[:120]}" for f in resume_text.findings]
    if job_text:
        findings += [f"job_posting/{f.kind}: {f.excerpt[:120]}" for f in job_text.findings]

    metrics.increment("resumeiq_llm_success_total")
    return FeedbackOutcome(
        feedback=LLMFeedback(
            payload=payload,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            latency_ms=response.latency_ms,
            request_id=response.request_id,
            dropped_rewrites=dropped,
            injection_findings=findings,
        )
    )


def _post_validate(
    payload: FeedbackPayload, analysis: AnalysisResult, redactor: Redactor
) -> tuple[FeedbackPayload, list[str]]:
    """Enforce the rules the schema alone cannot: no fabrication, no bloat.

    A model can satisfy a JSON schema and still assert "increased revenue 40%"
    for a candidate who never mentioned revenue. Those rewrites are dropped
    rather than shown, because a resume tool that invents achievements is worse
    than one that says nothing.
    """
    source = analysis.resume.raw_text
    kept: list[BulletRewrite] = []
    dropped: list[str] = []

    for rewrite in payload.bullet_rewrites[:MAX_BULLET_REWRITES]:
        fabricated = find_fabricated_metrics(rewrite.improved, source)
        if fabricated:
            dropped.append(
                f'Dropped a rewrite asserting unverifiable figures ({", ".join(fabricated)}): '
                f'"{rewrite.improved[:120]}"'
            )
            metrics.increment("resumeiq_llm_fabrication_blocked_total")
            continue
        kept.append(
            BulletRewrite(
                original=redactor.restore(rewrite.original),
                improved=redactor.restore(rewrite.improved),
                rationale=redactor.restore(rewrite.rationale),
            )
        )

    summary = redactor.restore(payload.summary_rewrite).strip()
    if summary and find_fabricated_metrics(summary, source):
        dropped.append("Dropped the suggested summary: it asserted figures not in the resume.")
        summary = ""

    return (
        FeedbackPayload(
            headline=redactor.restore(payload.headline).strip(),
            strengths=_clean_list(payload.strengths, redactor),
            priority_fixes=[
                PriorityFix(
                    target=redactor.restore(fix.target),
                    problem=redactor.restore(fix.problem),
                    fix=redactor.restore(fix.fix),
                    why_it_matters=redactor.restore(fix.why_it_matters),
                )
                for fix in payload.priority_fixes[:MAX_PRIORITY_FIXES]
            ],
            bullet_rewrites=kept,
            summary_rewrite=strip_placeholders(summary) if "[" in summary else summary,
            keyword_guidance=_clean_list(payload.keyword_guidance, redactor, limit=15),
            interview_risks=_clean_list(payload.interview_risks, redactor),
            integrity_notes=_clean_list(payload.integrity_notes, redactor),
        ),
        dropped,
    )


def _clean_list(items: list[str], redactor: Redactor, limit: int = MAX_LIST_ITEMS) -> list[str]:
    cleaned: list[str] = []
    for item in items[:limit]:
        text = redactor.restore(item).strip()
        if text and text not in cleaned:
            cleaned.append(text[:MAX_FIELD_CHARS])
    return cleaned


def feedback_to_dict(feedback: LLMFeedback) -> dict[str, Any]:
    """Serialise for the API layer."""
    return {
        **feedback.payload.model_dump(),
        "model": feedback.model,
        "usage": {
            "input_tokens": feedback.input_tokens,
            "output_tokens": feedback.output_tokens,
            "cache_read_tokens": feedback.cache_read_tokens,
            "latency_ms": feedback.latency_ms,
        },
        "dropped_rewrites": feedback.dropped_rewrites,
        "injection_findings": feedback.injection_findings,
        "request_id": feedback.request_id,
    }
