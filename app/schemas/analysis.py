"""API response models and the mapping from domain objects onto them.

The domain layer uses dataclasses; the API boundary uses Pydantic. Keeping them
separate means the scoring engine can be refactored without silently changing
the public contract, and the contract is what the OpenAPI schema documents.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.llm.service import FeedbackOutcome, feedback_to_dict
from app.core.parsing.jobdesc import ParsedJob
from app.core.scoring.aggregate import AnalysisResult
from app.core.scoring.base import DimensionId
from app.core.scoring.matching import MatchResult
from app.core.skills.matcher import SkillMention


class ContactOut(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None


class DateRangeOut(BaseModel):
    start_year: int | None = None
    start_month: int | None = None
    end_year: int | None = None
    end_month: int | None = None
    is_current: bool = False
    raw: str = ""


class ExperienceOut(BaseModel):
    title: str | None = None
    organization: str | None = None
    dates: DateRangeOut | None = None
    bullet_count: int = 0
    bullets: list[str] = Field(default_factory=list)


class EducationOut(BaseModel):
    degree: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    dates: DateRangeOut | None = None
    gpa: float | None = None


class SkillOut(BaseModel):
    name: str
    category: str
    confidence: float
    mentions: int
    demonstrated: bool = Field(
        description="True when the skill appears outside the skills list, i.e. shown in context."
    )
    evidence: list[str] = Field(default_factory=list)


class SignalOut(BaseModel):
    id: str
    label: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float
    severity: Literal["ok", "info", "warning", "critical"]
    detail: str | None = None
    evidence: list[str] = Field(default_factory=list)
    recommendation: str | None = None


class DimensionOut(BaseModel):
    id: str
    label: str
    score: float = Field(ge=0.0, le=100.0)
    weight: float
    severity: Literal["ok", "info", "warning", "critical"]
    signals: list[SignalOut] = Field(default_factory=list)


class RecommendationOut(BaseModel):
    id: str
    dimension: str
    severity: Literal["ok", "info", "warning", "critical"]
    title: str
    action: str
    evidence: list[str] = Field(default_factory=list)
    impact_points: float = Field(
        description="Points added to the overall score if this is fully addressed."
    )


class SkillGapOut(BaseModel):
    name: str
    category: str
    required: bool
    priority: Literal["high", "medium", "low"]
    adjacent_owned: list[str] = Field(
        default_factory=list,
        description="Related skills the candidate does have, useful for framing the gap.",
    )


class MatchOut(BaseModel):
    score: float
    required_coverage: float
    matched_required: list[str] = Field(default_factory=list)
    matched_preferred: list[str] = Field(default_factory=list)
    gaps: list[SkillGapOut] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    overlapping_terms: list[str] = Field(default_factory=list)
    surplus_skills: list[str] = Field(default_factory=list)


class JobOut(BaseModel):
    title: str | None = None
    company: str | None = None
    seniority: str
    min_years_experience: int | None = None
    degree_required: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class DocumentOut(BaseModel):
    filename: str | None = None
    source_format: str
    page_count: int
    word_count: int
    char_count: int
    warnings: list[str] = Field(default_factory=list)


class AiFeedbackOut(BaseModel):
    """Present only when the coaching layer ran successfully."""

    available: bool
    status: str = Field(description="Why AI feedback is or is not present.")
    content: dict[str, Any] | None = None


class AnalysisResponse(BaseModel):
    id: str
    created_at: datetime
    overall_score: float = Field(ge=0.0, le=100.0)
    band: Literal["excellent", "strong", "fair", "weak", "poor"]
    summary: str

    document: DocumentOut
    contact: ContactOut
    years_of_experience: float
    experience: list[ExperienceOut] = Field(default_factory=list)
    education: list[EducationOut] = Field(default_factory=list)
    skills: list[SkillOut] = Field(default_factory=list)

    dimensions: list[DimensionOut] = Field(default_factory=list)
    recommendations: list[RecommendationOut] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)

    job: JobOut | None = None
    match: MatchOut | None = None
    ai_feedback: AiFeedbackOut


class AnalysisSummary(BaseModel):
    """Row shape for the history endpoint."""

    id: str
    created_at: datetime
    filename: str | None
    overall_score: float
    job_title: str | None
    company: str | None
    llm_used: bool


class AnalysisListResponse(BaseModel):
    items: list[AnalysisSummary]
    total: int
    limit: int
    offset: int


class JobParseResponse(BaseModel):
    """Returned by the standalone job-parsing endpoint."""

    job: JobOut
    responsibilities: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    checks: dict[str, str]


# --------------------------------------------------------------------------
# Domain -> API mapping
# --------------------------------------------------------------------------


def _date_range(value: Any) -> DateRangeOut | None:
    if value is None:
        return None
    return DateRangeOut(
        start_year=value.start_year,
        start_month=value.start_month,
        end_year=value.end_year,
        end_month=value.end_month,
        is_current=value.is_current,
        raw=value.raw,
    )


def _skills_out(skills: dict[str, SkillMention]) -> list[SkillOut]:
    return [
        SkillOut(
            name=mention.name,
            category=mention.category,
            confidence=mention.confidence,
            mentions=mention.count,
            demonstrated=mention.is_demonstrated,
            evidence=mention.evidence[:2],
        )
        for mention in sorted(
            skills.values(), key=lambda m: (-m.confidence, -m.count, m.name)
        )
    ]


def job_out(job: ParsedJob) -> JobOut:
    return JobOut(
        title=job.title,
        company=job.company,
        seniority=job.seniority.value,
        min_years_experience=job.min_years_experience,
        degree_required=job.degree_required,
        required_skills=[s.name for s in job.required_skills],
        preferred_skills=[s.name for s in job.preferred_skills],
        keywords=job.keywords,
    )


def _match_out(match: MatchResult) -> MatchOut:
    return MatchOut(
        score=match.dimension.score,
        required_coverage=match.required_coverage,
        matched_required=match.matched_required,
        matched_preferred=match.matched_preferred,
        gaps=[
            SkillGapOut(
                name=gap.name,
                category=gap.category,
                required=gap.required,
                priority=gap.priority,
                adjacent_owned=gap.adjacent_owned,
            )
            for gap in match.gaps
        ],
        missing_keywords=match.missing_keywords,
        overlapping_terms=[term for term, _ in match.overlapping_terms],
        surplus_skills=match.surplus_skills,
    )


def build_summary(result: AnalysisResult) -> str:
    """The one-paragraph verdict shown above everything else."""
    parts = [
        f"Overall {result.overall_score}/100 ({result.band})."
    ]
    critical = result.critical_issues
    if critical:
        parts.append(
            f"{len(critical)} critical issue(s) would block or badly weaken this "
            f"application, starting with: {critical[0].title.lower()}."
        )
    if result.match and result.job:
        coverage = result.match.required_coverage
        parts.append(
            f"Covers {coverage:.0%} of the required skills for "
            f"{result.job.title or 'this role'}."
        )
    top = next((r for r in result.recommendations if r.impact >= 1.0), None)
    if top:
        parts.append(f"Highest-impact fix: {top.action}")
    return " ".join(parts)


def to_response(
    result: AnalysisResult,
    *,
    analysis_id: str,
    created_at: datetime,
    filename: str | None,
    feedback: FeedbackOutcome,
    include_signals: bool = True,
) -> AnalysisResponse:
    """Map a complete analysis onto the public response model."""
    weights = result.weights

    dimensions = [
        DimensionOut(
            id=dimension_id.value,
            label=dimension.label,
            score=dimension.score,
            weight=weights.get(dimension_id, 0.0),
            severity=dimension.severity.value,
            signals=[
                SignalOut(
                    id=signal.id,
                    label=signal.label,
                    score=signal.score,
                    weight=signal.weight,
                    severity=signal.severity.value,
                    detail=signal.detail,
                    evidence=signal.evidence,
                    recommendation=signal.recommendation,
                )
                for signal in dimension.signals
            ]
            if include_signals
            else [],
        )
        for dimension_id, dimension in _ordered_dimensions(result)
    ]

    return AnalysisResponse(
        id=analysis_id,
        created_at=created_at,
        overall_score=result.overall_score,
        band=result.band,
        summary=build_summary(result),
        document=DocumentOut(
            filename=filename,
            source_format=result.document.source_format.value,
            page_count=result.document.page_count,
            word_count=result.document.word_count,
            char_count=result.document.char_count,
            warnings=result.document.warnings,
        ),
        contact=ContactOut(
            name=result.resume.contact.name,
            email=result.resume.contact.email,
            phone=result.resume.contact.phone,
            location=result.resume.contact.location,
            linkedin=result.resume.contact.linkedin,
            github=result.resume.contact.github,
            website=result.resume.contact.website,
        ),
        years_of_experience=result.resume.years_of_experience,
        experience=[
            ExperienceOut(
                title=entry.title,
                organization=entry.organization,
                dates=_date_range(entry.dates),
                bullet_count=len(entry.bullets),
                bullets=entry.bullets[:6],
            )
            for entry in result.resume.experience
        ],
        education=[
            EducationOut(
                degree=entry.degree,
                field_of_study=entry.field_of_study,
                institution=entry.institution,
                dates=_date_range(entry.dates),
                gpa=entry.gpa,
            )
            for entry in result.resume.education
        ],
        skills=_skills_out(result.skills),
        dimensions=dimensions,
        recommendations=[
            RecommendationOut(
                id=recommendation.id,
                dimension=recommendation.dimension.value,
                severity=recommendation.severity.value,
                title=recommendation.title,
                action=recommendation.action,
                evidence=recommendation.evidence,
                impact_points=recommendation.impact,
            )
            for recommendation in result.recommendations
        ],
        strengths=sorted(set(result.strengths)),
        job=job_out(result.job) if result.job else None,
        match=_match_out(result.match) if result.match else None,
        ai_feedback=AiFeedbackOut(
            available=feedback.ok,
            status=(
                "AI coaching generated."
                if feedback.ok
                else (feedback.skipped_reason or "AI coaching unavailable.")
            ),
            content=feedback_to_dict(feedback.feedback) if feedback.feedback else None,
        ),
    )


def _ordered_dimensions(result: AnalysisResult) -> list[tuple[DimensionId, Any]]:
    """Stable presentation order: match first when present, then the rest."""
    order = [DimensionId.MATCH, DimensionId.ATS, DimensionId.CONTENT, DimensionId.STRUCTURE]
    return [(d, result.dimensions[d]) for d in order if d in result.dimensions]
