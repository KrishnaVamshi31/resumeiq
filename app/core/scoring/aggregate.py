"""Analysis orchestration: the single deterministic pipeline.

`analyze()` is the function the API, the CLI and the tests all call. It is
synchronous, pure (no I/O beyond loading the cached taxonomy) and produces the
same output for the same input - which is what makes the scores defensible and
the test suite meaningful. The LLM layer sits *on top* of this result and can
never change a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.core.extraction.base import ExtractedDocument
from app.core.parsing.jobdesc import ParsedJob, parse_job
from app.core.parsing.model import ParsedResume, SectionKind
from app.core.parsing.parser import parse_resume
from app.core.scoring.ats import score_ats, score_structure
from app.core.scoring.base import (
    WEIGHTS_WITH_JOB,
    WEIGHTS_WITHOUT_JOB,
    DimensionId,
    DimensionScore,
    Severity,
    Signal,
    band,
)
from app.core.scoring.content import score_content
from app.core.scoring.matching import MatchResult, score_match
from app.core.skills.matcher import SkillMention, extract_skills
from app.core.skills.taxonomy import Taxonomy, load_taxonomy
from app.observability import track

#: Map resume sections onto the context labels the skill matcher weights by.
_SECTION_CONTEXTS: dict[SectionKind, str] = {
    SectionKind.EXPERIENCE: "experience",
    SectionKind.PROJECTS: "projects",
    SectionKind.SUMMARY: "summary",
    SectionKind.SKILLS: "skills",
    SectionKind.CERTIFICATIONS: "certifications",
    SectionKind.EDUCATION: "education",
}


@dataclass(slots=True)
class Recommendation:
    """One prioritised, actionable fix."""

    id: str
    dimension: DimensionId
    severity: Severity
    title: str
    action: str
    evidence: list[str] = field(default_factory=list)
    #: Estimated points added to the overall score if fully addressed.
    impact: float = 0.0


@dataclass(slots=True)
class AnalysisResult:
    """Everything the deterministic pipeline knows about one resume."""

    resume: ParsedResume
    document: ExtractedDocument
    dimensions: dict[DimensionId, DimensionScore]
    skills: dict[str, SkillMention]
    recommendations: list[Recommendation]
    job: ParsedJob | None = None
    match: MatchResult | None = None

    @property
    def weights(self) -> dict[DimensionId, float]:
        return WEIGHTS_WITH_JOB if self.match else WEIGHTS_WITHOUT_JOB

    @property
    def overall_score(self) -> float:
        weights = self.weights
        total = sum(weights.values())
        weighted = sum(
            self.dimensions[dim].score * weight
            for dim, weight in weights.items()
            if dim in self.dimensions
        )
        return round(weighted / total, 1) if total else 0.0

    @property
    def band(self) -> Literal["excellent", "strong", "fair", "weak", "poor"]:
        return band(self.overall_score)

    @property
    def critical_issues(self) -> list[Recommendation]:
        return [r for r in self.recommendations if r.severity is Severity.CRITICAL]

    @property
    def strengths(self) -> list[str]:
        """Signals scoring full marks - what the resume already does well."""
        out: list[str] = []
        for dimension in self.dimensions.values():
            out.extend(s.label for s in dimension.signals if s.score >= 0.95)
        return out


def analyze(
    document: ExtractedDocument,
    *,
    job_text: str | None = None,
    job_title: str | None = None,
    company: str | None = None,
    taxonomy: Taxonomy | None = None,
) -> AnalysisResult:
    """Run the full deterministic analysis."""
    taxonomy = taxonomy or load_taxonomy()

    with track("analyze") as ctx:
        resume = parse_resume(document.text)
        skills = extract_skills(skill_contexts(resume), taxonomy)

        dimensions: dict[DimensionId, DimensionScore] = {
            DimensionId.ATS: score_ats(document, resume),
            DimensionId.STRUCTURE: score_structure(document, resume),
            DimensionId.CONTENT: score_content(resume),
        }

        job: ParsedJob | None = None
        match: MatchResult | None = None
        if job_text and job_text.strip():
            job = parse_job(job_text, title=job_title, company=company, taxonomy=taxonomy)
            match = score_match(resume, job, skills, taxonomy)
            dimensions[DimensionId.MATCH] = match.dimension

        result = AnalysisResult(
            resume=resume,
            document=document,
            dimensions=dimensions,
            skills=skills,
            recommendations=[],
            job=job,
            match=match,
        )
        result.recommendations = build_recommendations(result)

        ctx["overall"] = result.overall_score
        ctx["skills"] = len(skills)
        ctx["with_job"] = bool(job)

    return result


def skill_contexts(resume: ParsedResume) -> dict[str, str]:
    """Group resume text by the context labels that drive skill confidence."""
    contexts: dict[str, str] = {}
    for kind, label in _SECTION_CONTEXTS.items():
        section = resume.sections.get(kind)
        if section and section.body.strip():
            contexts[label] = section.body

    # Anything outside a recognised section still counts, at the lowest weight.
    covered = {resume.sections[k].body for k in _SECTION_CONTEXTS if k in resume.sections}
    leftover = resume.raw_text
    for body in covered:
        leftover = leftover.replace(body, " ")
    if leftover.strip():
        contexts["other"] = leftover
    return contexts


def build_recommendations(result: AnalysisResult) -> list[Recommendation]:
    """Turn every under-performing signal into a ranked, actionable fix.

    Impact is the number of *overall* points recovered by taking the signal to
    full marks, so the ordering reflects real return on effort rather than raw
    severity. That is the difference between a list of complaints and advice.
    """
    weights = result.weights
    recommendations: list[Recommendation] = []

    for dimension_id, dimension in result.dimensions.items():
        dimension_weight = weights.get(dimension_id, 0.0)
        signal_weight_total = sum(s.weight for s in dimension.signals) or 1.0
        weight_total = sum(weights.values()) or 1.0

        for signal in dimension.signals:
            if signal.score >= 0.95 or not signal.recommendation:
                continue
            # Points on the 0-100 overall scale recovered by fixing this signal.
            impact = (
                (1.0 - signal.score)
                * (signal.weight / signal_weight_total)
                * (dimension_weight / weight_total)
                * 100.0
            )
            recommendations.append(
                Recommendation(
                    id=signal.id,
                    dimension=dimension_id,
                    severity=signal.severity,
                    title=signal.label,
                    action=signal.recommendation,
                    evidence=signal.evidence[:4],
                    impact=round(impact, 2),
                )
            )

    recommendations.sort(key=_recommendation_sort_key)
    return recommendations


def _recommendation_sort_key(recommendation: Recommendation) -> tuple[int, float, str]:
    """Critical items first, then by impact. Ties broken by id for determinism."""
    severity_rank = {
        Severity.CRITICAL: 0,
        Severity.WARNING: 1,
        Severity.INFO: 2,
        Severity.OK: 3,
    }
    return (severity_rank[recommendation.severity], -recommendation.impact, recommendation.id)


def signal_index(result: AnalysisResult) -> dict[str, Signal]:
    """Flat id -> signal lookup, used by the report and API layers."""
    return {
        signal.id: signal
        for dimension in result.dimensions.values()
        for signal in dimension.signals
    }
