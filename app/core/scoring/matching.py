"""Resume-to-job matching and gap analysis.

The match score answers one question: if a recruiter compared this resume
against this posting, how much of what they asked for would they find?

It is composed of five sub-signals, weighted so that missing a *required* skill
hurts far more than missing a nice-to-have, and so that lexical similarity -
the least trustworthy of the five - can never dominate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from app.core.embeddings.vectorizer import similarity, top_overlapping_terms
from app.core.parsing.jobdesc import (
    SENIORITY_RANK,
    ParsedJob,
    Seniority,
    SkillRequirement,
    detect_seniority,
)
from app.core.parsing.model import ParsedResume, SectionKind
from app.core.parsing.normalize import content_tokens
from app.core.scoring.base import DimensionId, DimensionScore, Severity, Signal, make_signal
from app.core.skills.matcher import SkillMention
from app.core.skills.taxonomy import Taxonomy, load_taxonomy


@dataclass(slots=True)
class SkillGap:
    """A skill the posting wants that the resume does not evidence."""

    name: str
    required: bool
    category: str
    #: Skills the candidate *does* have that sit next to this one in the ontology.
    adjacent_owned: list[str] = field(default_factory=list)

    @property
    def priority(self) -> Literal["high", "medium", "low"]:
        if self.required:
            return "high" if not self.adjacent_owned else "medium"
        return "low"


@dataclass(slots=True)
class MatchResult:
    """The full matching picture, not just a number."""

    dimension: DimensionScore
    matched_required: list[str] = field(default_factory=list)
    matched_preferred: list[str] = field(default_factory=list)
    gaps: list[SkillGap] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    overlapping_terms: list[tuple[str, float]] = field(default_factory=list)
    #: Skills the candidate has that the posting never asks for.
    surplus_skills: list[str] = field(default_factory=list)

    @property
    def required_coverage(self) -> float:
        total = len(self.matched_required) + sum(1 for g in self.gaps if g.required)
        return round(len(self.matched_required) / total, 3) if total else 1.0

    @property
    def critical_gaps(self) -> list[SkillGap]:
        return [g for g in self.gaps if g.required]


def score_match(
    resume: ParsedResume,
    job: ParsedJob,
    resume_skills: Mapping[str, SkillMention],
    taxonomy: Taxonomy | None = None,
) -> MatchResult:
    """Score the fit between a parsed resume and a parsed job posting."""
    taxonomy = taxonomy or load_taxonomy()
    owned = set(resume_skills)

    required_hits, required_misses = _partition(job.required_skills, owned)
    preferred_hits, preferred_misses = _partition(job.preferred_skills, owned)

    gaps = [
        SkillGap(
            name=req.name,
            required=req.required,
            category=req.category,
            adjacent_owned=sorted(set(taxonomy.related_to(req.name)) & owned),
        )
        for req in (*required_misses, *preferred_misses)
    ]
    gaps.sort(key=lambda g: (not g.required, g.name))

    resume_text = resume.raw_text
    missing_keywords = _missing_keywords(job, resume_text)
    text_similarity = similarity(resume_text, job.raw_text)

    signals: list[Signal] = [
        _required_signal(job.required_skills, required_hits, required_misses, resume_skills),
        _preferred_signal(job.preferred_skills, preferred_hits),
        _keyword_signal(job, missing_keywords),
        _similarity_signal(text_similarity),
        _seniority_signal(resume, job),
        _experience_signal(resume, job),
    ]

    return MatchResult(
        dimension=DimensionScore(id=DimensionId.MATCH, label="Job match", signals=signals),
        matched_required=sorted(s.name for s in required_hits),
        matched_preferred=sorted(s.name for s in preferred_hits),
        gaps=gaps,
        missing_keywords=missing_keywords,
        overlapping_terms=top_overlapping_terms(resume_text, job.raw_text),
        surplus_skills=sorted(owned - job.skill_names())[:20],
    )


def _partition(
    requirements: list[SkillRequirement], owned: set[str]
) -> tuple[list[SkillRequirement], list[SkillRequirement]]:
    hits = [r for r in requirements if r.name in owned]
    misses = [r for r in requirements if r.name not in owned]
    return hits, misses


def _required_signal(
    requirements: list[SkillRequirement],
    hits: list[SkillRequirement],
    misses: list[SkillRequirement],
    resume_skills: Mapping[str, SkillMention],
) -> Signal:
    """Weighted coverage of must-have skills - the heaviest signal by far."""
    if not requirements:
        return make_signal(
            "match.required_skills", "Required skills covered", 0.75, weight=5.0,
            detail="The posting lists no clearly-required skills; scored neutrally.",
            recommendation=(
                "This posting has no explicit requirements section. Mirror its "
                "responsibilities language in your bullets instead."
            ),
        )

    total_weight = sum(r.weight for r in requirements)
    # Credit is scaled by how convincingly the resume evidences the skill: a
    # keyword in a skills list counts for less than one demonstrated in a bullet.
    earned = sum(
        r.weight * _evidence_multiplier(resume_skills.get(r.name))
        for r in hits
    )
    score = earned / total_weight if total_weight else 0.0

    listed_only = [
        r.name for r in hits
        if (m := resume_skills.get(r.name)) and not m.is_demonstrated
    ]
    evidence = [f"Missing required skill: {r.name}" for r in misses[:6]]
    evidence += [f"Listed but never demonstrated: {name}" for name in listed_only[:3]]

    return make_signal(
        "match.required_skills", "Required skills covered", score, weight=5.0,
        detail=f"{len(hits)} of {len(requirements)} required skills evidenced.",
        evidence=evidence,
        recommendation=(
            "Add the missing required skills where you genuinely have them, and move "
            "keyword-only skills into a bullet that shows how you used them."
        ),
        severity=Severity.CRITICAL if score < 0.5 else None,
    )


def _evidence_multiplier(mention: SkillMention | None) -> float:
    if mention is None:
        return 0.0
    if mention.is_demonstrated:
        return 1.0
    # Present in a skills list only.
    return 0.75


def _preferred_signal(requirements: list[SkillRequirement], hits: list[SkillRequirement]) -> Signal:
    if not requirements:
        return make_signal(
            "match.preferred_skills", "Preferred skills covered", 1.0, weight=1.5,
            detail="No preferred skills listed in the posting.",
        )
    score = len(hits) / len(requirements)
    missing = [r.name for r in requirements if r not in hits]
    return make_signal(
        "match.preferred_skills", "Preferred skills covered", score, weight=1.5,
        detail=f"{len(hits)} of {len(requirements)} preferred skills evidenced.",
        evidence=[f"Not evidenced: {name}" for name in missing[:6]],
        recommendation=(
            "Preferred skills are tie-breakers between shortlisted candidates. Add any "
            "you genuinely have, even at a basic level."
        ),
    )


def _keyword_signal(job: ParsedJob, missing: list[str]) -> Signal:
    """Domain keywords outside the taxonomy - the raw ATS filter terms."""
    if not job.keywords:
        return make_signal(
            "match.keywords", "Posting keywords mirrored", 1.0, weight=1.5,
            detail="No recurring domain keywords detected in the posting.",
        )
    covered = len(job.keywords) - len(missing)
    score = covered / len(job.keywords)
    return make_signal(
        "match.keywords", "Posting keywords mirrored", score, weight=1.5,
        detail=f"{covered} of {len(job.keywords)} recurring posting terms appear in the resume.",
        evidence=[f'Absent: "{k}"' for k in missing[:8]],
        recommendation=(
            "Mirror the posting's own vocabulary where it describes work you have "
            "actually done. Keyword filters match literal strings, not synonyms."
        ),
    )


#: Share of a keyword's words that must appear for it to count as covered.
KEYWORD_COVERAGE_THRESHOLD = 0.6
#: Characters compared per word - a crude stem, so "designing" matches "designed".
_STEM_LENGTH = 5


def _missing_keywords(job: ParsedJob, resume_text: str) -> list[str]:
    """Posting keywords whose *concept* is absent from the resume.

    Verbatim phrase matching is the wrong test here: no real resume contains
    "designing event-driven systems" word for word, so requiring it would mark
    every candidate as missing every keyword. Instead a keyword counts as
    covered when most of its words appear somewhere in the resume, compared on
    a short prefix so simple inflections ("designing"/"designed") still match.
    """
    resume_stems = {token[:_STEM_LENGTH] for token in content_tokens(resume_text)}
    missing: list[str] = []
    for keyword in job.keywords:
        words = content_tokens(keyword)
        if not words:
            continue
        hits = sum(1 for word in words if word[:_STEM_LENGTH] in resume_stems)
        if hits / len(words) < KEYWORD_COVERAGE_THRESHOLD:
            missing.append(keyword)
    return missing


def _similarity_signal(value: float) -> Signal:
    """Overall lexical overlap. Deliberately the lightest signal.

    Resume and job-post prose differ in register even for a perfect candidate,
    so a mid-range cosine is normal; the scale is calibrated accordingly.
    """
    score = min(1.0, value / 0.45)
    return make_signal(
        "match.similarity", "Overall language overlap", score, weight=1.0,
        detail=f"TF-IDF cosine similarity {value:.2f}.",
        recommendation=(
            "Rewrite your summary and top bullets in the posting's own terms. This is "
            "the cheapest way to raise every keyword-based score at once."
        ),
    )


def _seniority_signal(resume: ParsedResume, job: ParsedJob) -> Signal:
    if job.seniority is Seniority.UNSPECIFIED:
        return make_signal(
            "match.seniority", "Seniority aligns", 1.0, weight=1.0,
            detail="The posting does not state a seniority level.",
        )

    titles = " ".join(
        filter(None, (e.title for e in resume.experience[:3]))
    ) or resume.section_text(SectionKind.SUMMARY)
    candidate = detect_seniority(titles)
    if candidate is Seniority.UNSPECIFIED:
        return make_signal(
            "match.seniority", "Seniority aligns", 0.7, weight=1.0,
            detail=f"Posting targets {job.seniority.value}; candidate level could not be inferred.",
            recommendation=(
                f"Make your level explicit in your titles or summary - the posting is a "
                f"{job.seniority.value} role."
            ),
        )

    distance = abs(SENIORITY_RANK[candidate] - SENIORITY_RANK[job.seniority])
    score = max(0.0, 1.0 - 0.3 * distance)
    return make_signal(
        "match.seniority", "Seniority aligns", score, weight=1.0,
        detail=f"Posting: {job.seniority.value}; resume reads as {candidate.value}.",
        recommendation=(
            "Frame your summary for the level being hired - scope of ownership, team "
            "size and decision authority are what signal seniority."
        ),
    )


def _experience_signal(resume: ParsedResume, job: ParsedJob) -> Signal:
    required_years = job.min_years_experience
    if required_years is None:
        return make_signal(
            "match.experience", "Years of experience aligns", 1.0, weight=1.0,
            detail="The posting states no minimum experience.",
        )

    actual = resume.years_of_experience
    if actual <= 0:
        return make_signal(
            "match.experience", "Years of experience aligns", 0.4, weight=1.0,
            detail=f"Posting asks for {required_years}+ years; none could be computed from dates.",
            recommendation=(
                "Add explicit date ranges to every role so your total experience can be "
                "computed - both by an ATS and by a reviewer."
            ),
        )

    if actual >= required_years:
        return make_signal(
            "match.experience", "Years of experience aligns", 1.0, weight=1.0,
            detail=f"{actual} years vs {required_years}+ required.",
        )

    shortfall = required_years - actual
    score = max(0.0, 1.0 - shortfall / max(required_years, 1))
    return make_signal(
        "match.experience", "Years of experience aligns", score, weight=1.0,
        detail=f"{actual} years vs {required_years}+ required (short by {shortfall:.1f}).",
        recommendation=(
            "You are under the stated bar. Lead with depth over duration: scope, "
            "ownership and measurable outcomes. Stated minimums are frequently flexible."
        ),
    )
