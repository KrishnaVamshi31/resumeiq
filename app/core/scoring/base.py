"""Shared scoring vocabulary.

Every check in the system produces a `Signal`: a normalised 0-1 score, a weight,
the evidence that produced it, and - when it did not score full marks - a
concrete recommendation. Dimensions are weighted means of signals, and the
overall score is a weighted mean of dimensions. Nothing anywhere returns a
number without the signals that explain it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class Severity(StrEnum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DimensionId(StrEnum):
    ATS = "ats"
    CONTENT = "content"
    STRUCTURE = "structure"
    MATCH = "match"


#: Weights when a job description is supplied.
WEIGHTS_WITH_JOB: dict[DimensionId, float] = {
    DimensionId.MATCH: 0.35,
    DimensionId.ATS: 0.25,
    DimensionId.CONTENT: 0.25,
    DimensionId.STRUCTURE: 0.15,
}

#: Weights for a standalone resume review, renormalised over three dimensions.
WEIGHTS_WITHOUT_JOB: dict[DimensionId, float] = {
    DimensionId.CONTENT: 0.40,
    DimensionId.ATS: 0.35,
    DimensionId.STRUCTURE: 0.25,
}


def severity_for(score: float, *, critical_below: float = 0.34, warning_below: float = 0.67,
                 info_below: float = 0.95) -> Severity:
    """Map a 0-1 signal score onto a severity band."""
    if score < critical_below:
        return Severity.CRITICAL
    if score < warning_below:
        return Severity.WARNING
    if score < info_below:
        return Severity.INFO
    return Severity.OK


@dataclass(slots=True)
class Signal:
    """One atomic, explainable check."""

    id: str
    label: str
    score: float
    weight: float = 1.0
    severity: Severity = Severity.OK
    evidence: list[str] = field(default_factory=list)
    recommendation: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        self.score = round(max(0.0, min(1.0, self.score)), 4)

    @property
    def points_lost(self) -> float:
        """Weighted shortfall - the basis for ranking recommendations."""
        return round((1.0 - self.score) * self.weight, 4)


def make_signal(
    signal_id: str,
    label: str,
    score: float,
    *,
    weight: float = 1.0,
    evidence: list[str] | None = None,
    recommendation: str | None = None,
    detail: str | None = None,
    severity: Severity | None = None,
) -> Signal:
    """Build a signal, deriving severity from the score unless told otherwise."""
    score = max(0.0, min(1.0, score))
    return Signal(
        id=signal_id,
        label=label,
        score=score,
        weight=weight,
        severity=severity or severity_for(score),
        evidence=evidence or [],
        recommendation=recommendation if score < 0.95 else None,
        detail=detail,
    )


@dataclass(slots=True)
class DimensionScore:
    """A weighted group of signals, reported on a 0-100 scale."""

    id: DimensionId
    label: str
    signals: list[Signal] = field(default_factory=list)
    #: Set when the dimension is not applicable (e.g. match with no job description).
    applicable: bool = True

    @property
    def score(self) -> float:
        total_weight = sum(s.weight for s in self.signals)
        if not total_weight:
            return 0.0
        weighted = sum(s.score * s.weight for s in self.signals)
        return round(100.0 * weighted / total_weight, 1)

    @property
    def severity(self) -> Severity:
        """The worst severity present, so a single critical defect is visible."""
        order = [Severity.CRITICAL, Severity.WARNING, Severity.INFO, Severity.OK]
        for level in order:
            if any(s.severity is level for s in self.signals):
                return level
        return Severity.OK

    def ranked_issues(self, limit: int | None = None) -> list[Signal]:
        """Signals that cost the most points, worst first."""
        issues = [s for s in self.signals if s.score < 0.95 and s.recommendation]
        issues.sort(key=lambda s: (-s.points_lost, s.id))
        return issues[:limit] if limit else issues


def band(score: float) -> Literal["excellent", "strong", "fair", "weak", "poor"]:
    """Human label for a 0-100 score. Used in reports and the UI."""
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "strong"
    if score >= 55:
        return "fair"
    if score >= 40:
        return "weak"
    return "poor"
