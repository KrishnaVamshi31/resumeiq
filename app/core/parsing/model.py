"""The structured resume representation produced by the parsing layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SectionKind(StrEnum):
    """Canonical section identities. Everything else lands in ``OTHER``."""

    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    PUBLICATIONS = "publications"
    AWARDS = "awards"
    VOLUNTEER = "volunteer"
    LANGUAGES = "languages"
    INTERESTS = "interests"
    REFERENCES = "references"
    OTHER = "other"


#: Sections an employer expects to find. Their absence is a scored defect.
CORE_SECTIONS: frozenset[SectionKind] = frozenset(
    {SectionKind.EXPERIENCE, SectionKind.EDUCATION, SectionKind.SKILLS}
)


@dataclass(slots=True)
class Section:
    kind: SectionKind
    heading: str
    body: str
    start_line: int
    end_line: int

    @property
    def word_count(self) -> int:
        return len(self.body.split())


@dataclass(slots=True)
class ContactInfo:
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin: str | None = None
    github: str | None = None
    website: str | None = None
    other_links: list[str] = field(default_factory=list)

    @property
    def has_minimum(self) -> bool:
        """An employer needs at minimum a way to reply."""
        return bool(self.email or self.phone)


@dataclass(slots=True)
class DateRange:
    start_year: int | None = None
    start_month: int | None = None
    end_year: int | None = None
    end_month: int | None = None
    is_current: bool = False
    raw: str = ""

    @property
    def months(self) -> int | None:
        """Duration in months, or None when the range cannot be resolved."""
        if self.start_year is None:
            return None
        end_year, end_month = self.end_year, self.end_month
        if self.is_current or end_year is None:
            from datetime import date

            today = date.today()
            end_year, end_month = today.year, today.month
        start_month = self.start_month or 1
        end_month = end_month or 12
        span = (end_year - self.start_year) * 12 + (end_month - start_month) + 1
        return max(span, 0)


@dataclass(slots=True)
class ExperienceEntry:
    title: str | None = None
    organization: str | None = None
    dates: DateRange | None = None
    location: str | None = None
    bullets: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass(slots=True)
class EducationEntry:
    degree: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    dates: DateRange | None = None
    gpa: float | None = None
    raw: str = ""


@dataclass(slots=True)
class ParsedResume:
    """Everything the scorers need, derived once from the raw text."""

    raw_text: str
    contact: ContactInfo
    sections: dict[SectionKind, Section] = field(default_factory=dict)
    unknown_headings: list[str] = field(default_factory=list)
    experience: list[ExperienceEntry] = field(default_factory=list)
    education: list[EducationEntry] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    total_experience_months: int = 0

    @property
    def word_count(self) -> int:
        return len(self.raw_text.split())

    @property
    def years_of_experience(self) -> float:
        return round(self.total_experience_months / 12, 1)

    @property
    def missing_core_sections(self) -> list[SectionKind]:
        return sorted(CORE_SECTIONS - set(self.sections), key=lambda s: s.value)

    def section_text(self, kind: SectionKind) -> str:
        section = self.sections.get(kind)
        return section.body if section else ""
