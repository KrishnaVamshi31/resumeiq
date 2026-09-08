"""The parsing entry point: raw text in, `ParsedResume` out."""

from __future__ import annotations

from app.core.parsing.contact import extract_contact
from app.core.parsing.entities import (
    extract_education,
    extract_experience,
    total_experience_months,
)
from app.core.parsing.model import ParsedResume, SectionKind
from app.core.parsing.normalize import clean_text, is_bullet, strip_bullet
from app.core.parsing.sections import segment
from app.observability import track


def parse_resume(raw_text: str) -> ParsedResume:
    """Turn extracted text into the structured form every scorer consumes."""
    with track("parse") as ctx:
        text = clean_text(raw_text)
        sections, unknown_headings = segment(text)
        contact = extract_contact(text)

        experience_body = sections[SectionKind.EXPERIENCE].body if SectionKind.EXPERIENCE in sections else ""
        education_body = sections[SectionKind.EDUCATION].body if SectionKind.EDUCATION in sections else ""

        experience = extract_experience(experience_body) if experience_body else []
        education = extract_education(education_body) if education_body else []

        # Bullets from anywhere in the document: projects and volunteering carry
        # achievements too, and content quality is judged across all of them.
        bullets = [strip_bullet(ln) for ln in text.split("\n") if is_bullet(ln)]

        parsed = ParsedResume(
            raw_text=text,
            contact=contact,
            sections=sections,
            unknown_headings=unknown_headings,
            experience=experience,
            education=education,
            bullets=[b for b in bullets if b],
            total_experience_months=total_experience_months(experience),
        )

        ctx["sections"] = len(sections)
        ctx["experience_entries"] = len(experience)
        ctx["bullets"] = len(parsed.bullets)
    return parsed
