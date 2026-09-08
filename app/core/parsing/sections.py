"""Section segmentation.

A resume is a sequence of headed blocks, but the headings are free text and the
formatting cues (bold, size) are gone by the time we see plain text. We
therefore identify headings structurally - short, standalone, title-cased or
upper-cased lines - and then map them onto a canonical vocabulary.
"""

from __future__ import annotations

import re

from app.core.parsing.model import Section, SectionKind

# Ordered longest-first within each kind so "work experience" wins over "work".
_HEADING_VOCAB: dict[SectionKind, tuple[str, ...]] = {
    SectionKind.SUMMARY: (
        "professional summary", "career summary", "executive summary", "summary of qualifications",
        "profile summary", "about me", "objective", "career objective", "summary", "profile",
        "overview", "highlights",
    ),
    SectionKind.EXPERIENCE: (
        "professional experience", "work experience", "employment history", "relevant experience",
        "industry experience", "career history", "work history", "experience", "employment",
        "professional background",
    ),
    SectionKind.EDUCATION: (
        "education and training", "academic background", "education", "academics",
        "academic qualifications", "qualifications",
    ),
    SectionKind.SKILLS: (
        "technical skills", "core competencies", "key skills", "skills and abilities",
        "areas of expertise", "technologies", "technical proficiencies", "tech stack",
        "competencies", "skills", "expertise", "tools",
    ),
    SectionKind.PROJECTS: (
        "personal projects", "selected projects", "side projects", "key projects",
        "projects", "portfolio",
    ),
    SectionKind.CERTIFICATIONS: (
        "certifications and licenses", "licenses and certifications", "certifications",
        "certificates", "licenses", "accreditations",
    ),
    SectionKind.PUBLICATIONS: ("publications", "papers", "research", "patents"),
    SectionKind.AWARDS: ("awards and honors", "honors and awards", "awards", "achievements", "honors"),
    SectionKind.VOLUNTEER: ("volunteer experience", "community involvement", "volunteering", "volunteer"),
    SectionKind.LANGUAGES: ("languages",),
    SectionKind.INTERESTS: ("interests", "hobbies", "activities"),
    SectionKind.REFERENCES: ("references",),
    SectionKind.CONTACT: ("contact", "contact information", "personal details", "personal information"),
}

# Flat lookup built once: phrase -> kind, longest phrases first.
_PHRASE_TO_KIND: list[tuple[str, SectionKind]] = sorted(
    ((phrase, kind) for kind, phrases in _HEADING_VOCAB.items() for phrase in phrases),
    key=lambda pair: -len(pair[0]),
)

MAX_HEADING_WORDS = 5
MAX_HEADING_CHARS = 60
_DECORATION_RE = re.compile(r"^[\s\-_=*#|~<>\[\]•]+|[\s\-_=*#|~<>\[\]:•]+$")
_SENTENCE_END_RE = re.compile(r"[.!?,;]$")


def normalise_heading(line: str) -> str:
    """Strip decoration and punctuation so headings compare cleanly."""
    return _DECORATION_RE.sub("", line).strip().lower()


#: A loose match must cover at least this share of the line, so a body line that
#: merely mentions a vocabulary word is not mistaken for a heading.
LOOSE_MATCH_COVERAGE = 0.5


def classify_heading(line: str) -> SectionKind | None:
    """Map a candidate heading line onto a canonical section, if it is one."""
    text = normalise_heading(line)
    if not text:
        return None
    for phrase, kind in _PHRASE_TO_KIND:
        if text == phrase:
            return kind

    # Decorated forms such as "TECHNICAL SKILLS & TOOLS" should still match, but
    # only for lines that structurally look like headings and where the matched
    # phrase dominates the line. Without both guards, an ordinary body line -
    # "Senior Engineer | Razorline Technologies | Mar 2021 - Present" - matches
    # the skills vocabulary on the word "technologies" and silently truncates
    # the section above it.
    if not looks_like_heading(line):
        return None
    for phrase, kind in _PHRASE_TO_KIND:
        if len(phrase) < 6 or not re.search(rf"\b{re.escape(phrase)}\b", text):
            continue
        if len(phrase) / len(text) >= LOOSE_MATCH_COVERAGE:
            return kind
    return None


def looks_like_heading(line: str) -> bool:
    """Structural test, independent of the vocabulary.

    Used to close the previous section when a resume uses a heading we do not
    recognise - otherwise an unknown heading swallows the rest of the document.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    core = _DECORATION_RE.sub("", stripped)
    if not core or _SENTENCE_END_RE.search(core):
        return False
    words = core.split()
    if not (1 <= len(words) <= MAX_HEADING_WORDS):
        return False
    if any(ch.isdigit() for ch in core) and not core.isupper():
        return False
    letters = [c for c in core if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio > 0.85:
        return True
    # Title Case With Every Word Capitalised.
    return all(w[0].isupper() for w in words if w[0].isalpha())


def segment(text: str) -> tuple[dict[SectionKind, Section], list[str]]:
    """Split a resume into canonical sections.

    Returns the section map and the list of headings that looked structural but
    matched no known vocabulary (surfaced to the user as ATS naming advice).
    """
    raw_lines = text.split("\n")
    found: dict[SectionKind, Section] = {}
    unknown: list[str] = []

    # Locate every heading first, then slice between them.
    first_content_line = next(
        (i for i, line in enumerate(raw_lines) if line.strip()), len(raw_lines)
    )

    marks: list[tuple[int, SectionKind | None, str]] = []
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        kind = classify_heading(line)
        if kind is not None:
            marks.append((index, kind, line.strip()))
        elif (
            # The first line of a resume is the candidate's name, which looks
            # exactly like a heading. Treating it as one would discard the
            # whole contact block that follows it.
            index != first_content_line
            and looks_like_heading(line)
            and _is_isolated(raw_lines, index)
        ):
            marks.append((index, None, line.strip()))

    # Everything above the first heading is the contact/header block.
    first_mark = marks[0][0] if marks else len(raw_lines)
    header_body = "\n".join(raw_lines[:first_mark]).strip()
    if header_body:
        found[SectionKind.CONTACT] = Section(
            kind=SectionKind.CONTACT,
            heading="(header)",
            body=header_body,
            start_line=0,
            end_line=first_mark,
        )

    for position, (index, kind, heading) in enumerate(marks):
        end = marks[position + 1][0] if position + 1 < len(marks) else len(raw_lines)
        body = "\n".join(raw_lines[index + 1 : end]).strip()
        if not body:
            continue
        if kind is None:
            # The heading is unrecognised, but its content is still the
            # candidate's resume. Keeping it under OTHER means skills and
            # writing quality inside creatively-named sections are still
            # analysed, instead of vanishing from every downstream check.
            unknown.append(heading)
            _merge(found, SectionKind.OTHER, heading, body, index, end)
            continue
        _merge(found, kind, heading, body, index, end)

    return found, unknown


def _merge(
    found: dict[SectionKind, Section],
    kind: SectionKind,
    heading: str,
    body: str,
    start: int,
    end: int,
) -> None:
    """Add a section, merging into any existing one of the same kind.

    Resumes legitimately repeat headings (an "Experience" block per employer),
    so the later occurrence must extend the section rather than replace it.
    """
    existing = found.get(kind)
    if existing is None:
        found[kind] = Section(
            kind=kind, heading=heading, body=body, start_line=start, end_line=end
        )
        return
    found[kind] = Section(
        kind=kind,
        heading=existing.heading,
        body=f"{existing.body}\n{body}".strip(),
        start_line=existing.start_line,
        end_line=end,
    )


def _is_isolated(raw_lines: list[str], index: int) -> bool:
    """A heading sits on its own line, with a blank line or nothing before it."""
    if index == 0:
        return True
    return not raw_lines[index - 1].strip()
