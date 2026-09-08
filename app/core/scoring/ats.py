"""ATS compatibility scoring.

Every check here corresponds to a documented way applicant tracking systems
mangle or reject a resume: unreadable text layers, multi-column layouts that
interleave, tables whose cells are read out of order, contact details stranded
in a header region, headings the parser cannot map to a known section, and dates
in formats the parser cannot normalise.

Two dimensions come out of this module. `score_ats` covers machine
*readability*; `score_structure` covers whether the expected content is present
and proportionate. They are separated because the fixes are completely
different - one is "export differently", the other is "write the missing
section".
"""

from __future__ import annotations

import re

from app.core.extraction.base import ExtractedDocument, SourceFormat
from app.core.parsing.entities import DATE_RANGE_RE
from app.core.parsing.model import CORE_SECTIONS, ParsedResume, SectionKind
from app.core.scoring.base import DimensionId, DimensionScore, Severity, Signal, make_signal

#: Resume length guidance in words. Below the floor reads as thin; above the
#: ceiling, recruiters skim and ATS relevance scoring dilutes.
MIN_WORDS = 250
IDEAL_MIN_WORDS = 400
IDEAL_MAX_WORDS = 1000
MAX_WORDS = 1400

#: Characters of extractable text per page for a healthy text layer.
HEALTHY_CHARS_PER_PAGE = 1200

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
#: Replacement chars and C0 controls: the fingerprint of a damaged text layer.
_JUNK_CHARS = chr(0xFFFD) + chr(0) + chr(1) + chr(2) + chr(3) + chr(4) + chr(5)
_CONTROL_JUNK_RE = re.compile('[' + re.escape(_JUNK_CHARS) + ']')


def score_ats(document: ExtractedDocument, resume: ParsedResume) -> DimensionScore:
    """Machine readability of the file as submitted."""
    signals: list[Signal] = [
        _text_layer(document),
        _file_format(document),
        _layout(document),
        _tables(document),
        _headers_footers(document),
        _contact_reachable(resume),
        _headings_recognised(resume),
        _date_formats(resume),
        _encoding_health(document),
        _bullet_glyphs(document, resume),
    ]
    return DimensionScore(id=DimensionId.ATS, label="ATS compatibility", signals=signals)


def score_structure(document: ExtractedDocument, resume: ParsedResume) -> DimensionScore:
    """Presence, order and proportion of the expected content."""
    signals: list[Signal] = [
        _core_sections(resume),
        _length(document, resume),
        _section_order(resume),
        _experience_detail(resume),
        _education_present(resume),
        _summary_present(resume),
        _online_presence(resume),
    ]
    return DimensionScore(id=DimensionId.STRUCTURE, label="Structure & completeness", signals=signals)


# --------------------------------------------------------------------------
# ATS signals
# --------------------------------------------------------------------------


def _text_layer(document: ExtractedDocument) -> Signal:
    """Is there a real, selectable text layer on every page?"""
    if document.source_format is not SourceFormat.PDF:
        return make_signal(
            "ats.text_layer", "Text is machine-readable", 1.0, weight=3.0,
            detail=f"{document.source_format.upper()} files expose text directly.",
        )

    dead = document.pages_without_text
    density = document.chars_per_page
    if dead:
        score = max(0.0, 1.0 - dead / max(document.page_count, 1))
        return make_signal(
            "ats.text_layer", "Text is machine-readable", score, weight=3.0,
            evidence=[f"{dead} of {document.page_count} page(s) contain almost no extractable text."],
            recommendation=(
                "Re-export the PDF from the source document rather than scanning or "
                "screenshotting it. An ATS reads a scanned page as blank."
            ),
        )

    score = min(1.0, density / HEALTHY_CHARS_PER_PAGE)
    return make_signal(
        "ats.text_layer", "Text is machine-readable", score, weight=3.0,
        detail=f"{density:.0f} characters of text per page.",
        recommendation=(
            "Text density is low for a resume page - check that nothing important "
            "is rendered as an image or a graphic."
        ),
    )


def _file_format(document: ExtractedDocument) -> Signal:
    """PDF and DOCX both parse well; anything else is a compromise."""
    if document.source_format is SourceFormat.PDF:
        if document.is_encrypted:
            return make_signal(
                "ats.file_format", "File format parses reliably", 0.6, weight=1.5,
                evidence=["PDF carries encryption or permission restrictions."],
                recommendation=(
                    "Remove PDF password/permission protection before applying - some "
                    "parsers refuse restricted files outright."
                ),
            )
        return make_signal("ats.file_format", "File format parses reliably", 1.0, weight=1.5)
    if document.source_format is SourceFormat.DOCX:
        return make_signal(
            "ats.file_format", "File format parses reliably", 1.0, weight=1.5,
            detail="DOCX is the most reliably parsed format across ATS vendors.",
        )
    return make_signal(
        "ats.file_format", "File format parses reliably", 0.7, weight=1.5,
        evidence=["Plain text submitted."],
        recommendation="Submit a PDF or DOCX; plain text loses all visual hierarchy for the human reviewer.",
    )


def _layout(document: ExtractedDocument) -> Signal:
    if document.has_multi_column_hint:
        return make_signal(
            "ats.layout", "Single-column layout", 0.25, weight=2.5,
            evidence=["Line-length distribution indicates a multi-column or sidebar layout."],
            recommendation=(
                "Convert to a single-column layout. Parsers read left-to-right across "
                "the full page width and interleave columns into nonsense."
            ),
        )
    return make_signal("ats.layout", "Single-column layout", 1.0, weight=2.5)


def _tables(document: ExtractedDocument) -> Signal:
    if not document.has_tables:
        return make_signal("ats.tables", "No tables used for layout", 1.0, weight=1.5)
    score = 0.5 if document.table_count <= 2 else 0.2
    return make_signal(
        "ats.tables", "No tables used for layout", score, weight=1.5,
        evidence=[f"{document.table_count} table(s) detected."],
        recommendation=(
            "Replace tables with plain paragraphs and bullet lists. Many parsers read "
            "table cells column-first, scrambling the order of your content."
        ),
    )


def _headers_footers(document: ExtractedDocument) -> Signal:
    if not document.has_text_in_headers_footers:
        return make_signal("ats.header_footer", "No content stranded in header/footer", 1.0, weight=2.0)
    return make_signal(
        "ats.header_footer", "No content stranded in header/footer", 0.3, weight=2.0,
        evidence=["Text found in the document header or footer region."],
        recommendation=(
            "Move contact details into the body of the first page. Header and footer "
            "regions are discarded by a large share of ATS parsers."
        ),
    )


def _contact_reachable(resume: ParsedResume) -> Signal:
    contact = resume.contact
    present = [
        name
        for name, value in (
            ("name", contact.name), ("email", contact.email), ("phone", contact.phone),
            ("location", contact.location),
        )
        if value
    ]
    missing = [n for n in ("name", "email", "phone") if n not in present]

    if not contact.has_minimum:
        return make_signal(
            "ats.contact", "Contact details are parseable", 0.0, weight=3.0,
            severity=Severity.CRITICAL,
            evidence=["Neither an email address nor a phone number could be extracted."],
            recommendation=(
                "Put a plain-text email address and phone number at the top of page one. "
                "Without them the application cannot be actioned even if it passes screening."
            ),
        )

    score = len(present) / 4
    return make_signal(
        "ats.contact", "Contact details are parseable", max(score, 0.5), weight=3.0,
        detail=f"Extracted: {', '.join(present)}.",
        recommendation=(
            f"Add your {' and '.join(missing)} as plain text near the top."
            if missing
            else "Add a city and country line - many ATS filters sort by location."
        ),
    )


def _headings_recognised(resume: ParsedResume) -> Signal:
    """Creative section names break the parser's section mapping."""
    known = len(resume.sections)
    unknown = resume.unknown_headings
    if not unknown:
        return make_signal("ats.headings", "Section headings use standard names", 1.0, weight=2.0)
    score = max(0.3, known / (known + len(unknown)))
    return make_signal(
        "ats.headings", "Section headings use standard names", score, weight=2.0,
        evidence=[f'Unrecognised heading: "{h}"' for h in unknown[:5]],
        recommendation=(
            "Rename creative headings to the conventional labels an ATS looks for: "
            "Experience, Education, Skills, Projects, Certifications."
        ),
    )


def _date_formats(resume: ParsedResume) -> Signal:
    """Roles without a parseable date range lose their tenure entirely."""
    if not resume.experience:
        return make_signal(
            "ats.dates", "Employment dates are parseable", 0.4, weight=2.0,
            evidence=["No dated roles were detected."],
            recommendation=(
                "Give every role an explicit date range in a standard format, "
                'e.g. "Mar 2021 - Present".'
            ),
        )
    dated = sum(1 for entry in resume.experience if entry.dates and entry.dates.start_year)
    score = dated / len(resume.experience)
    undated = [e.title or e.organization or "(untitled role)"
               for e in resume.experience if not (e.dates and e.dates.start_year)]
    return make_signal(
        "ats.dates", "Employment dates are parseable", score, weight=2.0,
        evidence=[f"No parseable dates on: {name}" for name in undated[:4]],
        recommendation=(
            'Use a consistent "Mon YYYY - Mon YYYY" format on every role. Parsers that '
            "cannot read a date treat the role as having zero duration."
        ),
    )


def _encoding_health(document: ExtractedDocument) -> Signal:
    """Replacement characters and glyph soup mean the text layer is damaged."""
    text = document.text
    if not text:
        return make_signal("ats.encoding", "Characters encode cleanly", 0.0, weight=1.0)
    junk = len(_CONTROL_JUNK_RE.findall(text))
    non_ascii_ratio = len(_NON_ASCII_RE.findall(text)) / len(text)

    if junk > 5:
        return make_signal(
            "ats.encoding", "Characters encode cleanly", 0.2, weight=1.0,
            evidence=[f"{junk} unreadable/replacement characters in the extracted text."],
            recommendation=(
                "The PDF's font encoding is damaged - text copies out as garbage. "
                "Re-export using a standard font (Arial, Calibri, Georgia)."
            ),
        )
    # Accented names and non-Latin scripts are legitimate; only a very high
    # ratio suggests symbol fonts or decorative glyphs.
    if non_ascii_ratio > 0.25:
        return make_signal(
            "ats.encoding", "Characters encode cleanly", 0.6, weight=1.0,
            detail=f"{non_ascii_ratio:.0%} of characters are non-ASCII.",
            recommendation=(
                "Check that decorative icons and symbol fonts are not being used for "
                "content - they extract as meaningless characters."
            ),
        )
    return make_signal("ats.encoding", "Characters encode cleanly", 1.0, weight=1.0)


def _bullet_glyphs(document: ExtractedDocument, resume: ParsedResume) -> Signal:
    """Bullets should be a real list, not a manual glyph, and should exist."""
    if resume.bullets:
        return make_signal("ats.bullets", "Achievements use bullet lists", 1.0, weight=1.0)
    if document.word_count < MIN_WORDS:
        return make_signal(
            "ats.bullets", "Achievements use bullet lists", 0.5, weight=1.0,
            recommendation="Break your experience into bullet points - dense paragraphs are skimmed past.",
        )
    return make_signal(
        "ats.bullets", "Achievements use bullet lists", 0.3, weight=1.0,
        evidence=["No bullet points detected; experience appears to be written as prose."],
        recommendation=(
            "Convert experience paragraphs into 3-6 bullets per role. Recruiters spend "
            "seconds per resume and read bullets, not paragraphs."
        ),
    )


# --------------------------------------------------------------------------
# Structure signals
# --------------------------------------------------------------------------


def _core_sections(resume: ParsedResume) -> Signal:
    missing = resume.missing_core_sections
    score = (len(CORE_SECTIONS) - len(missing)) / len(CORE_SECTIONS)
    if not missing:
        return make_signal("structure.core_sections", "Core sections present", 1.0, weight=3.0)
    return make_signal(
        "structure.core_sections", "Core sections present", score, weight=3.0,
        evidence=[f"Missing section: {kind.value}" for kind in missing],
        recommendation=(
            "Add the missing section(s): "
            + ", ".join(k.value.title() for k in missing)
            + ". ATS filters query these by name."
        ),
    )


def _length(document: ExtractedDocument, resume: ParsedResume) -> Signal:
    words = max(document.word_count, resume.word_count)
    if IDEAL_MIN_WORDS <= words <= IDEAL_MAX_WORDS:
        return make_signal(
            "structure.length", "Length is appropriate", 1.0, weight=2.0,
            detail=f"{words} words across ~{document.page_count} page(s).",
        )
    if words < MIN_WORDS:
        return make_signal(
            "structure.length", "Length is appropriate", 0.2, weight=2.0,
            evidence=[f"Only {words} words."],
            recommendation=(
                f"The resume is very short ({words} words). Expand each role to 3-5 "
                "achievement bullets with concrete outcomes."
            ),
        )
    if words < IDEAL_MIN_WORDS:
        score = 0.5 + 0.5 * (words - MIN_WORDS) / (IDEAL_MIN_WORDS - MIN_WORDS)
        return make_signal(
            "structure.length", "Length is appropriate", score, weight=2.0,
            evidence=[f"{words} words - on the thin side."],
            recommendation="Add detail to your most recent role; aim for 400-1000 words overall.",
        )
    if words <= MAX_WORDS:
        score = 1.0 - 0.4 * (words - IDEAL_MAX_WORDS) / (MAX_WORDS - IDEAL_MAX_WORDS)
        return make_signal(
            "structure.length", "Length is appropriate", score, weight=2.0,
            evidence=[f"{words} words - slightly long."],
            recommendation="Trim older roles to two bullets each and cut anything over ten years old.",
        )
    return make_signal(
        "structure.length", "Length is appropriate", 0.3, weight=2.0,
        evidence=[f"{words} words across ~{document.page_count} pages."],
        recommendation=(
            "Cut to two pages. Beyond that, relevance dilutes and reviewers stop reading; "
            "keep only the last 10-12 years in detail."
        ),
    )


def _section_order(resume: ParsedResume) -> Signal:
    """Experience should precede education for anyone past their first job."""
    experience = resume.sections.get(SectionKind.EXPERIENCE)
    education = resume.sections.get(SectionKind.EDUCATION)
    if not experience or not education:
        return make_signal("structure.order", "Sections are ordered sensibly", 1.0, weight=1.0)

    experienced = resume.total_experience_months >= 24
    education_first = education.start_line < experience.start_line
    if experienced and education_first:
        return make_signal(
            "structure.order", "Sections are ordered sensibly", 0.5, weight=1.0,
            evidence=["Education appears above Experience."],
            recommendation=(
                f"With ~{resume.years_of_experience} years of experience, lead with "
                "Experience and move Education below it."
            ),
        )
    return make_signal("structure.order", "Sections are ordered sensibly", 1.0, weight=1.0)


def _experience_detail(resume: ParsedResume) -> Signal:
    """Each role needs enough bullets to be evaluable."""
    if not resume.experience:
        return make_signal(
            "structure.experience_detail", "Roles are described in enough detail", 0.0, weight=2.5,
            severity=Severity.CRITICAL,
            evidence=["No work experience entries could be identified."],
            recommendation=(
                'Add an "Experience" section listing each role with a title, employer, '
                "dates and 3-5 bullets."
            ),
        )
    thin = [
        entry.title or entry.organization or "(untitled role)"
        for entry in resume.experience
        if len(entry.bullets) < 2
    ]
    score = 1.0 - len(thin) / len(resume.experience)
    return make_signal(
        "structure.experience_detail", "Roles are described in enough detail", score, weight=2.5,
        evidence=[f"Fewer than 2 bullets: {name}" for name in thin[:4]],
        recommendation=(
            "Give every role at least two achievement bullets. A role with a title and "
            "no substance reads as filler."
        ),
    )


def _education_present(resume: ParsedResume) -> Signal:
    if resume.education:
        return make_signal("structure.education", "Education is listed", 1.0, weight=1.0)
    if SectionKind.EDUCATION in resume.sections:
        return make_signal(
            "structure.education", "Education is listed", 0.6, weight=1.0,
            evidence=["An education section exists but no degree could be parsed from it."],
            recommendation='State the qualification explicitly, e.g. "BSc Computer Science, University of X, 2019".',
        )
    return make_signal(
        "structure.education", "Education is listed", 0.3, weight=1.0,
        recommendation=(
            "Add an Education section. Even without a degree, list the highest "
            "qualification or relevant training - many ATS filters require the field."
        ),
    )


def _summary_present(resume: ParsedResume) -> Signal:
    summary = resume.sections.get(SectionKind.SUMMARY)
    if not summary:
        return make_signal(
            "structure.summary", "Opening summary present", 0.5, weight=1.0,
            recommendation=(
                "Add a 2-3 line professional summary at the top naming your role, years "
                "of experience and specialism. It frames everything the reviewer reads next."
            ),
        )
    words = summary.word_count
    if words > 120:
        return make_signal(
            "structure.summary", "Opening summary present", 0.6, weight=1.0,
            evidence=[f"Summary is {words} words."],
            recommendation="Cut the summary to 2-3 lines - long summaries get skipped entirely.",
        )
    if words < 15:
        return make_signal(
            "structure.summary", "Opening summary present", 0.7, weight=1.0,
            recommendation="Expand the summary slightly - name your specialism and years of experience.",
        )
    return make_signal("structure.summary", "Opening summary present", 1.0, weight=1.0)


def _online_presence(resume: ParsedResume) -> Signal:
    contact = resume.contact
    links = [x for x in (contact.linkedin, contact.github, contact.website) if x]
    if links:
        return make_signal(
            "structure.links", "Professional links included", 1.0, weight=1.0,
            detail=f"{len(links)} link(s) found.",
        )
    return make_signal(
        "structure.links", "Professional links included", 0.4, weight=1.0,
        recommendation=(
            "Add a LinkedIn URL, and a GitHub or portfolio link if your field has one. "
            "Recruiters look for them and their absence is noticed."
        ),
    )


def has_parseable_dates(text: str) -> bool:
    """Utility used by tests and the report layer."""
    return bool(DATE_RANGE_RE.search(text))
