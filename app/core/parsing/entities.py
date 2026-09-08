"""Date, experience and education extraction from resume sections."""

from __future__ import annotations

import re
from datetime import date

from app.core.parsing.model import DateRange, EducationEntry, ExperienceEntry
from app.core.parsing.normalize import is_bullet, strip_bullet

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_PRESENT = r"present|current|now|to date|ongoing|till date"
_DASH = r"(?:\s*(?:-|–|—|to|until|through)\s*)"

# "Jan 2020", "January 2020", "01/2020", "2020"
_POINT = rf"(?:(?:{_MONTH_ALT})\.?\s*,?\s*\d{{4}}|\d{{1,2}}[/.\-]\d{{4}}|\d{{4}})"

DATE_RANGE_RE = re.compile(
    rf"\b(?P<start>{_POINT}){_DASH}(?P<end>{_POINT}|{_PRESENT})\b", re.IGNORECASE
)
_SINGLE_YEAR_RE = re.compile(r"\b(?P<year>19[6-9]\d|20[0-4]\d)\b")
GPA_RE = re.compile(
    r"\b(?:gpa|cgpa|grade point average)\b[^0-9]{0,12}(\d\.\d{1,2})(?:\s*/\s*(\d\.?\d?))?", re.I
)

DEGREE_RE = re.compile(
    r"\b(ph\.?\s?d|doctorate|d\.?phil|m\.?b\.?a|m\.?sc?|master(?:'s)?(?:\s+of\s+\w+)?|"
    r"b\.?sc?|b\.?a|b\.?e|b\.?tech|m\.?tech|bachelor(?:'s)?(?:\s+of\s+\w+)?|"
    r"associate(?:'s)?|a\.?a\.?s|diploma|high school|secondary school)\b",
    re.I,
)
_INSTITUTION_RE = re.compile(
    r"\b([\w.'&\- ]*?(?:university|college|institute|institution|academy|school|polytechnic)"
    r"[\w.'&\- ]*)\b",
    re.I,
)
_FIELD_RE = re.compile(
    r"\b(?:in|of)\s+([A-Z][\w&\-]*(?:\s+(?:and\s+)?[A-Z&][\w&\-]*){0,4})", re.I
)

_TITLE_HINTS = (
    "engineer", "developer", "manager", "analyst", "scientist", "designer", "architect",
    "consultant", "director", "lead", "head", "intern", "specialist", "administrator",
    "officer", "associate", "coordinator", "president", "founder", "owner", "researcher",
    "programmer", "technician", "supervisor", "strategist", "recruiter", "accountant",
    "nurse", "teacher", "professor", "attorney", "advisor", "principal", "producer",
)
_ORG_HINTS = (
    "inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "corporation", "company",
    "technologies", "solutions", "systems", "labs", "group", "gmbh", "plc", "pvt",
    "consulting", "partners", "holdings", "bank", "university", "hospital", "agency",
)
_SPLIT_RE = re.compile(r"\s*(?:\||•|·|—|–|\s-\s|,\s|\bat\b|\bfor\b)\s*", re.I)


def parse_date_range(text: str) -> DateRange | None:
    """Parse the first date range found in ``text``."""
    match = DATE_RANGE_RE.search(text)
    if match:
        start_year, start_month = _parse_point(match.group("start"))
        end_raw = match.group("end")
        is_current = bool(re.fullmatch(_PRESENT, end_raw.strip(), re.I))
        end_year, end_month = (None, None) if is_current else _parse_point(end_raw)
        if start_year is None:
            return None
        return DateRange(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            is_current=is_current,
            raw=match.group(0).strip(),
        )

    # A lone year, as education entries often carry ("B.S. Computer Science, 2019").
    years = _SINGLE_YEAR_RE.findall(text)
    if len(years) == 1:
        year = int(years[0])
        return DateRange(start_year=year, end_year=year, raw=years[0])
    return None


def _parse_point(token: str) -> tuple[int | None, int | None]:
    token = token.strip().rstrip(",")
    month_match = re.match(rf"({_MONTH_ALT})\.?\s*,?\s*(\d{{4}})", token, re.I)
    if month_match:
        return int(month_match.group(2)), _MONTHS[month_match.group(1).lower()]
    numeric = re.match(r"(\d{1,2})[/.\-](\d{4})", token)
    if numeric:
        month = int(numeric.group(1))
        return int(numeric.group(2)), month if 1 <= month <= 12 else None
    year_only = re.match(r"(\d{4})", token)
    if year_only:
        return int(year_only.group(1)), None
    return None, None


def extract_experience(body: str) -> list[ExperienceEntry]:
    """Split an experience section into entries with their bullets attached.

    An entry begins at a non-bullet line carrying a date range, or at the line
    immediately above one (the common "Company \\n Title, Jan 2020 - Present"
    two-line header).
    """
    raw_lines = [ln for ln in body.split("\n") if ln.strip()]
    if not raw_lines:
        return []

    starts: list[int] = []
    for index, line in enumerate(raw_lines):
        if is_bullet(line):
            continue
        if not DATE_RANGE_RE.search(line):
            continue
        # Absorb an immediately preceding non-bullet, dateless line as part of
        # this header rather than treating it as the tail of the previous entry.
        anchor = index
        if (
            index > 0
            and not is_bullet(raw_lines[index - 1])
            and not DATE_RANGE_RE.search(raw_lines[index - 1])
            and (not starts or starts[-1] < index - 1)
            and len(raw_lines[index - 1]) < 90
        ):
            anchor = index - 1
        if not starts or anchor > starts[-1]:
            starts.append(anchor)

    if not starts:
        # No dates anywhere: treat the whole section as one undated entry.
        bullets = [strip_bullet(ln) for ln in raw_lines if is_bullet(ln)]
        header = next((ln for ln in raw_lines if not is_bullet(ln)), "")
        title, organization = _split_header(header)
        return [
            ExperienceEntry(
                title=title, organization=organization, bullets=bullets, raw=body.strip()
            )
        ]

    entries: list[ExperienceEntry] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(raw_lines)
        block = raw_lines[start:end]
        header_lines = [ln for ln in block[:2] if not is_bullet(ln)]
        header = " ".join(header_lines)
        dates = parse_date_range(header) or parse_date_range(" ".join(block[:3]))
        title, organization = _split_header(_strip_dates(header))
        entries.append(
            ExperienceEntry(
                title=title,
                organization=organization,
                dates=dates,
                bullets=[strip_bullet(ln) for ln in block if is_bullet(ln)],
                raw="\n".join(block),
            )
        )
    return entries


def _strip_dates(text: str) -> str:
    return DATE_RANGE_RE.sub("", text).strip(" ,|-–—•·")


def _split_header(header: str) -> tuple[str | None, str | None]:
    """Guess which half of an entry header is the title and which the employer."""
    header = header.strip(" ,|-–—•·")
    if not header:
        return None, None
    parts = [p.strip(" ,|-–—•·") for p in _SPLIT_RE.split(header) if p.strip(" ,|-–—•·")]
    if not parts:
        return None, None
    if len(parts) == 1:
        single = parts[0]
        looks_like_title = any(h in single.lower() for h in _TITLE_HINTS)
        return (single, None) if looks_like_title else (None, single)

    def title_score(part: str) -> int:
        return sum(1 for hint in _TITLE_HINTS if hint in part.lower())

    def org_score(part: str) -> int:
        lowered = part.lower()
        return sum(1 for hint in _ORG_HINTS if re.search(rf"\b{re.escape(hint)}", lowered))

    best_title = max(parts, key=title_score)
    remaining = [p for p in parts if p is not best_title]
    best_org = max(remaining, key=org_score) if remaining else None

    if title_score(best_title) == 0 and org_score(parts[0]) == 0:
        # No hints at all: fall back to the conventional "Title, Company" order.
        return parts[0], parts[1] if len(parts) > 1 else None
    return best_title, best_org


def extract_education(body: str) -> list[EducationEntry]:
    """Split an education section into degree entries.

    Entries are anchored on degree keywords; an institution line with no degree
    keyword still produces an entry so the section is never silently dropped.
    """
    raw_lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if not raw_lines:
        return []

    starts = [i for i, ln in enumerate(raw_lines) if DEGREE_RE.search(ln) and not is_bullet(ln)]
    if not starts:
        starts = [i for i, ln in enumerate(raw_lines) if _INSTITUTION_RE.search(ln)]
    if not starts:
        starts = [0]

    entries: list[EducationEntry] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(raw_lines)
        block = " \n".join(raw_lines[start:end])
        flat = block.replace("\n", " ")

        degree_match = DEGREE_RE.search(flat)
        institution_match = _INSTITUTION_RE.search(flat)
        gpa_match = GPA_RE.search(flat)

        field_of_study = None
        if degree_match:
            tail = flat[degree_match.end() :]
            field_match = _FIELD_RE.match(tail.strip()) or _FIELD_RE.search(tail[:80])
            if field_match:
                field_of_study = field_match.group(1).strip(" ,.")

        entries.append(
            EducationEntry(
                degree=degree_match.group(0).strip() if degree_match else None,
                field_of_study=field_of_study,
                institution=institution_match.group(1).strip(" ,.") if institution_match else None,
                dates=parse_date_range(flat),
                gpa=_parse_gpa(gpa_match),
                raw=block.strip(),
            )
        )
    return entries


def _parse_gpa(match: re.Match[str] | None) -> float | None:
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    scale = float(match.group(2)) if match.group(2) else 4.0
    if scale and scale not in (4.0, 4.3, 5.0, 10.0):
        scale = 4.0
    # Normalise everything to a 4.0 scale so comparisons are meaningful.
    normalised = value * (4.0 / scale) if scale else value
    return round(min(normalised, 4.0), 2)


def total_experience_months(entries: list[ExperienceEntry]) -> int:
    """Sum tenure across entries, merging overlaps so concurrent roles count once."""
    intervals: list[tuple[int, int]] = []
    today = date.today()
    horizon = today.year * 12 + today.month

    for entry in entries:
        dates = entry.dates
        if not dates or dates.start_year is None:
            continue
        start = dates.start_year * 12 + (dates.start_month or 1)
        if dates.is_current or dates.end_year is None:
            end = horizon
        else:
            end = dates.end_year * 12 + (dates.end_month or 12)
        end = min(end, horizon)
        if end >= start:
            intervals.append((start, end))

    if not intervals:
        return 0

    intervals.sort()
    merged: list[list[int]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start + 1 for start, end in merged)
