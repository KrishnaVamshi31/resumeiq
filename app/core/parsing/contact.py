"""Contact-block extraction.

Everything here runs on the top of the document, where contact details live.
Restricting the search window avoids matching a reference's email or a company
URL buried in the experience section.
"""

from __future__ import annotations

import re

from app.core.parsing.model import ContactInfo

HEADER_LINES = 15

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
# International-friendly: optional country code, 7-15 digits, common separators.
PHONE_RE = re.compile(
    r"(?<![\d\-])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?"
    r"\d{3,5}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?![\d\-])"
)
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9\-]+\.[A-Za-z]{2,}(?:/[^\s,;)]*)?")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub)/[\w\-%.]+/?", re.I)
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-.]+/?", re.I)

# "City, ST" / "City, Country" / "City, ST 12345" - allows accented city names.
LOCATION_RE = re.compile(
    r"\b([A-Z][\w'’\-]+(?:[ ][A-Z][\w'’\-]+){0,2}),\s*"
    r"([A-Z]{2}|[A-Z][a-z]+(?:[ ][A-Z][a-z]+)?)\b(?:\s+\d{5}(?:-\d{4})?)?"
)

_NAME_STOPWORDS = frozenset(
    {"resume", "cv", "curriculum", "vitae", "profile", "portfolio", "contact"}
)
_NAME_RE = re.compile(r"^[A-Z][A-Za-z'’.\-]+(?:\s+[A-Z][A-Za-z'’.\-]+){1,3}$")

_LABELS_RE = re.compile(
    r"^\s*(phone|tel|telephone|mobile|cell|e-?mail|mail|address|location|linkedin|github|"
    r"portfolio|website|web)\s*[:\-|]\s*",
    re.I,
)


def extract_contact(text: str) -> ContactInfo:
    """Pull contact details from the document header."""
    all_lines = [ln.strip() for ln in text.split("\n")]
    header_lines = [ln for ln in all_lines[:HEADER_LINES] if ln]
    header = "\n".join(header_lines)

    info = ContactInfo()

    # Email and phone are unambiguous enough to search the whole document if the
    # header comes up empty (some templates put contact details in a sidebar).
    info.email = _first(EMAIL_RE, header) or _first(EMAIL_RE, text)
    info.phone = _find_phone(header) or _find_phone(text[:4000])

    info.linkedin = _normalise_url(_first(LINKEDIN_RE, text))
    info.github = _normalise_url(_first(GITHUB_RE, text))
    info.website, info.other_links = _find_links(header, info)
    info.name = _find_name(header_lines, info)
    info.location = _find_location(header_lines)

    return info


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def _find_phone(text: str) -> str | None:
    """Take the first candidate with a plausible digit count.

    The naive pattern also matches date ranges and zip codes, so candidates are
    filtered on digit count and rejected when they sit inside a year range.
    """
    for match in PHONE_RE.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if not 7 <= len(digits) <= 15:
            continue
        # "2019 - 2022" and "2019-2022" are date ranges, not phone numbers.
        if re.fullmatch(r"(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}", candidate):
            continue
        return candidate
    return None


def _normalise_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.rstrip("/")
    return url if url.startswith("http") else f"https://{url}"


def _find_links(header: str, info: ContactInfo) -> tuple[str | None, list[str]]:
    """Any remaining header URLs: the first is treated as a personal site."""
    skip = ("linkedin.com", "github.com")
    others: list[str] = []
    for match in URL_RE.finditer(header):
        raw = match.group(0).rstrip(".,;")
        lowered = raw.lower()
        if any(s in lowered for s in skip) or "@" in raw:
            continue
        # URL_RE also matches the domain half of an email address.
        if re.search(rf"@{re.escape(raw)}", header):
            continue
        if not re.search(r"\.(com|net|org|io|dev|me|co|ai|app|xyz|tech|page|site)\b", lowered):
            continue
        normalised = _normalise_url(raw)
        if normalised and normalised not in others:
            others.append(normalised)
    website = others[0] if others else None
    return website, others[1:]


def _find_name(header_lines: list[str], info: ContactInfo) -> str | None:
    """The name is nearly always the first substantial line of the document."""
    for line in header_lines[:5]:
        candidate = _LABELS_RE.sub("", line).strip()
        # Templates often render the name as "JOHN A. SMITH".
        if candidate.isupper() and 2 <= len(candidate.split()) <= 4:
            candidate = candidate.title()
        if not _NAME_RE.match(candidate):
            continue
        if any(w.lower() in _NAME_STOPWORDS for w in candidate.split()):
            continue
        if "@" in candidate or any(ch.isdigit() for ch in candidate):
            continue
        return candidate

    # Fall back to the local part of the email: "jane.doe@x.com" -> "Jane Doe".
    if info.email:
        local = info.email.split("@")[0]
        parts = [p for p in re.split(r"[._\-]+", local) if p.isalpha() and len(p) > 1]
        if len(parts) >= 2:
            return " ".join(p.capitalize() for p in parts[:3])
    return None


def _find_location(header_lines: list[str]) -> str | None:
    for line in header_lines:
        cleaned = _LABELS_RE.sub("", line)
        match = LOCATION_RE.search(cleaned)
        if match:
            return match.group(0).strip()
    return None
