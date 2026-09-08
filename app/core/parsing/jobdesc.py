"""Job description parsing.

A job post is far less structured than a resume, but it reliably separates hard
requirements from nice-to-haves under recognisable headings. Getting that split
right matters more than anything else here: a missing *required* skill should
cost the candidate far more than a missing *preferred* one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.parsing.normalize import STOPWORDS, clean_text, is_bullet, strip_bullet
from app.core.skills.matcher import SkillMention, extract_skills
from app.core.skills.taxonomy import Taxonomy, load_taxonomy


class Seniority(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"
    UNSPECIFIED = "unspecified"


#: Ordered by rank so seniority distance is computable.
SENIORITY_RANK: dict[Seniority, int] = {
    Seniority.INTERN: 0,
    Seniority.JUNIOR: 1,
    Seniority.MID: 2,
    Seniority.SENIOR: 3,
    Seniority.STAFF: 4,
    Seniority.PRINCIPAL: 5,
    Seniority.MANAGER: 5,
    Seniority.DIRECTOR: 6,
    Seniority.EXECUTIVE: 7,
    Seniority.UNSPECIFIED: -1,
}

_SENIORITY_PATTERNS: tuple[tuple[Seniority, str], ...] = (
    (Seniority.EXECUTIVE, r"\b(chief|c[toei]o|vp|vice president|head of)\b"),
    (Seniority.DIRECTOR, r"\bdirector\b"),
    (Seniority.MANAGER, r"\b(manager|engineering manager|em)\b"),
    (Seniority.PRINCIPAL, r"\b(principal|distinguished|fellow)\b"),
    (Seniority.STAFF, r"\b(staff|architect)\b"),
    (Seniority.SENIOR, r"\b(senior|sr\.?|lead|iii|iv)\b"),
    (Seniority.INTERN, r"\b(intern|internship|co-?op|trainee)\b"),
    (Seniority.JUNIOR, r"\b(junior|jr\.?|entry[- ]level|graduate|associate|i{1,2}\b)"),
)

_REQUIRED_HEADINGS = (
    "requirements", "required", "basic qualifications", "minimum qualifications",
    "must have", "must-have", "what you need", "what you'll need", "qualifications",
    "who you are", "essential", "required skills", "required qualifications",
    "we're looking for", "we are looking for", "you have",
)
_PREFERRED_HEADINGS = (
    "preferred", "preferred qualifications", "nice to have", "nice-to-have",
    "bonus", "bonus points", "plus", "desired", "desirable", "good to have",
    "additionally", "even better", "pluses", "preferred skills",
)
_RESPONSIBILITY_HEADINGS = (
    "responsibilities", "what you'll do", "what you will do", "the role",
    "about the role", "your impact", "day to day", "duties", "key responsibilities",
)
_IGNORED_HEADINGS = (
    "benefits", "perks", "compensation", "salary", "equal opportunity", "eeo",
    "about us", "about the company", "our mission", "how to apply", "why join",
)

_YEARS_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*(\d{1,2})?\s*\+?\s*years?\b[^.]{0,40}?"
    r"\b(?:experience|exp\b|working)",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(
    r"\b(bachelor(?:'s)?|b\.?s\.?|b\.?a\.?|master(?:'s)?|m\.?s\.?|m\.?b\.?a\.?|"
    r"ph\.?d|doctorate|associate(?:'s)?)\b",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^\s*([A-Za-z][A-Za-z'’/&\- ]{2,50})\s*:?\s*$")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}(?:\d{1,2})?\b")


@dataclass(slots=True)
class SkillRequirement:
    """A single skill the posting asks for."""

    name: str
    required: bool
    category: str = "other"
    mentions: int = 1
    in_taxonomy: bool = True
    evidence: str = ""

    @property
    def weight(self) -> float:
        """Relative importance inside its own bucket.

        A skill named three times in the requirements is genuinely more central
        than one named once, but the effect is capped so a keyword-stuffed post
        cannot let a single term dominate the score.
        """
        return round(min(1.0, 0.7 + 0.15 * (self.mentions - 1)), 3)


@dataclass(slots=True)
class ParsedJob:
    raw_text: str
    title: str | None = None
    company: str | None = None
    seniority: Seniority = Seniority.UNSPECIFIED
    min_years_experience: int | None = None
    max_years_experience: int | None = None
    degree_required: str | None = None
    required_skills: list[SkillRequirement] = field(default_factory=list)
    preferred_skills: list[SkillRequirement] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)

    @property
    def all_skills(self) -> list[SkillRequirement]:
        return [*self.required_skills, *self.preferred_skills]

    def skill_names(self) -> set[str]:
        return {s.name for s in self.all_skills}


def parse_job(
    text: str, *, title: str | None = None, company: str | None = None,
    taxonomy: Taxonomy | None = None,
) -> ParsedJob:
    """Parse a job posting into requirements, preferences and keywords."""
    taxonomy = taxonomy or load_taxonomy()
    cleaned = clean_text(text)
    buckets = _split_buckets(cleaned)

    inferred_title = title or _infer_title(cleaned)
    seniority = detect_seniority(f"{inferred_title or ''} {buckets['required']}")
    min_years, max_years = _extract_years(cleaned)

    # Skills are scanned per bucket so a mention's bucket decides required vs
    # preferred. A skill in both buckets counts as required.
    mentions = extract_skills(
        {
            "required": buckets["required"],
            "preferred": buckets["preferred"],
            "responsibilities": buckets["responsibilities"],
            "other": buckets["other"],
        },
        taxonomy,
        fuzzy=False,
    )

    required: list[SkillRequirement] = []
    preferred: list[SkillRequirement] = []
    for mention in mentions.values():
        is_required = bool(mention.contexts & {"required", "responsibilities"})
        # A post with no headed requirements section still has real requirements.
        if not is_required and mention.contexts == {"other"} and not buckets["required"].strip():
            is_required = True
        requirement = SkillRequirement(
            name=mention.name,
            required=is_required,
            category=mention.category,
            mentions=mention.count,
            evidence=mention.evidence[0] if mention.evidence else "",
        )
        (required if is_required else preferred).append(requirement)

    required.sort(key=lambda s: (-s.mentions, s.name))
    preferred.sort(key=lambda s: (-s.mentions, s.name))

    return ParsedJob(
        raw_text=cleaned,
        title=inferred_title,
        company=company,
        seniority=seniority,
        min_years_experience=min_years,
        max_years_experience=max_years,
        degree_required=_extract_degree(buckets["required"] or cleaned),
        required_skills=required,
        preferred_skills=preferred,
        keywords=_extract_keywords(cleaned, buckets, mentions),
        responsibilities=_bullets_of(buckets["responsibilities"])[:12],
    )


def _split_buckets(text: str) -> dict[str, str]:
    """Route each line into required / preferred / responsibilities / other."""
    buckets: dict[str, list[str]] = {
        "required": [], "preferred": [], "responsibilities": [], "other": []
    }
    current = "other"

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        heading = _match_heading(stripped)
        if heading is not None:
            current = heading
            continue
        buckets[current].append(stripped)

    return {key: "\n".join(value) for key, value in buckets.items()}


def _match_heading(line: str) -> str | None:
    """Identify a bucket-switching heading, including inline ``Requirements:``."""
    match = _HEADING_RE.match(line)
    candidate = (match.group(1) if match else line).strip().lower().rstrip(":")
    if not match and not line.rstrip().endswith(":"):
        return None
    if len(candidate) > 60:
        return None
    # Check preferred first: "Preferred Qualifications" also contains
    # "qualifications", which is a required-bucket heading.
    for heading in _PREFERRED_HEADINGS:
        if heading in candidate:
            return "preferred"
    for heading in _RESPONSIBILITY_HEADINGS:
        if heading in candidate:
            return "responsibilities"
    for heading in _REQUIRED_HEADINGS:
        if heading in candidate:
            return "required"
    for heading in _IGNORED_HEADINGS:
        if heading in candidate:
            return "other"
    return None


def _infer_title(text: str) -> str | None:
    """The job title is nearly always the first non-boilerplate line."""
    for line in text.split("\n")[:6]:
        stripped = line.strip(" :-–—|")
        if not stripped or len(stripped) > 80:
            continue
        lowered = stripped.lower()
        if any(word in lowered for word in ("about", "we are", "job description", "apply")):
            continue
        if _match_heading(stripped):
            continue
        if len(stripped.split()) <= 10:
            return stripped
    return None


def detect_seniority(text: str) -> Seniority:
    lowered = text.lower()
    for level, pattern in _SENIORITY_PATTERNS:
        if re.search(pattern, lowered):
            return level
    return Seniority.UNSPECIFIED


def _extract_years(text: str) -> tuple[int | None, int | None]:
    """Lowest stated minimum wins - postings repeat the bar per skill."""
    matches = list(_YEARS_RE.finditer(text))
    if not matches:
        return None, None
    lows: list[int] = []
    highs: list[int] = []
    for match in matches:
        low = int(match.group(1))
        if low > 40:
            continue
        lows.append(low)
        if match.group(2):
            high = int(match.group(2))
            if low <= high <= 40:
                highs.append(high)
    if not lows:
        return None, None
    return min(lows), (max(highs) if highs else None)


def _extract_degree(text: str) -> str | None:
    match = _DEGREE_RE.search(text)
    return match.group(0) if match else None


def _bullets_of(text: str) -> list[str]:
    out = [strip_bullet(ln) for ln in text.split("\n") if is_bullet(ln)]
    if out:
        return out
    return [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 20]


#: Words that appear in every job post and carry no matching signal.
_POSTING_BOILERPLATE: frozenset[str] = frozenset(
    ["experience", "experiences", "year", "years", "strong", "solid", "proven", "demonstrated", "deep", "hands", "excellent", "good", "great", "ability", "able", "skills", "skill", "knowledge", "understanding", "background", "familiarity", "exposure", "equivalent", "practical", "relevant", "related", "degree", "bachelor", "bachelors", "master", "masters", "preferred", "required", "must", "plus", "role", "position", "job", "candidate", "candidates", "applicant", "team", "teams", "company", "work", "working", "works", "opportunity", "opportunities", "including", "etc"]
)

#: Terms in the requirements section are the literal strings ATS filters query,
#: so they count for more than the same phrase in a responsibilities bullet.
_BUCKET_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("required", 2.0),
    ("responsibilities", 1.0),
)
_KEYWORD_MIN_SCORE = 2.0


def _extract_keywords(
    text: str, buckets: dict[str, str], mentions: dict[str, SkillMention], limit: int = 25
) -> list[str]:
    """ATS keywords the taxonomy does not know about.

    Job posts always contain domain terms outside any fixed ontology
    ("claims adjudication", "FDA submissions"). Recruiters' ATS filters key off
    exactly those, so they are surfaced as literal keywords to mirror back.
    """
    known = {name.lower() for name in mentions}
    sources: list[tuple[str, float]] = [
        (buckets[name], weight) for name, weight in _BUCKET_WEIGHTS if buckets.get(name, "").strip()
    ]
    if not sources:
        sources = [(text, 1.0)]

    # Scores are fractional (bucket weight x phrase length), so this is a plain
    # float map rather than a Counter, which is integer-valued.
    scores: dict[str, float] = {}
    for source, weight in sources:
        for line in source.split("\n"):
            cleaned_line = strip_bullet(line)
            # Acronyms are high-signal ATS keywords and survive on their own.
            for acronym in _ACRONYM_RE.findall(cleaned_line):
                if acronym.lower() not in STOPWORDS and acronym.lower() not in known:
                    key = acronym.lower()
                    scores[key] = scores.get(key, 0.0) + weight + 1.0
            words = re.findall(r"[A-Za-z][A-Za-z\-']+", cleaned_line)
            for size in (3, 2):
                for i in range(len(words) - size + 1):
                    gram = [w.lower() for w in words[i : i + size]]
                    if any(w in STOPWORDS or w in _POSTING_BOILERPLATE for w in gram):
                        continue
                    if any(len(w) < 3 for w in gram):
                        continue
                    phrase = " ".join(gram)
                    if phrase in known:
                        continue
                    # Longer phrases are more specific, so they score higher.
                    scores[phrase] = scores.get(phrase, 0.0) + weight * (
                        1.0 + 0.25 * (size - 2)
                    )

    ranked = [
        phrase
        for phrase, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if score >= _KEYWORD_MIN_SCORE
    ][: limit * 4]
    return _drop_redundant(ranked, limit)


def _drop_redundant(phrases: list[str], limit: int) -> list[str]:
    """Keep the most specific form of each phrase.

    Without this, selecting "payment settlement pipeline" also drags in
    "payment settlement" and "settlement pipeline", filling the keyword list
    with three views of one term.
    """
    kept: list[str] = []
    for phrase in phrases:
        if any(phrase in existing for existing in kept):
            continue
        kept = [k for k in kept if k not in phrase]
        kept.append(phrase)
        if len(kept) >= limit:
            break
    return kept
