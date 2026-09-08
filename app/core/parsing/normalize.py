"""Text normalisation primitives shared by every downstream analyser."""

from __future__ import annotations

import re
import unicodedata

_BULLET_CHARS = "•‣▪●◦⁃∙·‐‒–—"
_BULLET_RE = re.compile(rf"^\s*[{re.escape(_BULLET_CHARS)}*\- ]+\s*")
_WHITESPACE_RE = re.compile(r"[ \t  -​]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-/]*", re.IGNORECASE)

# Tokens that carry no signal for similarity or keyword matching.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been being but by for from had has have he her his i if in into is
    it its me my of on or our ours she that the their them there these they this those to
    was we were what when where which who will with you your about across after all also am
    any because before both can could do does doing during each few had how more most no nor
    not only other over own same so some such than then through too under up very via while
    would should may might must shall
    """.split()
)

# Verb-noun pairs that make bullets sound busy without adding information.
FILLER_PHRASES: tuple[str, ...] = (
    "responsible for",
    "duties included",
    "duties involved",
    "tasked with",
    "worked on",
    "helped with",
    "involved in",
    "in charge of",
    "assisted with",
    "participated in",
    "hard worker",
    "team player",
    "detail oriented",
    "detail-oriented",
    "go-getter",
    "results-driven",
    "self-starter",
    "think outside the box",
    "proven track record",
    "excellent communication skills",
    "familiar with",
    "exposure to",
)

FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'll|me|my|mine|myself)\b")

# Weak lead-ins: a bullet starting here describes a role, not an achievement.
WEAK_OPENERS: tuple[str, ...] = (
    "responsible",
    "worked",
    "helped",
    "assisted",
    "participated",
    "involved",
    "tasked",
    "handled",
    "dealt",
    "attended",
)


def clean_text(text: str) -> str:
    """Canonicalise unicode and whitespace without destroying line structure."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def strip_bullet(line: str) -> str:
    """Remove a leading bullet glyph or dash from a line."""
    return _BULLET_RE.sub("", line).strip()


def is_bullet(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _BULLET_RE.match(stripped) and len(strip_bullet(stripped)) >= 3:
        return True
    # "1. Built X" style numbered achievements.
    return bool(re.match(r"^\s*\(?\d{1,2}[.)]\s+\S", stripped))


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping the punctuation that matters in tech terms.

    `c++`, `node.js` and `ci/cd` survive intact, but *trailing* punctuation is
    stripped - without that, a sentence-final "Python." is a different token
    from "Python" and the two texts share nothing.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).lower().rstrip("./-")
        if token:
            tokens.append(token)
    return tokens


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and single characters removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    """Split into rough sentences. Good enough for readability heuristics."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.strip()]


def collapse(text: str) -> str:
    """Single-line form, for comparisons where layout is irrelevant."""
    return _WHITESPACE_RE.sub(" ", text.replace("\n", " ")).strip()
