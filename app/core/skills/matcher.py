"""Skill extraction with span masking, context weighting and fuzzy recovery."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.core.skills.taxonomy import Taxonomy, load_taxonomy, normalise_surface

#: How much weight a mention carries depending on where it appears. A skill
#: demonstrated in a work bullet is stronger evidence than one listed in a
#: keyword dump, and the confidence score reflects that.
CONTEXT_WEIGHT: dict[str, float] = {
    "experience": 1.0,
    "projects": 0.9,
    "summary": 0.8,
    "certifications": 0.8,
    "skills": 0.7,
    "education": 0.6,
    "other": 0.5,
}

FUZZY_THRESHOLD = 0.88
_EVIDENCE_WINDOW = 90


@dataclass(slots=True)
class SkillMention:
    """One canonical skill, plus where and how often it was seen."""

    name: str
    category: str
    count: int = 0
    contexts: set[str] = field(default_factory=set)
    evidence: list[str] = field(default_factory=list)
    fuzzy: bool = False

    @property
    def confidence(self) -> float:
        """0-1 confidence that the candidate genuinely has this skill.

        Driven by the strongest context it appears in, nudged up by repetition
        (a skill named in three bullets is more credible than one keyword).
        """
        best_context = max(
            (CONTEXT_WEIGHT.get(c, CONTEXT_WEIGHT["other"]) for c in self.contexts),
            default=CONTEXT_WEIGHT["other"],
        )
        repetition_bonus = min(0.15, 0.05 * (self.count - 1))
        score = min(1.0, best_context + repetition_bonus)
        return round(score * (0.85 if self.fuzzy else 1.0), 3)

    @property
    def is_demonstrated(self) -> bool:
        """True when the skill appears somewhere other than a keyword list."""
        return bool(self.contexts - {"skills", "other"})


def extract_skills(
    contexts: Mapping[str, str],
    taxonomy: Taxonomy | None = None,
    *,
    fuzzy: bool = True,
) -> dict[str, SkillMention]:
    """Find every taxonomy skill across a set of labelled text blocks.

    ``contexts`` maps a context label (``"experience"``, ``"skills"``, ...) to
    the text of that block. Labels drive confidence weighting.
    """
    taxonomy = taxonomy or load_taxonomy()
    found: dict[str, SkillMention] = {}

    for label, text in contexts.items():
        if not text or not text.strip():
            continue
        for name, count, evidence in _scan(text, taxonomy):
            mention = found.setdefault(
                name, SkillMention(name=name, category=taxonomy.category_of(name))
            )
            mention.count += count
            mention.contexts.add(label)
            for snippet in evidence:
                if len(mention.evidence) < 3 and snippet not in mention.evidence:
                    mention.evidence.append(snippet)

    if fuzzy:
        _recover_typos(contexts, taxonomy, found)
    return found


def _scan(text: str, taxonomy: Taxonomy) -> list[tuple[str, int, list[str]]]:
    """Exact-match scan with span masking so longer skills win.

    Patterns are pre-sorted longest-first, so once "React Native" claims its
    characters, the "React" pattern cannot re-match the same span.
    """
    consumed = bytearray(len(text))
    hits: dict[str, tuple[int, list[str]]] = {}

    for pattern, name in taxonomy.patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(consumed[start:end]):
                continue
            consumed[start:end] = b"\x01" * (end - start)
            count, evidence = hits.get(name, (0, []))
            if len(evidence) < 3:
                evidence.append(_snippet(text, start, end))
            hits[name] = (count + 1, evidence)

    return [(name, count, evidence) for name, (count, evidence) in hits.items()]


def _snippet(text: str, start: int, end: int) -> str:
    """A readable one-line quote around the match, used as UI evidence."""
    left = max(0, start - _EVIDENCE_WINDOW // 2)
    right = min(len(text), end + _EVIDENCE_WINDOW // 2)
    fragment = text[left:right].replace("\n", " ").strip()
    fragment = re.sub(r"\s+", " ", fragment)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{fragment}{suffix}"


def _recover_typos(
    contexts: Mapping[str, str], taxonomy: Taxonomy, found: dict[str, SkillMention]
) -> None:
    """Catch near-misses such as "Kubernets" or "Postgress".

    Only comma/pipe-delimited list items are considered - those are the skill
    dumps where typos hide - and only tokens long enough that a near match is
    unlikely to be a coincidence.
    """
    candidates: set[str] = set()
    for label, text in contexts.items():
        if label not in {"skills", "summary"} or not text:
            continue
        for chunk in re.split(r"[,|•\n;/]+", text):
            token = normalise_surface(chunk)
            if 4 <= len(token) <= 25 and " " not in token:
                candidates.add(token)

    known_keys = list(taxonomy.surface_index)
    for candidate in candidates:
        if candidate in taxonomy.surface_index:
            continue
        best_key, best_ratio = None, 0.0
        for key in known_keys:
            if abs(len(key) - len(candidate)) > 2 or key[0] != candidate[0]:
                continue
            ratio = SequenceMatcher(None, candidate, key).ratio()
            if ratio > best_ratio:
                best_key, best_ratio = key, ratio
        if best_key and best_ratio >= FUZZY_THRESHOLD:
            name = taxonomy.surface_index[best_key]
            if name in found:
                continue
            mention = SkillMention(name=name, category=taxonomy.category_of(name), fuzzy=True)
            mention.count = 1
            mention.contexts.add("skills")
            mention.evidence.append(f'Matched from "{candidate}" (possible misspelling)')
            found[name] = mention


def top_skills(mentions: Mapping[str, SkillMention], limit: int = 20) -> list[SkillMention]:
    """Highest-confidence skills first, ties broken by frequency."""
    return sorted(
        mentions.values(), key=lambda m: (-m.confidence, -m.count, m.name)
    )[:limit]
