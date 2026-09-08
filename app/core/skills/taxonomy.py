"""Skill ontology loading and lookup.

The taxonomy is data (``data/skills.json``), not code, so it can be extended by
a domain expert without touching the matcher. Everything is indexed once at
import time into surface-form -> canonical-skill maps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.logging_conf import get_logger

logger = get_logger(__name__)

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "data" / "skills.json"


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...] = ()
    related: tuple[str, ...] = ()

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(slots=True)
class Taxonomy:
    """Indexed skill ontology."""

    skills: dict[str, Skill] = field(default_factory=dict)
    #: normalised surface form -> canonical skill name
    surface_index: dict[str, str] = field(default_factory=dict)
    #: compiled regex per surface form, longest-first for greedy matching
    patterns: list[tuple[re.Pattern[str], str]] = field(default_factory=list)

    def canonical(self, surface: str) -> str | None:
        return self.surface_index.get(normalise_surface(surface))

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def related_to(self, name: str) -> tuple[str, ...]:
        skill = self.skills.get(name)
        return skill.related if skill else ()

    def category_of(self, name: str) -> str:
        skill = self.skills.get(name)
        return skill.category if skill else "other"

    def __len__(self) -> int:
        return len(self.skills)


def normalise_surface(text: str) -> str:
    """Fold a surface form to its comparison key.

    Keeps the characters that distinguish real technologies (`c++` vs `c`,
    `.net`, `ci/cd`, `node.js`) and collapses everything else.
    """
    text = text.strip().lower()
    text = re.sub(r"[‐-―]", "-", text)
    text = re.sub(r"[^a-z0-9+#./\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_pattern(surface: str) -> re.Pattern[str] | None:
    """Word-boundary regex that survives `+`, `#`, `.` and `/` in skill names."""
    escaped = re.escape(surface)
    # `\b` does not fire after a non-word character such as `+` in "c++".
    prefix = r"(?<![\w+#])" if surface[0].isalnum() else r"(?<![\w])"
    suffix = r"(?![\w+#])" if surface[-1].isalnum() else r"(?![\w])"
    try:
        return re.compile(prefix + escaped.replace(r"\ ", r"[\s\-]+") + suffix, re.IGNORECASE)
    except re.error:
        logger.warning("skill_pattern_invalid", extra={"surface": surface})
        return None


@lru_cache(maxsize=4)
def load_taxonomy(path: str | None = None) -> Taxonomy:
    """Load and index the ontology. Cached per path."""
    source = Path(path) if path else DEFAULT_TAXONOMY_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("taxonomy_missing", extra={"path": str(source)})
        return Taxonomy()
    except json.JSONDecodeError as exc:
        logger.error("taxonomy_invalid_json", extra={"path": str(source), "detail": str(exc)})
        return Taxonomy()

    taxonomy = Taxonomy()
    for entry in payload.get("skills", []):
        skill = Skill(
            name=entry["name"],
            category=entry.get("category", "other"),
            aliases=tuple(entry.get("aliases", ())),
            related=tuple(entry.get("related", ())),
        )
        taxonomy.skills[skill.name] = skill
        for surface in skill.surface_forms:
            key = normalise_surface(surface)
            if not key:
                continue
            # First writer wins so a canonical name never loses to another
            # skill's alias (e.g. "aws lambda" is an alias of both AWS and
            # Serverless; AWS is declared first and keeps the key).
            taxonomy.surface_index.setdefault(key, skill.name)

    # Longest surface forms first: "machine learning" must win over "learning",
    # and "react native" over "react".
    for surface_key in sorted(taxonomy.surface_index, key=len, reverse=True):
        pattern = _build_pattern(surface_key)
        if pattern is not None:
            taxonomy.patterns.append((pattern, taxonomy.surface_index[surface_key]))

    logger.info(
        "taxonomy_loaded",
        extra={"skills": len(taxonomy.skills), "surface_forms": len(taxonomy.surface_index)},
    )
    return taxonomy
