"""Content quality scoring.

This is the dimension that separates a resume which *passes* screening from one
that *wins* an interview. Every check targets a specific, well-evidenced failure
mode in resume writing: describing duties instead of outcomes, omitting the
numbers that make an outcome credible, opening bullets with weak verbs,
repeating the same verb throughout, and padding with self-assessed adjectives no
reviewer believes.
"""

from __future__ import annotations

import re
from collections import Counter

from app.core.parsing.model import ParsedResume, SectionKind
from app.core.parsing.normalize import (
    FILLER_PHRASES,
    FIRST_PERSON_RE,
    WEAK_OPENERS,
    collapse,
)
from app.core.scoring.base import DimensionId, DimensionScore, Signal, make_signal

#: Strong, outcome-oriented verbs. Grouped only for readability.
ACTION_VERBS: frozenset[str] = frozenset(
    """
    accelerated achieved acquired adapted addressed advanced advised advocated analyzed
    architected assembled audited authored automated balanced boosted brokered budgeted
    built centralized chaired championed clarified coached collaborated compiled completed
    composed conceived conducted consolidated constructed consulted converted coordinated
    created cultivated cut decreased defined delivered demonstrated deployed designed
    detected determined developed devised diagnosed directed discovered documented doubled
    drafted drove earned edited educated eliminated enabled engineered enhanced enlisted
    ensured established evaluated examined executed expanded expedited facilitated finalized
    forecasted formulated founded generated grew guided halved handled headed identified
    implemented improved increased influenced initiated innovated inspected installed
    instituted instructed integrated introduced invented investigated launched led leveraged
    maintained managed mapped marketed maximized measured mediated mentored migrated
    minimized modeled modernized monitored motivated negotiated operated optimized
    orchestrated organized outlined overhauled oversaw performed pioneered planned prepared
    presented prioritized produced programmed promoted proposed prototyped published
    quantified raised ranked rearchitected rebuilt received recruited redesigned reduced
    refactored refined reorganized repaired replaced reported researched resolved restored
    restructured revamped reviewed revised saved scaled scheduled secured shaped shipped
    simplified solved sourced spearheaded standardized steered streamlined strengthened
    structured supervised supported surveyed sustained synthesized systematized taught
    tested tracked trained transformed translated tripled troubleshot unified upgraded
    validated verified won wrote
    """.split()
)

#: Self-assessed adjectives that assert rather than evidence.
BUZZWORDS: tuple[str, ...] = (
    "synergy", "synergistic", "world-class", "best-in-class", "cutting-edge",
    "thought leader", "guru", "ninja", "rockstar", "wizard", "visionary",
    "dynamic", "passionate", "motivated", "hardworking", "hard-working",
    "seasoned", "results-oriented", "value-add", "leverage synergies",
)

#: A quantified bullet contains a number that means something.
_METRIC_RE = re.compile(
    r"(?:\$\s?\d[\d,.]*\s*(?:k|m|b|bn|mm)?|"          # money
    r"\d[\d,.]*\s*%|"                                  # percentage
    r"\b\d[\d,.]*\s*(?:x|times)\b|"                    # multiples
    r"\b\d[\d,.]*\s*(?:k|m|bn|million|billion|thousand)\b|"
    r"\b\d[\d,.]*\s*(?:users?|customers?|clients?|people|engineers?|reports?|"
    r"hours?|days?|weeks?|months?|requests?|transactions?|records?|tickets?|"
    r"accounts?|projects?|countries|stores?|units?|ms|seconds?|qps|rps)\b|"
    r"\b\d{2,}\b)",                                    # any bare number >= 10
    re.IGNORECASE,
)
_ING_OPENER_RE = re.compile(r"^\w+ing\b", re.IGNORECASE)

IDEAL_BULLET_WORDS = (10, 28)
#: Below this, the resume has too few bullets to judge fairly.
MIN_BULLETS_FOR_RATIOS = 3


def score_content(resume: ParsedResume) -> DimensionScore:
    """Judge how persuasively the resume is written."""
    bullets = [b for b in resume.bullets if len(b.split()) >= 3]
    signals: list[Signal] = [
        _action_verbs(bullets),
        _quantification(bullets),
        _bullet_length(bullets),
        _filler_language(resume, bullets),
        _first_person(resume),
        _verb_variety(bullets),
        _buzzwords(resume),
        _skills_section_quality(resume),
        _tense_consistency(resume, bullets),
    ]
    return DimensionScore(id=DimensionId.CONTENT, label="Content quality", signals=signals)


def _lead_verb(bullet: str) -> str:
    match = re.match(r"[^A-Za-z]*([A-Za-z][\w'-]*)", bullet)
    return match.group(1).lower() if match else ""


def _insufficient(signal_id: str, label: str, weight: float) -> Signal:
    """Neutral result when there is not enough text to judge."""
    return make_signal(
        signal_id, label, 0.5, weight=weight,
        detail="Too few bullet points to assess reliably.",
        recommendation="Add achievement bullets to your roles so this can be evaluated.",
    )


def _action_verbs(bullets: list[str]) -> Signal:
    if len(bullets) < MIN_BULLETS_FOR_RATIOS:
        return _insufficient("content.action_verbs", "Bullets open with strong verbs", 2.5)

    weak: list[str] = []
    strong = 0
    for bullet in bullets:
        verb = _lead_verb(bullet)
        if verb in ACTION_VERBS:
            strong += 1
        elif verb in WEAK_OPENERS or not verb:
            weak.append(bullet)
        # Unknown verbs are neither rewarded nor penalised: the list cannot be
        # exhaustive and a false penalty is worse than a missed reward.

    ratio = strong / len(bullets)
    penalty = 0.5 * (len(weak) / len(bullets))
    score = max(0.0, min(1.0, ratio + 0.25) - penalty)

    return make_signal(
        "content.action_verbs", "Bullets open with strong verbs", score, weight=2.5,
        detail=f"{strong} of {len(bullets)} bullets open with a strong action verb.",
        evidence=[f'Weak opener: "{collapse(b)[:110]}"' for b in weak[:4]],
        recommendation=(
            'Start every bullet with a past-tense achievement verb - "Led", "Built", '
            '"Reduced", "Shipped" - not "Responsible for" or "Worked on".'
        ),
    )


def _quantification(bullets: list[str]) -> Signal:
    if len(bullets) < MIN_BULLETS_FOR_RATIOS:
        return _insufficient("content.quantification", "Achievements are quantified", 3.0)

    quantified = [b for b in bullets if _METRIC_RE.search(b)]
    ratio = len(quantified) / len(bullets)
    # ~40% quantified is the practical ceiling for most roles; scale to that.
    score = min(1.0, ratio / 0.4)
    unquantified = [b for b in bullets if b not in quantified]

    return make_signal(
        "content.quantification", "Achievements are quantified", score, weight=3.0,
        detail=f"{len(quantified)} of {len(bullets)} bullets ({ratio:.0%}) contain a metric.",
        evidence=[f'No metric: "{collapse(b)[:110]}"' for b in unquantified[:4]],
        recommendation=(
            "Attach a number to at least a third of your bullets - money saved, percent "
            "improved, users served, time cut, team size. Unquantified claims read as opinion."
        ),
    )


def _bullet_length(bullets: list[str]) -> Signal:
    if len(bullets) < MIN_BULLETS_FOR_RATIOS:
        return _insufficient("content.bullet_length", "Bullets are a readable length", 1.5)

    low, high = IDEAL_BULLET_WORDS
    lengths = [len(b.split()) for b in bullets]
    in_range = sum(1 for n in lengths if low <= n <= high)
    score = in_range / len(lengths)
    too_long = [b for b in bullets if len(b.split()) > high]
    too_short = [b for b in bullets if len(b.split()) < low]

    evidence = [f'Too long ({len(b.split())} words): "{collapse(b)[:110]}"' for b in too_long[:3]]
    evidence += [f'Too short: "{collapse(b)[:110]}"' for b in too_short[:2]]

    return make_signal(
        "content.bullet_length", "Bullets are a readable length", score, weight=1.5,
        detail=f"Median bullet length {sorted(lengths)[len(lengths) // 2]} words.",
        evidence=evidence,
        recommendation=(
            f"Keep bullets between {low} and {high} words - one clear achievement each. "
            "Split anything longer; expand anything shorter into action plus outcome."
        ),
    )


def _filler_language(resume: ParsedResume, bullets: list[str]) -> Signal:
    text = resume.raw_text.lower()
    hits = [phrase for phrase in FILLER_PHRASES if phrase in text]
    if not hits:
        return make_signal("content.filler", "Free of filler phrasing", 1.0, weight=2.0)

    counted = sorted(((text.count(phrase), phrase) for phrase in hits), reverse=True)
    occurrences = sum(count for count, _ in counted)
    denominator = max(len(bullets), 5)
    score = max(0.0, 1.0 - occurrences / denominator)
    examples = [
        f'"{collapse(b)[:110]}"'
        for b in bullets
        if any(phrase in b.lower() for phrase in hits)
    ][:4]
    # Lead with the phrases used most - those are the ones worth rewriting.
    worst = ", ".join(phrase for _, phrase in counted[:6])

    return make_signal(
        "content.filler", "Free of filler phrasing", score, weight=2.0,
        detail=f"{occurrences} filler phrase(s): {worst}.",
        evidence=examples,
        recommendation=(
            'Replace duty language with outcomes. "Responsible for the deployment '
            'pipeline" becomes "Cut deployment time from 40 to 6 minutes by rebuilding '
            'the CI pipeline".'
        ),
    )


def _first_person(resume: ParsedResume) -> Signal:
    # The summary conventionally tolerates first person; bullets do not.
    body = "\n".join(
        section.body
        for kind, section in resume.sections.items()
        if kind not in {SectionKind.SUMMARY, SectionKind.CONTACT}
    )
    matches = FIRST_PERSON_RE.findall(body)
    if not matches:
        return make_signal("content.first_person", "Written in resume voice", 1.0, weight=1.0)
    score = max(0.0, 1.0 - len(matches) / 8)
    return make_signal(
        "content.first_person", "Written in resume voice", score, weight=1.0,
        detail=f"{len(matches)} first-person pronoun(s) outside the summary.",
        recommendation=(
            'Drop first-person pronouns from bullets. "I led a team of six" becomes '
            '"Led a team of six".'
        ),
    )


def _verb_variety(bullets: list[str]) -> Signal:
    if len(bullets) < 5:
        return _insufficient("content.verb_variety", "Verb choice is varied", 1.5)

    verbs = [_lead_verb(b) for b in bullets]
    verbs = [v for v in verbs if v]
    if not verbs:
        return _insufficient("content.verb_variety", "Verb choice is varied", 1.5)

    counts = Counter(verbs)
    unique_ratio = len(counts) / len(verbs)
    overused = [(verb, n) for verb, n in counts.most_common(3) if n >= 3]
    score = min(1.0, unique_ratio / 0.75)

    return make_signal(
        "content.verb_variety", "Verb choice is varied", score, weight=1.5,
        detail=f"{len(counts)} distinct opening verbs across {len(verbs)} bullets.",
        evidence=[f'"{verb.title()}" opens {n} bullets' for verb, n in overused],
        recommendation=(
            "Vary your opening verbs. Repeating the same verb makes distinct "
            "accomplishments blur into one."
        ),
    )


def _buzzwords(resume: ParsedResume) -> Signal:
    text = resume.raw_text.lower()
    hits = [word for word in BUZZWORDS if word in text]
    if not hits:
        return make_signal("content.buzzwords", "Avoids empty buzzwords", 1.0, weight=1.0)
    score = max(0.0, 1.0 - 0.2 * len(hits))
    return make_signal(
        "content.buzzwords", "Avoids empty buzzwords", score, weight=1.0,
        detail=f"Found: {', '.join(sorted(hits)[:8])}.",
        recommendation=(
            "Cut self-assessed adjectives and replace them with evidence. Nobody is "
            "convinced by 'passionate'; they are convinced by what you shipped."
        ),
    )


def _skills_section_quality(resume: ParsedResume) -> Signal:
    section = resume.sections.get(SectionKind.SKILLS)
    if not section:
        return make_signal(
            "content.skills_section", "Skills section is scannable", 0.3, weight=1.5,
            recommendation=(
                "Add a Skills section grouping your technologies by category. It is the "
                "first thing both keyword filters and human screeners look for."
            ),
        )

    body = section.body
    items = [i.strip() for i in re.split(r"[,|•\n;]+", body) if i.strip()]
    if not items:
        return make_signal(
            "content.skills_section", "Skills section is scannable", 0.3, weight=1.5,
            recommendation="Populate the Skills section with comma-separated technologies.",
        )

    average_length = sum(len(i.split()) for i in items) / len(items)
    if len(items) < 5:
        return make_signal(
            "content.skills_section", "Skills section is scannable", 0.5, weight=1.5,
            detail=f"Only {len(items)} skills listed.",
            recommendation="List 10-20 concrete skills, grouped by category (languages, tools, platforms).",
        )
    if average_length > 5:
        return make_signal(
            "content.skills_section", "Skills section is scannable", 0.6, weight=1.5,
            detail="Skills are written as prose rather than a list.",
            recommendation=(
                "Rewrite the Skills section as short comma-separated terms. Screeners "
                "scan it in about two seconds; sentences defeat that."
            ),
        )
    if len(items) > 45:
        return make_signal(
            "content.skills_section", "Skills section is scannable", 0.6, weight=1.5,
            detail=f"{len(items)} items listed.",
            recommendation=(
                "Trim the skills list to what you would defend in an interview. An "
                "indiscriminate list reads as padding and dilutes your real strengths."
            ),
        )
    return make_signal("content.skills_section", "Skills section is scannable", 1.0, weight=1.5)


def _tense_consistency(resume: ParsedResume, bullets: list[str]) -> Signal:
    """Past roles in past tense; the current role may use present tense."""
    if len(bullets) < MIN_BULLETS_FOR_RATIOS or not resume.experience:
        return _insufficient("content.tense", "Verb tense is consistent", 1.0)

    past_roles = [e for e in resume.experience if e.dates and not e.dates.is_current]
    # A finished role described with a gerund ("Building the pipeline...") is the
    # one tense error frequent enough - and unambiguous enough - to flag.
    offenders = [
        bullet
        for entry in past_roles
        for bullet in entry.bullets
        if _ING_OPENER_RE.match(bullet.strip())
    ]

    checked = sum(len(e.bullets) for e in past_roles)
    if not checked:
        return _insufficient("content.tense", "Verb tense is consistent", 1.0)

    score = 1.0 - len(offenders) / checked
    return make_signal(
        "content.tense", "Verb tense is consistent", score, weight=1.0,
        evidence=[f'"{collapse(b)[:110]}"' for b in offenders[:3]],
        recommendation=(
            'Use past tense for finished roles ("Built", not "Building"). Present tense '
            "belongs only in your current position."
        ),
    )
