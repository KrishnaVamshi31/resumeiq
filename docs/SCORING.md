# Scoring methodology

Every number this service produces is traceable to a rule you can read. This
document is the specification; `app/core/scoring/` is the implementation, and
`tests/test_scoring.py` pins the behaviour.

## The shape of a score

Three levels, all weighted means:

```
Signal   0.0 - 1.0    one atomic check + weight + evidence + recommendation
   ↓
Dimension  0 - 100    weighted mean of its signals
   ↓
Overall    0 - 100    weighted mean of the dimensions
```

A `Signal` (`app/core/scoring/base.py`) always carries:

| Field | Meaning |
|---|---|
| `score` | 0–1, clamped |
| `weight` | Relative importance inside its dimension |
| `severity` | Derived from score: `<0.34` critical, `<0.67` warning, `<0.95` info, else ok |
| `evidence` | The specific lines or facts that produced the score |
| `recommendation` | What to do about it — `None` when the signal passes |

A signal that scores ≥ 0.95 carries no recommendation. This is enforced in
`make_signal()` so a passing check can never generate advice.

## Dimension weights

With a job description:

| Dimension | Weight |
|---|---|
| Job match | 0.35 |
| ATS compatibility | 0.25 |
| Content quality | 0.25 |
| Structure & completeness | 0.15 |

Without one, the match dimension does not exist and the remainder are
renormalised to Content 0.40, ATS 0.35, Structure 0.25. Content is weighted
highest in standalone review because with no target role, *how well the resume
is written* is the only thing left to judge.

Bands: `≥85 excellent`, `≥70 strong`, `≥55 fair`, `≥40 weak`, else `poor`.

---

## Job match (6 signals)

The question: if a recruiter compared this resume against this posting, how much
of what they asked for would they find?

| Signal | Weight | Scoring |
|---|---|---|
| `match.required_skills` | 5.0 | Weighted coverage of must-haves × evidence multiplier |
| `match.preferred_skills` | 1.5 | Plain coverage of nice-to-haves |
| `match.keywords` | 1.5 | Share of recurring posting terms present |
| `match.similarity` | 1.0 | TF-IDF cosine, scaled |
| `match.seniority` | 1.0 | `1 − 0.3 × rank distance` |
| `match.experience` | 1.0 | Shortfall against the stated minimum |

Required skills carry **more than half** the dimension's weight deliberately: a
missing must-have is the thing that actually gets a resume rejected.

### Required-skill coverage

```
score = Σ(requirement.weight × evidence_multiplier) / Σ(requirement.weight)
```

`requirement.weight = min(1.0, 0.7 + 0.15 × (mentions − 1))` — a skill named
three times in the requirements is more central than one named once, capped so a
keyword-stuffed post cannot let one term dominate.

`evidence_multiplier` is where this differs from keyword matching:

| Where the skill appears in the resume | Multiplier |
|---|---|
| Demonstrated in experience, projects, summary or certifications | 1.00 |
| Listed only in a skills section | 0.75 |
| Absent | 0.00 |

Two resumes with identical keyword sets therefore score differently.
`tests/test_matching.py::test_demonstrated_skills_beat_keyword_lists` asserts it.

### Keyword coverage

Job posts contain domain terms outside any fixed ontology ("claims
adjudication", "FDA submissions"), and those are the literal strings ATS filters
query. They are extracted from the requirements and responsibilities buckets,
weighted 2× for requirements, with acronyms boosted and boilerplate
("experience", "years", "strong", "degree"…) removed. Redundant sub-phrases are
dropped so one concept does not occupy three slots.

Coverage is **concept-level, not verbatim**: a keyword counts as present when
≥60% of its content words appear in the resume, compared on a 5-character prefix
so "designing"/"designed" match. Requiring exact phrases would mark every
candidate as missing every keyword.

### Similarity

TF-IDF cosine over unigrams and bigrams, L2-normalised, sublinear term frequency
(`1 + log tf`) so repeating "Python" nine times is not nine times more relevant.
IDF is estimated from the bullets and sentences of both documents, so resume
boilerplate is down-weighted automatically.

Resume and job-post prose differ in register even for a perfect candidate, so
the raw cosine is low by nature; the signal scales it as `min(1, cosine / 0.45)`
and carries the **lowest weight in the dimension**. It is a tiebreaker, never a
verdict.

### Gap analysis

Every missing requirement becomes a `SkillGap` with a priority:

| Priority | Condition |
|---|---|
| high | Required, and no adjacent skill owned |
| medium | Required, but an adjacent skill *is* owned |
| low | Preferred |

Adjacency comes from the `related` edges in `data/skills.json`. Owning Docker
softens a Kubernetes gap to medium — because that is a gap the candidate can
credibly speak to in an interview, not a disqualification.

---

## ATS compatibility (10 signals)

Machine readability of the file **as submitted**. Every check maps to a
documented way applicant tracking systems mangle a resume.

| Signal | Weight | Fails when |
|---|---|---|
| `ats.text_layer` | 3.0 | Pages yield no extractable text (a scan), or density is far below ~1200 chars/page |
| `ats.contact` | 3.0 | No email *and* no phone — scored 0 and marked critical |
| `ats.layout` | 2.5 | Line-length distribution indicates multi-column or a sidebar |
| `ats.header_footer` | 2.0 | Content sits in the DOCX header/footer region |
| `ats.headings` | 2.0 | Section headings do not map to the standard vocabulary |
| `ats.dates` | 2.0 | Roles have no parseable date range |
| `ats.file_format` | 1.5 | Plain text, or a PDF with encryption/permission flags |
| `ats.tables` | 1.5 | Tables used for layout (0.5 for ≤2, 0.2 above) |
| `ats.encoding` | 1.0 | Replacement characters or a very high non-ASCII ratio |
| `ats.bullets` | 1.0 | Experience written as prose with no bullets |

**These are measured during extraction, not after.** Column layout, tables,
embedded images and header/footer regions are all invisible once a document is
flattened to text, so `ExtractedDocument` carries them forward
(`app/core/extraction/`). This is why the `/analyses/text` endpoint explicitly
reports that layout checks could not be evaluated rather than silently passing
them.

Non-ASCII is treated carefully: accented names and non-Latin scripts are
legitimate, so only a ratio above 25% is flagged, and only replacement
characters and C0 controls count as real damage.

---

## Content quality (9 signals)

What separates a resume that *passes* screening from one that *wins* an
interview.

| Signal | Weight | Scoring |
|---|---|---|
| `content.quantification` | 3.0 | `min(1, quantified_ratio / 0.4)` |
| `content.action_verbs` | 2.5 | `(strong_ratio + 0.25) − 0.5 × weak_ratio` |
| `content.filler` | 2.0 | `1 − occurrences / max(bullets, 5)` |
| `content.bullet_length` | 1.5 | Share of bullets in the 10–28 word band |
| `content.verb_variety` | 1.5 | `min(1, unique_ratio / 0.75)` |
| `content.skills_section` | 1.5 | Present, listed not prose, 5–45 items |
| `content.first_person` | 1.0 | `1 − pronouns / 8`, summary exempt |
| `content.buzzwords` | 1.0 | `1 − 0.2 × distinct buzzwords` |
| `content.tense` | 1.0 | Gerund openers in finished roles |

Three deliberate choices:

**40% quantified is full marks, not 100%.** Not every achievement has a number,
and demanding one produces invented metrics. The ratio scales to a realistic
ceiling.

**Unknown verbs are neither rewarded nor penalised.** The action-verb list
cannot be exhaustive, so only verbs on the *weak* list ("responsible", "worked",
"helped", "assisted"…) are penalised. A false penalty is worse than a missed
reward.

**Too little text scores 0.5, not 0.** A resume with two bullets is already
penalised by the structure dimension; scoring its content ratios at zero would
punish the same defect twice. Signals return a neutral 0.5 with an explanation
when there is not enough material to judge.

---

## Structure & completeness (7 signals)

| Signal | Weight | Fails when |
|---|---|---|
| `structure.core_sections` | 3.0 | Experience, Education or Skills missing |
| `structure.experience_detail` | 2.5 | Roles have fewer than 2 bullets |
| `structure.length` | 2.0 | Outside 400–1000 words |
| `structure.summary` | 1.0 | No summary, or over 120 words |
| `structure.education` | 1.0 | No parseable degree |
| `structure.links` | 1.0 | No LinkedIn/GitHub/portfolio |
| `structure.order` | 1.0 | Education above Experience with ≥2 years of history |

Length scoring is piecewise: full marks in 400–1000 words, tapering to 0.5 at
250, and down to 0.3 above 1400. Under 250 words scores 0.2.

---

## Ranking recommendations

Severity alone is a poor ordering — a critical signal with a tiny weight matters
less than a warning that costs ten points. Each recommendation's impact is the
number of **overall** points recovered by taking that signal to full marks:

```
impact = (1 − signal.score)
       × (signal.weight / Σ signal weights in dimension)
       × (dimension.weight / Σ dimension weights)
       × 100
```

Recommendations sort by severity first (so a blocking defect is never buried),
then by impact descending, then by id for deterministic ordering.

## Determinism

`analyze()` performs no I/O beyond loading the cached taxonomy and never calls a
model. The same document produces the same score, the same signals and the same
recommendation order on every run — asserted by
`tests/test_scoring.py::test_analysis_is_deterministic`.

This is what makes the scores defensible to a candidate who disagrees with them,
and what makes the test suite meaningful.

## Calibrating for your own use

The three places to adjust, in order of likely need:

1. **`data/skills.json`** — the ontology. Pure data; no code change needed.
2. **`WEIGHTS_WITH_JOB` / `WEIGHTS_WITHOUT_JOB`** in `base.py` — dimension weights.
3. **Signal weights** — the `weight=` argument at each `make_signal()` call site.

Every threshold named in this document is a module-level constant with a comment
explaining the number, not a literal buried in a function.
