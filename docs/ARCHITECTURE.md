# Architecture

## The organising principle

There are two systems here, and the boundary between them is the most important
thing in the codebase.

```
┌──────────────────────────────────────────────────────────────┐
│  DETERMINISTIC CORE                                          │
│  pure functions · no network · same input → same output      │
│  produces: scores, signals, evidence, ranked recommendations │
└──────────────────────────────────────────────────────────────┘
                              │
                              │  AnalysisResult (trusted input)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  COACHING LAYER (optional)                                   │
│  guards · Claude · validation                                │
│  produces: narrative, rewrites — never a number              │
└──────────────────────────────────────────────────────────────┘
```

The coaching layer consumes the analysis and cannot modify it. `generate_feedback()`
returns a `FeedbackOutcome` that is either feedback or a reason there is none —
it raises nothing. Removing the LLM entirely would not change a single score.

This matters for three reasons: scores stay defensible to a candidate who
disagrees with them, the test suite can assert exact numbers, and a provider
outage degrades the product instead of breaking it.

## Request flow

```
POST /api/v1/analyses
  │
  ├─ RequestContextMiddleware      assign request-id, start timer
  ├─ rate_limit dependency          sliding window per client
  │
  ├─ extract()          [threadpool]  magic bytes → PDF/DOCX/TXT reader
  │                                   captures layout defects before flattening
  ├─ analyze()          [threadpool]  parse → skills → 4 scorers → rank fixes
  ├─ generate_feedback()  [async]     redact → fence → Claude → validate
  ├─ save_analysis()    [threadpool]  best-effort; never fails the request
  │
  └─ to_response()                    domain dataclasses → Pydantic contract
```

CPU-bound work runs in the threadpool so a 40-page PDF cannot block the event
loop while an LLM call is in flight.

## Layers

### `app/core/extraction/`

Bytes in, `ExtractedDocument` out. Format detection is driven by **magic bytes
first** — the filename and browser-supplied content type are both
attacker-controlled and routinely wrong.

The critical design point: this layer captures more than text. Page-level text
density, line-length distribution (which reveals multi-column layouts), table
counts, embedded images and header/footer content are all recorded here, because
they are unrecoverable once the document becomes a string. Downstream ATS
scoring reads them off the `ExtractedDocument`.

Legacy `.doc` and `.rtf` are rejected with guidance rather than parsed — they
parse poorly in real ATS systems, so refusing them is itself useful advice.

### `app/core/parsing/`

`ExtractedDocument.text` → `ParsedResume`: sections, contact block, experience
entries with bullets attached, education, computed tenure.

Section segmentation is the load-bearing piece. Headings are free text and all
visual cues (bold, size) are gone, so headings are identified *structurally*
(short, standalone, title- or upper-cased) and then mapped onto a canonical
vocabulary. Two failure modes are guarded explicitly:

- A body line containing a vocabulary word must not be read as a heading. A
  loose match requires the line to look like a heading *and* the matched phrase
  to cover ≥50% of it. Without this, `Senior Engineer | Razorline Technologies |
  Mar 2021 – Present` matches the skills vocabulary on "technologies" and
  silently truncates the entire Experience section.
- The first line of a resume is the candidate's name, which looks exactly like a
  heading. Treating it as one discards the whole contact block below it.

Content under an *unrecognised* heading is kept as `SectionKind.OTHER` rather
than dropped, so skills and writing quality in creatively-named sections are
still analysed.

`jobdesc.py` is the mirror image for postings: it routes lines into
required / preferred / responsibilities / ignored buckets, which is what makes
"missing a must-have" cost more than "missing a nice-to-have".

### `app/core/skills/`

A 157-skill ontology in `data/skills.json` — **data, not code** — with aliases
and adjacency edges, indexed once at import into surface-form → canonical maps.

The matcher scans with **span masking**: patterns are sorted longest-first and
matched characters are consumed, so "React Native" claims its span and the
"React" pattern cannot re-match it. `C++` does not register as `C`.

Mentions are labelled by context (experience, projects, skills, …), which feeds
the confidence score and the evidence multiplier in job matching. A light fuzzy
pass over comma-delimited skill lists recovers typos like "Kubernets" at reduced
confidence.

### `app/core/embeddings/`

Pure-Python TF-IDF and cosine similarity, no dependencies.

A resume/JD pair is two short documents. A dense embedding model costs a
download, a cold start, GPU or an outbound call per request, and buys little
over well-tuned lexical similarity — which is additionally **explainable**: the
API returns the exact overlapping terms that drove the score. `top_overlapping_terms()`
and `missing_terms()` exist for that reason.

### `app/core/scoring/`

Four scorers producing `DimensionScore` objects, aggregated in `aggregate.py`.
`analyze()` is the single entry point used by the API, the CLI and the tests, so
there is exactly one code path to reason about. Full methodology in
[SCORING.md](SCORING.md).

### `app/core/llm/`

Four modules with one responsibility each:

| Module | Responsibility |
|---|---|
| `guards.py` | Injection detection, PII redaction, fabrication checking |
| `prompts.py` | Frozen system prompt + strict JSON schema + message assembly |
| `client.py` | Anthropic SDK wrapper: timeouts, retries, circuit breaker, token accounting |
| `service.py` | Orchestration and graceful degradation |

The prompt is built in two halves that never mix: the **trusted** half (system
prompt + the analysis this service computed) and the **untrusted** half (resume
and posting, fenced with a per-request nonce). See [LLM_SAFETY.md](LLM_SAFETY.md).

### `app/db/`

SQLAlchemy 2.0, synchronous sessions called through `run_in_threadpool`. SQLite
by default; the repository functions are the only thing call sites touch, so
moving to Postgres is a URL change.

The full API response is stored as JSON alongside indexed scalar columns, so
history replays exactly what was shown without a migration for every new field.

### `app/schemas/`

The API contract, deliberately separate from the domain dataclasses. The scoring
engine can be refactored without silently changing the public shape, and the
OpenAPI document describes the contract rather than the internals.

## Cross-cutting concerns

**Errors.** One taxonomy (`app/errors.py`), one JSON envelope, always carrying
the request id. Unhandled exceptions log a full traceback and return a generic
message — internals never reach the client.

**Observability.** Structured JSON logs with a `ContextVar` request id, so a log
line emitted deep inside the scoring code correlates without threading a logger
through every function. A dependency-free metrics registry exposes counters and
histograms at `/metrics` in Prometheus format. `track()` is the single primitive
for timing a unit of work.

**Configuration.** One typed `Settings` object, parsed once and cached.
`settings.llm_available` is the single source of truth for whether coaching can run.

**Readiness.** `/health` is cheap and touches nothing. `/health/ready` checks the
database and taxonomy and reports LLM status — but **an LLM outage never fails
readiness**, because the service is fully functional without it and must not be
pulled from the load balancer.

## Known limits

- The rate limiter and metrics registry are in-process, so both are per-replica.
  Behind multiple replicas, move the limiter to Redis (the interface does not
  change) and aggregate metrics at the scrape layer.
- Seniority and title parsing are tuned for English-language,
  technology-sector conventions. The taxonomy covers business, finance,
  marketing, design and healthcare terms, but title heuristics are weakest
  outside tech.
- Tenure is computed from parsed date ranges; a resume with no dates reports
  zero years, which is surfaced as an ATS defect rather than silently assumed.
- `POST /analyses/text` cannot evaluate layout-dependent ATS signals. The
  response says so in `document.warnings` instead of passing them by default.
