# ResumeIQ — AI Resume Intelligence & Job Matching

A production-grade resume analysis and job-matching service. It scores a resume
across four dimensions, returns **every signal that produced the score**, and
ranks the fixes by how many points each one recovers.

The core principle: **scoring is deterministic and the LLM cannot change it.**
A resume gets the same number every time, from explainable rules you can read in
`app/core/scoring/`. Claude sits on top as a coaching layer that explains the
findings and drafts rewrites. If there is no API key, or the provider is down,
or the model returns something malformed, the service still returns a complete
analysis and says why the narrative is missing.

```
                   ┌────────────────────────────────────────────┐
  PDF / DOCX / TXT │  extract → parse → skills → score          │  deterministic
  ────────────────►│  (pure, no I/O, same input → same output)  │  0-100 + signals
                   └────────────────────┬───────────────────────┘
                                        │  analysis (trusted)
                                        ▼
                   ┌────────────────────────────────────────────┐
                   │  guards → prompt → Claude → validate       │  optional
                   │  redact PII · fence untrusted · block      │  narrative only
                   │  fabricated metrics · schema-check         │
                   └────────────────────────────────────────────┘
```

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -e ".[dev,ui]"
```

Run the API:

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Run the dashboard in a second terminal:

```bash
.venv/Scripts/python.exe -m streamlit run ui/streamlit_app.py
```

The API is on <http://localhost:8000> (docs at `/docs`), the UI on
<http://localhost:8501>.

Score a resume without any server at all:

```bash
.venv/Scripts/python.exe -m app.cli analyze tests/fixtures/strong_resume.txt --job tests/fixtures/backend_job.txt
```

```
==========================================================================
strong_resume.txt  ->  88.4/100 (excellent)
==========================================================================
  ATS compatibility           97.7  ####################  [info]
  Structure & completeness    94.9  ###################.  [info]
  Content quality            100.0  ####################  [ok]
  Job match                   70.7  ##############......  [warning]

  Required-skill coverage: 77%
  Missing required skills: Incident Response, Leadership, Load Balancing
```

### Enabling AI coaching

```bash
cp .env.example .env          # then set ANTHROPIC_API_KEY
```

Everything works without it. The `ai_feedback` block simply reports
`available: false` and the reason.

---

## What it actually checks

**32 individual signals** across four dimensions. Each one carries a 0–1 score,
a weight, the evidence that produced it, and a concrete recommendation.

| Dimension | Weight¹ | Signals | Examples |
|---|---|---|---|
| **Job match** | 35% | 6 | required-skill coverage (weighted by evidence), preferred skills, posting keywords, language overlap, seniority, years of experience |
| **ATS compatibility** | 25% | 10 | text layer, multi-column layout, tables, header/footer content, parseable contact details, standard headings, date formats, encoding damage |
| **Content quality** | 25% | 9 | quantified achievements, action verbs, filler phrasing, bullet length, verb variety, first person, buzzwords, tense |
| **Structure** | 15% | 7 | core sections present, length, section order, per-role detail, education, summary, professional links |

¹ Weights when a job description is supplied. Without one, the match dimension
is dropped and the rest are renormalised to 40/35/25.

Three ideas do most of the work:

- **Evidence beats keywords.** A skill demonstrated inside a work bullet scores
  higher than the same word in a skills list. Two resumes with identical keyword
  sets get different match scores.
- **Recommendations are ranked by recovered points**, not by severity alone, so
  the top item is genuinely the best use of the candidate's time.
- **Layout defects are captured during extraction**, before the document is
  flattened to text — you cannot detect a two-column layout after the fact.

See [docs/SCORING.md](docs/SCORING.md) for the full methodology and every
formula.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/analyses` | Analyse an uploaded resume (multipart) |
| `POST` | `/api/v1/analyses/text` | Analyse raw resume text |
| `GET` | `/api/v1/analyses` | Paginated history |
| `GET` | `/api/v1/analyses/{id}` | Fetch a stored analysis |
| `DELETE` | `/api/v1/analyses/{id}` | Delete one |
| `POST` | `/api/v1/jobs/parse` | Inspect how a posting was interpreted |
| `GET` | `/health`, `/health/ready` | Liveness / readiness |
| `GET` | `/metrics`, `/metrics/json` | Prometheus and JSON metrics |

```bash
curl -s -X POST http://localhost:8000/api/v1/analyses \
  -F "file=@resume.pdf" \
  -F "job_description=$(cat job.txt)" | jq '.overall_score, .recommendations[0]'
```

Full request/response reference: [docs/API.md](docs/API.md).

---

## LLM safety

The resume is attacker-controlled text, so the coaching layer treats it that
way. Four guards, all tested offline against a stubbed client:

1. **Prompt-injection containment** — untrusted content is fenced with a
   per-request random nonce, tag markup is neutralised, and steering attempts
   are detected and *reported* rather than silently stripped.
2. **PII redaction** — name, email, phone, links and addresses are replaced with
   placeholders before the text leaves the process, and restored in the
   response. The model never sees the candidate's identity.
3. **Fabrication blocking** — any number in a suggested rewrite that does not
   appear in the candidate's own text causes the rewrite to be dropped. A resume
   tool that invents achievements is worse than one that says nothing.
4. **Schema validation** — output is constrained by a strict JSON schema and
   re-validated with Pydantic; anything that fails is discarded, not shown.

Plus a circuit breaker, explicit timeouts, and prompt caching on the frozen
system prompt. Details in [docs/LLM_SAFETY.md](docs/LLM_SAFETY.md).

---

## Testing

```bash
.venv/Scripts/python.exe -m pytest -q                     # 221 tests
.venv/Scripts/python.exe -m pytest -q --cov=app           # 92% coverage
.venv/Scripts/ruff.exe check app tests ui                 # lint
```

| Suite | What it covers |
|---|---|
| `test_extraction.py` | Magic-byte detection, size limits, DOCX tables/headers, corrupt files |
| `test_parsing.py` | Sections, contact block, date formats, experience/education entities |
| `test_skills.py` | Taxonomy, span masking, context confidence, fuzzy recovery, JD parsing |
| `test_scoring.py` | Every signal, aggregation, determinism, degenerate input |
| `test_matching.py` | TF-IDF properties, coverage, gaps, seniority, evidence weighting |
| `test_llm.py` | Injection detection, redaction round-trip, fabrication, every failure mode |
| `test_api.py` | HTTP contract, error envelope, persistence, rate limiting |
| `test_pdf_and_cli.py` | Real generated PDFs end to end, CLI report and JSON output |
| `test_ui.py` | Streamlit dashboard against a live API, in-process |

No test touches the network. The Anthropic client is stubbed so every provider
failure — refusal, rate limit, timeout, malformed JSON, open circuit — is
reproducible offline.

---

## Deployment

```bash
docker compose up -d --build     # API on :8000, UI on :8501
```

Both images run as non-root with healthchecks. SQLite is the default store;
point `RESUMEIQ_DATABASE_URL` at Postgres for multi-replica deployments. See
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for scaling notes, the readiness
contract, and what to change before going to production.

---

## Project layout

```
app/
  main.py            FastAPI factory, middleware, lifespan
  config.py          Typed settings; cached once per process
  errors.py          Error taxonomy + the single JSON envelope
  observability.py   Metrics registry, timing, request-ID middleware
  api/               Routes and the rate-limit dependency
  core/
    extraction/      PDF/DOCX/TXT → text + layout diagnostics
    parsing/         Sections, contact, dates, entities, job postings
    skills/          157-skill ontology (data/skills.json) + matcher
    embeddings/      Pure-Python TF-IDF and cosine similarity
    scoring/         ats · content · matching · aggregate
    llm/             guards · prompts · client · service
  db/                SQLAlchemy models and repository functions
  schemas/           Pydantic API contract + domain→API mapping
ui/                  Streamlit dashboard (thin HTTP client)
docs/                Architecture, scoring, LLM safety, API, deployment
```

Roughly 6,600 lines of application code and 2,000 lines of tests.

---

## Design decisions worth knowing

**No embedding model.** A resume/JD pair is two short documents. A dense model
costs a download, a cold start and often a network call, and buys little over
well-tuned lexical similarity — which is also *explainable*: the API returns the
exact terms that drove the score. TF-IDF is implemented in ~150 lines with no
dependencies (`app/core/embeddings/vectorizer.py`).

**The skill ontology is data, not code.** `data/skills.json` holds 157 skills
with aliases and adjacency. A domain expert can extend it without touching the
matcher, and the "you already have X, which is adjacent to the missing Y"
guidance falls out of the `related` edges.

**Two scoring dimensions for what looks like one problem.** ATS compatibility
and structure are separate because the fixes are unrelated: one is "export the
file differently", the other is "write the missing section".

**Persistence never fails a request.** If the database is down, the analysis the
caller is waiting for is still returned; the write failure is logged.
