# API reference

Base URL: `http://localhost:8000`. The generated OpenAPI document lives at
`/openapi.json`, with Swagger UI at `/docs` outside production.

Every response carries an `x-request-id` header. Send your own to propagate a
trace id; it is echoed back and appears in every log line for that request.

---

## `POST /api/v1/analyses`

Analyse an uploaded resume. `multipart/form-data`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | PDF, DOCX, TXT or MD. Default limit 5 MiB. |
| `job_description` | string | no | Full posting text. Unlocks the match dimension. |
| `job_title` | string | no | Overrides the inferred title. |
| `company` | string | no | Stored with the analysis. |
| `use_ai` | bool | no | Default `true`. Set `false` to skip the coaching call. |

```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -F "file=@resume.pdf" \
  -F "job_description=$(cat job.txt)" \
  -F "use_ai=true"
```

Returns `201` with the full analysis document described below.

## `POST /api/v1/analyses/text`

Identical, but takes `resume_text` as a form field instead of a file. Convenient
for integrations that already hold the text.

Layout-dependent ATS checks (columns, tables, header regions) cannot fire on raw
text. Rather than silently passing them, the response states this in
`document.warnings`.

## `GET /api/v1/analyses`

Paginated history, most recent first.

| Query | Default | Range |
|---|---|---|
| `limit` | 25 | 1–100 |
| `offset` | 0 | ≥ 0 |

```json
{
  "items": [
    {
      "id": "9f2c…",
      "created_at": "2026-09-04T13:34:41.203Z",
      "filename": "resume.pdf",
      "overall_score": 88.4,
      "job_title": "Senior Backend Engineer, Payments",
      "company": "Northwind Financial",
      "llm_used": false
    }
  ],
  "total": 12, "limit": 25, "offset": 0
}
```

## `GET /api/v1/analyses/{id}`

Returns the stored analysis — byte-identical to what was originally returned.

## `DELETE /api/v1/analyses/{id}`

`204` on success, `404` if it does not exist. Deliberately not idempotent, so a
client can tell the difference between "deleted" and "was never there".

## `POST /api/v1/jobs/parse`

Inspect how a posting was interpreted before scoring against it. Parsing
mistakes are far easier to spot here than inside a composite score.

```json
{ "text": "…posting…", "title": null, "company": null }
```

```json
{
  "job": {
    "title": "Senior Backend Engineer, Payments",
    "seniority": "senior",
    "min_years_experience": 6,
    "degree_required": "Bachelor's",
    "required_skills": ["Python", "PostgreSQL", "Kubernetes", "Docker", "Kafka", "AWS"],
    "preferred_skills": ["Go", "Terraform", "Rust", "Observability"],
    "keywords": ["professional backend engineering", "designing event-driven systems"]
  },
  "responsibilities": ["Design, build and operate distributed backend services…"]
}
```

## Ops endpoints

| Path | Purpose |
|---|---|
| `GET /health` | Liveness. Touches nothing. Always `200` while the process is up. |
| `GET /health/ready` | Readiness. Checks database and taxonomy; reports LLM status. `503` only on a real dependency failure — **an LLM outage never fails readiness.** |
| `GET /metrics` | Prometheus text format. |
| `GET /metrics/json` | The same counters and histograms as JSON. |

```json
{
  "status": "ok", "version": "1.0.0", "environment": "production",
  "checks": { "database": "ok", "taxonomy": "ok", "llm": "unconfigured" }
}
```

`llm` is one of `ok`, `disabled`, `unconfigured`, `circuit_open`.

---

## The analysis document

```jsonc
{
  "id": "9f2c…",
  "created_at": "2026-09-04T13:34:41.203Z",
  "overall_score": 88.4,
  "band": "excellent",              // excellent | strong | fair | weak | poor
  "summary": "Overall 88.4/100 (excellent). Covers 77% of the required skills…",

  "document": {
    "filename": "resume.pdf", "source_format": "pdf",
    "page_count": 1, "word_count": 314, "char_count": 2149,
    "warnings": []
  },

  "contact":  { "name": "…", "email": "…", "phone": "…", "location": "…",
                "linkedin": "…", "github": "…", "website": null },
  "years_of_experience": 10.2,

  "experience": [
    { "title": "Senior Software Engineer", "organization": "Razorline Technologies",
      "dates": { "start_year": 2021, "start_month": 3, "is_current": true,
                 "raw": "Mar 2021 - Present" },
      "bullet_count": 4, "bullets": ["Rebuilt the payment settlement pipeline…"] }
  ],
  "education": [
    { "degree": "B.Tech", "field_of_study": "Computer Science",
      "institution": "National Institute of Technology Trichy", "gpa": 3.44 }
  ],

  "skills": [
    { "name": "Kubernetes", "category": "devops", "confidence": 1.0,
      "mentions": 3, "demonstrated": true, "evidence": ["…migration of 38 microservices…"] }
  ],

  "dimensions": [
    { "id": "match", "label": "Job match", "score": 70.7, "weight": 0.35,
      "severity": "warning",
      "signals": [
        { "id": "match.required_skills", "label": "Required skills covered",
          "score": 0.77, "weight": 5.0, "severity": "warning",
          "detail": "10 of 13 required skills evidenced.",
          "evidence": ["Missing required skill: Incident Response"],
          "recommendation": "Add the missing required skills where you genuinely have them…" }
      ] }
  ],

  "recommendations": [
    { "id": "match.required_skills", "dimension": "match", "severity": "warning",
      "title": "Required skills covered",
      "action": "Add the missing required skills where you genuinely have them…",
      "evidence": ["Missing required skill: Incident Response"],
      "impact_points": 3.65 }
  ],
  "strengths": ["Achievements are quantified", "Single-column layout"],

  "job":   { "title": "…", "seniority": "senior", "min_years_experience": 6, "…": "…" },
  "match": {
    "score": 70.7, "required_coverage": 0.77,
    "matched_required": ["AWS", "Docker", "Kafka", "Kubernetes", "PostgreSQL", "Python"],
    "gaps": [
      { "name": "Incident Response", "category": "devops", "required": true,
        "priority": "medium", "adjacent_owned": ["Site Reliability Engineering"] }
    ],
    "missing_keywords": ["production postgresql"],
    "overlapping_terms": ["kafka", "kubernetes"],
    "surplus_skills": ["Airflow", "Redis"]
  },

  "ai_feedback": {
    "available": false,
    "status": "AI coaching is unavailable: no ANTHROPIC_API_KEY is configured…",
    "content": null
  }
}
```

### Reading it

- **`recommendations`** is the field to render first. It is already sorted:
  critical severity first, then by `impact_points` (overall points recovered by
  fixing that signal), then by id for stable ordering.
- **`dimensions[].signals`** contains every check, including the ones that
  passed, so a UI can show the full audit rather than only complaints.
- **`skills[].demonstrated`** distinguishes a skill shown in a work bullet from
  one listed in a keyword dump. It is what drives the match score's evidence
  weighting.
- **`match.gaps[].adjacent_owned`** is how to frame a gap constructively: the
  candidate has a related skill and can speak to it.
- **`ai_feedback.status`** always explains itself. Render it whenever
  `available` is `false` rather than showing an empty panel.

### When AI coaching succeeded

```jsonc
"content": {
  "headline": "Strong backend profile that undersells its scale.",
  "strengths": ["Quantified outcomes throughout"],
  "priority_fixes": [
    { "target": "Summary", "problem": "…", "fix": "…", "why_it_matters": "…" }
  ],
  "bullet_rewrites": [
    { "original": "…", "improved": "…", "rationale": "…" }
  ],
  "summary_rewrite": "…",
  "keyword_guidance": ["event-driven systems"],
  "interview_risks": ["Short tenure at first employer"],
  "integrity_notes": [],
  "dropped_rewrites": ["Dropped a rewrite asserting unverifiable figures (73%)…"],
  "injection_findings": [],
  "model": "claude-opus-5",
  "usage": { "input_tokens": 1200, "output_tokens": 400,
             "cache_read_tokens": 900, "latency_ms": 850.0 },
  "request_id": "req_…"
}
```

`dropped_rewrites` and `injection_findings` should be surfaced, not hidden —
they are the audit trail for the safety guards.

---

## Errors

One envelope for every failure:

```json
{
  "error": {
    "code": "unsupported_file_type",
    "message": "Legacy .doc files are not supported and parse poorly in most ATS. Re-save as .docx or PDF.",
    "details": {},
    "request_id": "3f8a2c1e9b7d4a05"
  }
}
```

| Code | Status | Cause |
|---|---|---|
| `validation_error` | 422 | Request failed schema validation; `details.errors` has specifics |
| `empty_document` | 422 | Uploaded file or text is empty |
| `extraction_failed` | 422 | Corrupt file, or a PDF with no text layer |
| `unsupported_file_type` | 415 | Not PDF/DOCX/TXT — message explains what to do |
| `file_too_large` | 413 | Over `RESUMEIQ_MAX_UPLOAD_BYTES`; `details` has size and limit |
| `not_found` | 404 | No analysis with that id |
| `rate_limited` | 429 | `details.retry_after_seconds` |
| `llm_unavailable` | 503 | Only when AI output was explicitly demanded |
| `internal_error` | 500 | Generic; correlate via `request_id` |

Error messages are written for the end user, not the developer. An unsupported
file type explains how to re-save it; a missing text layer explains that the PDF
is probably a scan.

## Rate limiting

Sliding window, default 60 requests per 60 seconds per client, applied to the
analysis and job-parsing endpoints. Health and history endpoints are not limited.

The window is in-process and therefore **per replica** — adequate as a second
line of defence behind an edge limiter. For multi-replica deployments move it to
Redis; `SlidingWindowLimiter` in `app/api/deps.py` is a drop-in interface.
