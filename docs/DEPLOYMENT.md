# Deployment

## Docker Compose (single host)

```bash
cp .env.example .env          # optionally set ANTHROPIC_API_KEY
docker compose up -d --build
```

API on `:8000`, dashboard on `:8501`. Both images run as non-root
(uid 10001/10002) with healthchecks; the API image contains no build toolchain
and the database lives on a named volume so it survives container replacement.

```bash
docker compose logs -f api
docker compose down            # add -v to drop the data volume
```

## Without Docker

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

One worker per process. Scale with replicas rather than `--workers`, so the
in-process rate limiter and metrics registry stay coherent per instance.

## Configuration

Everything is environment-driven; `.env.example` is the full list.

| Variable | Default | Notes |
|---|---|---|
| `RESUMEIQ_ENV` | `development` | `production` disables `/docs` and `/redoc` |
| `RESUMEIQ_LOG_FORMAT` | `json` | `console` for local readability |
| `RESUMEIQ_LOG_LEVEL` | `INFO` | |
| `RESUMEIQ_DATABASE_URL` | `sqlite:///./data/resumeiq.db` | Any SQLAlchemy URL |
| `RESUMEIQ_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated |
| `RESUMEIQ_MAX_UPLOAD_BYTES` | `5242880` | 5 MiB |
| `RESUMEIQ_MAX_RESUME_CHARS` | `60000` | Oversized documents are truncated, not rejected |
| `RESUMEIQ_RATE_LIMIT_REQUESTS` | `60` | Per window, per client, per replica |
| `RESUMEIQ_RATE_LIMIT_WINDOW_SECONDS` | `60` | |
| `ANTHROPIC_API_KEY` | unset | Absent ⇒ coaching skipped and reported |

LLM settings are documented in [LLM_SAFETY.md](LLM_SAFETY.md#configuration).

## Before production

- [ ] **Set `RESUMEIQ_ENV=production`.** Disables interactive docs.
- [ ] **Move off SQLite** if running more than one replica. Set
      `RESUMEIQ_DATABASE_URL` to Postgres; no code changes required.
- [ ] **Lock `RESUMEIQ_CORS_ORIGINS`** to your real front-end origin.
- [ ] **Add authentication.** There is none — the service is designed to sit
      behind your gateway or an auth middleware. Do not expose it directly.
- [ ] **Add an edge rate limiter.** The built-in one is per-replica and is a
      second line of defence, not the first.
- [ ] **Set the upload limit at the proxy too.** The app enforces
      `RESUMEIQ_MAX_UPLOAD_BYTES`, but the request body still reaches it first.
- [ ] **Decide on resume retention.** `AnalysisRecord.resume_text` stores the
      extracted text so an analysis can be re-run against a new posting. If your
      privacy posture does not allow that, null the column and add a retention
      job — this is the main data-protection decision in the system.
- [ ] **Confirm `X-Forwarded-For` is set by your proxy** before relying on it
      for client identification (`app/api/deps.py`).
- [ ] **Point the LLM at a budget.** Coaching is the only per-request cost;
      `use_ai=false` disables it per request, `RESUMEIQ_LLM_ENABLED=false`
      globally.

## Health and readiness

| Probe | Endpoint | Behaviour |
|---|---|---|
| Liveness | `GET /health` | Touches nothing. Restart the pod if this fails. |
| Readiness | `GET /health/ready` | Checks database and taxonomy. `503` on real failure. |

**An LLM outage must never fail readiness.** The service returns complete
deterministic analysis without it, so pulling the instance from the load
balancer would turn a partial degradation into an outage. `/health/ready`
reports `llm: circuit_open` and stays `200`.

Kubernetes sketch:

```yaml
livenessProbe:
  httpGet: { path: /health, port: 8000 }
  initialDelaySeconds: 10
  periodSeconds: 30
readinessProbe:
  httpGet: { path: /health/ready, port: 8000 }
  initialDelaySeconds: 5
  periodSeconds: 10
resources:
  requests: { memory: 256Mi, cpu: 250m }
  limits:   { memory: 512Mi, cpu: "1" }
```

## Observability

**Logs** are one JSON object per line, carrying `request_id` on every record —
including lines emitted deep inside the scoring code. Ship to
Datadog/CloudWatch/Loki with no parsing rule. Set `RESUMEIQ_LOG_FORMAT=console`
locally.

**Metrics** are exposed at `/metrics` in Prometheus text format:

| Metric | Type | Labels |
|---|---|---|
| `resumeiq_http_requests_total` | counter | `method`, `path`, `status` |
| `resumeiq_http_request_seconds` | histogram | `method`, `path` |
| `resumeiq_operation_total` | counter | `operation`, `outcome` |
| `resumeiq_operation_seconds` | histogram | `operation` |
| `resumeiq_llm_input_tokens_total` | counter | — |
| `resumeiq_llm_output_tokens_total` | counter | — |
| `resumeiq_llm_cache_read_tokens_total` | counter | — |
| `resumeiq_llm_failure_total` | counter | `retryable` |
| `resumeiq_llm_refusal_total` | counter | `category` |
| `resumeiq_llm_circuit_open_total` | counter | — |
| `resumeiq_injection_detected_total` | counter | `kind` |
| `resumeiq_llm_fabrication_blocked_total` | counter | — |

Paths are label-bounded by the matched route template, not the raw URL, so
cardinality stays fixed.

### Alerts worth having

| Condition | Why |
|---|---|
| `resumeiq_http_requests_total{status=~"5.."}` rising | Real failures |
| p99 of `resumeiq_http_request_seconds` > 10s | Pathological documents |
| `resumeiq_llm_circuit_open_total` increasing | Provider degraded; product is degraded but up |
| `resumeiq_llm_cache_read_tokens_total` flat while input tokens rise | Prompt caching broken — a silent cost regression |
| `resumeiq_injection_detected_total` spiking | Someone probing the screening pipeline |
| `resumeiq_operation_total{operation="extract",outcome="error"}` rising | A format regression after a dependency bump |

The metrics registry is in-process, so each replica reports its own counters;
aggregate at the scrape layer.

## Scaling notes

The service is stateless apart from the database. Two things are per-replica and
must move if you scale horizontally and need global behaviour:

1. **Rate limiter** — `SlidingWindowLimiter` in `app/api/deps.py`. Swap the
   window for Redis; the `check()` interface does not change.
2. **LLM circuit breaker** — opens per replica. This is usually acceptable
   (each instance discovers the outage independently within a few requests); a
   shared breaker only matters at high replica counts.

Analysis is CPU-bound and runs in the threadpool, so a single replica handles
concurrent requests well. The LLM call is the long pole when enabled — budget
several seconds and keep client timeouts above `RESUMEIQ_LLM_TIMEOUT_SECONDS`.

## CI

```bash
ruff check app tests ui
mypy app
pytest -q --cov=app --cov-report=term-missing
```

221 tests, 92% statement coverage, no network access required. `tests/test_ui.py`
starts a real uvicorn server in a thread and drives the Streamlit app headlessly;
skip it with `--ignore=tests/test_ui.py` for the fastest signal.
