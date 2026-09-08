"""Lightweight, dependency-free metrics plus request/timing middleware.

Counters and histograms live in-process and are exposed at `/metrics` in a
Prometheus-compatible text format, so the service can be scraped without
pulling in a client library. `track()` is the one primitive used across the
codebase to time a unit of work.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_conf import get_logger, new_request_id, request_id_var

logger = get_logger(__name__)

# Buckets in seconds; the tail matters most for PDF parsing and LLM calls.
_BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

_LabelKey = tuple[tuple[str, str], ...]


class MetricsRegistry:
    """Thread-safe counters and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[_LabelKey, float]] = defaultdict(dict)
        self._hist_counts: dict[str, dict[_LabelKey, list[int]]] = defaultdict(dict)
        self._hist_sums: dict[str, dict[_LabelKey, float]] = defaultdict(dict)

    @staticmethod
    def _key(labels: dict[str, str] | None) -> _LabelKey:
        return tuple(sorted((labels or {}).items()))

    def increment(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        key = self._key(labels)
        with self._lock:
            self._counters[name][key] = self._counters[name].get(key, 0.0) + value

    def observe(self, name: str, seconds: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._hist_counts[name].setdefault(key, [0] * (len(_BUCKETS) + 1))
            self._hist_sums[name][key] = self._hist_sums[name].get(key, 0.0) + seconds
            for i, bound in enumerate(_BUCKETS):
                if seconds <= bound:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict view of every metric, used by /metrics/json and tests."""
        with self._lock:
            return {
                "counters": {
                    name: {repr(dict(k)): v for k, v in series.items()}
                    for name, series in self._counters.items()
                },
                "histograms": {
                    name: {
                        repr(dict(k)): {
                            "count": sum(counts),
                            "sum": round(self._hist_sums[name][k], 6),
                        }
                        for k, counts in series.items()
                    }
                    for name, series in self._hist_counts.items()
                },
            }

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, series in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for key, value in series.items():
                    lines.append(f"{name}{_fmt_labels(key)} {value}")
            for name, hist_series in self._hist_counts.items():
                lines.append(f"# TYPE {name} histogram")
                for key, counts in hist_series.items():
                    cumulative = 0
                    for i, bound in enumerate(_BUCKETS):
                        cumulative += counts[i]
                        lines.append(
                            f"{name}_bucket{_fmt_labels(key, le=str(bound))} {cumulative}"
                        )
                    cumulative += counts[-1]
                    lines.append(f"{name}_bucket{_fmt_labels(key, le='+Inf')} {cumulative}")
                    lines.append(f"{name}_count{_fmt_labels(key)} {cumulative}")
                    lines.append(f"{name}_sum{_fmt_labels(key)} {self._hist_sums[name][key]:.6f}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._hist_counts.clear()
            self._hist_sums.clear()


def _fmt_labels(key: _LabelKey, **extra: str) -> str:
    items = list(key) + list(extra.items())
    if not items:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in items)
    return "{" + inner + "}"


metrics = MetricsRegistry()


@contextmanager
def track(operation: str, **labels: str) -> Iterator[dict[str, Any]]:
    """Time a block, emit a histogram sample and a structured log line.

    Yields a mutable dict; anything put in it is added to the log record, so
    callers can attach domain facts (page counts, token usage, ...).
    """
    started = time.perf_counter()
    context: dict[str, Any] = {}
    outcome = "ok"
    try:
        yield context
    except Exception as exc:
        outcome = "error"
        context["exc_type"] = type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter() - started
        metrics.observe("resumeiq_operation_seconds", elapsed, {"operation": operation, **labels})
        metrics.increment(
            "resumeiq_operation_total", {"operation": operation, "outcome": outcome, **labels}
        )
        logger.info(
            "operation",
            extra={
                "operation": operation,
                "duration_ms": round(elapsed * 1000, 2),
                "outcome": outcome,
                **labels,
                **context,
            },
        )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, time the request, and record HTTP metrics."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        incoming = request.headers.get("x-request-id")
        rid = incoming if incoming and len(incoming) <= 64 else new_request_id()
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            elapsed = time.perf_counter() - started
            # Use the matched route template, not the raw path, to bound cardinality.
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            metrics.observe(
                "resumeiq_http_request_seconds", elapsed, {"method": request.method, "path": path}
            )
            metrics.increment(
                "resumeiq_http_requests_total",
                {"method": request.method, "path": path, "status": str(status_code)},
            )
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": path,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
            request_id_var.reset(token)
