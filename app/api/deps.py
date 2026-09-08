"""Shared API dependencies."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Request

from app.config import Settings, get_settings
from app.errors import RateLimited


class SlidingWindowLimiter:
    """Per-client sliding-window rate limiter.

    In-process and therefore per-replica: adequate behind a single instance or
    as a second line of defence behind an edge limiter. For a multi-replica
    deployment, move the window to Redis - the interface here does not change.
    """

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Return (allowed, seconds_until_reset)."""
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            cutoff = now - self.window
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self.limit:
                return False, max(1, int(window[0] + self.window - now))
            window.append(now)

            # Opportunistic cleanup so idle clients do not leak memory.
            if len(self._hits) > 4096:
                for stale_key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale_key]
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter: SlidingWindowLimiter | None = None


def get_limiter(settings: Settings | None = None) -> SlidingWindowLimiter:
    global _limiter
    if _limiter is None:
        settings = settings or get_settings()
        _limiter = SlidingWindowLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )
    return _limiter


def client_key(request: Request) -> str:
    """Identify the caller.

    `X-Forwarded-For` is only trusted when the service sits behind a proxy that
    sets it; document that in your ingress config before relying on it.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    """Dependency that enforces the limit on write-heavy endpoints."""
    allowed, retry_after = get_limiter().check(client_key(request))
    if not allowed:
        raise RateLimited(
            "Too many requests. Slow down and try again shortly.",
            details={"retry_after_seconds": retry_after},
        )
