"""Async Anthropic client wrapper.

Adds the operational behaviour a production service needs around the SDK:
a single shared client, explicit timeouts, a circuit breaker so a provider
outage degrades the service instead of hanging every request, token accounting,
and a typed error the caller can act on.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types import OutputConfigParam, TextBlockParam

from app.config import Settings, get_settings
from app.logging_conf import get_logger
from app.observability import metrics, track

logger = get_logger(__name__)

#: Consecutive failures before the breaker opens.
BREAKER_THRESHOLD = 4
#: Seconds the breaker stays open before allowing a probe request.
BREAKER_COOLDOWN = 60.0


class LLMError(RuntimeError):
    """Any failure that should degrade to deterministic-only output."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMRefusal(LLMError):
    """The model declined the request (`stop_reason == "refusal"`)."""


@dataclass(slots=True)
class LLMResponse:
    data: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    latency_ms: float
    request_id: str | None


class _CircuitBreaker:
    """Minimal in-process breaker: closed -> open -> half-open probe."""

    def __init__(self) -> None:
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= BREAKER_COOLDOWN:
            # Half-open: let exactly one request through to probe recovery.
            self._opened_at = None
            self._failures = BREAKER_THRESHOLD - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= BREAKER_THRESHOLD and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.error("llm_circuit_open", extra={"cooldown_s": BREAKER_COOLDOWN})
            metrics.increment("resumeiq_llm_circuit_open_total")


class AnthropicClient:
    """Thin, testable wrapper over `AsyncAnthropic`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._breaker = _CircuitBreaker()
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def available(self) -> bool:
        return self.settings.llm_available and not self._breaker.is_open

    def _ensure_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            if not self.settings.anthropic_api_key:
                raise LLMError("No ANTHROPIC_API_KEY configured.")
            self._client = anthropic.AsyncAnthropic(
                api_key=self.settings.anthropic_api_key,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def structured(
        self,
        *,
        system: str,
        user_message: str,
        schema: dict[str, Any],
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        """One structured-output call, returning validated JSON.

        The system prompt is marked cacheable: it is byte-identical on every
        request, so after the first call it is served from cache at a fraction
        of the input cost.
        """
        if self._breaker.is_open:
            raise LLMError("LLM circuit breaker is open.", retryable=True)

        client = self._ensure_client()
        started = time.perf_counter()

        # The system prompt is byte-identical on every request, so marking it
        # cacheable makes every call after the first read it from cache.
        system_blocks: list[TextBlockParam] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        output_config: OutputConfigParam = {
            "effort": effort or self.settings.llm_effort,  # type: ignore[typeddict-item]
            "format": {"type": "json_schema", "schema": schema},
        }

        try:
            with track("llm_call", model=self.settings.llm_model) as ctx:
                response = await client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=max_tokens or self.settings.llm_max_tokens,
                    system=system_blocks,
                    messages=[{"role": "user", "content": user_message}],
                    thinking={"type": "adaptive"},
                    output_config=output_config,
                )
                ctx["stop_reason"] = response.stop_reason
                ctx["input_tokens"] = response.usage.input_tokens
                ctx["output_tokens"] = response.usage.output_tokens
        except anthropic.BadRequestError as exc:
            self._breaker.record_failure()
            raise LLMError(f"Request rejected by the API: {exc}") from exc
        except anthropic.AuthenticationError as exc:
            self._breaker.record_failure()
            raise LLMError("Anthropic API key is invalid.") from exc
        except anthropic.PermissionDeniedError as exc:
            self._breaker.record_failure()
            raise LLMError("Anthropic API key lacks permission for this model.") from exc
        except anthropic.RateLimitError as exc:
            self._breaker.record_failure()
            raise LLMError("Rate limited by the Anthropic API.", retryable=True) from exc
        except anthropic.APITimeoutError as exc:
            self._breaker.record_failure()
            raise LLMError("The Anthropic API timed out.", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            self._breaker.record_failure()
            raise LLMError("Could not reach the Anthropic API.", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            self._breaker.record_failure()
            raise LLMError(
                f"Anthropic API error {exc.status_code}.", retryable=exc.status_code >= 500
            ) from exc

        # A refusal is a successful HTTP call, so it is checked after the
        # exception handlers, not inside them.
        if response.stop_reason == "refusal":
            self._breaker.record_success()
            category = getattr(response.stop_details, "category", None)
            metrics.increment("resumeiq_llm_refusal_total", {"category": str(category)})
            raise LLMRefusal(f"The model declined this request (category={category}).")

        self._breaker.record_success()
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        metrics.increment("resumeiq_llm_input_tokens_total", value=usage.input_tokens)
        metrics.increment("resumeiq_llm_output_tokens_total", value=usage.output_tokens)
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        metrics.increment("resumeiq_llm_cache_read_tokens_total", value=cache_read)

        return LLMResponse(
            data=_extract_json(response),
            model=self.settings.llm_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            latency_ms=round(latency_ms, 1),
            request_id=response._request_id,
        )


def _extract_json(response: Any) -> dict[str, Any]:
    """Read the JSON object out of the response content blocks.

    `output_config.format` guarantees valid JSON in a text block, but the
    response may also carry thinking blocks, so the blocks are filtered by type
    rather than indexed positionally.
    """
    texts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if not texts:
        raise LLMError("The model returned no text content.")
    raw = "".join(texts).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model returned malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("The model returned JSON that is not an object.")
    return parsed


_client: AnthropicClient | None = None


def get_llm_client(settings: Settings | None = None) -> AnthropicClient:
    """Process-wide singleton so connections and the breaker are shared."""
    global _client
    if _client is None:
        _client = AnthropicClient(settings)
    return _client


async def close_llm_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
