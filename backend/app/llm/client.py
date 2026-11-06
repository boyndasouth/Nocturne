"""One client for every OpenAI-compatible chat endpoint (Groq, Gemini, OpenRouter).

- Rate limiting: requests are spaced to at most `rpm` per minute, per client.
- Retries: 408/409/425/429/5xx and transport errors, honouring Retry-After (header, or
  Gemini's `retryDelay` in the body), else exponential backoff with full jitter.
- Tracing: each call is a Langfuse "generation" with token usage.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from app.llm.tracing import Tracer


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMError(RuntimeError):
    """Non-retryable provider error (bad request, auth, malformed response)."""


class RetryableLLMError(LLMError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExhaustedError(LLMError):
    """A 429 for a *daily* quota. Retrying within minutes can't help, so it isn't retried:
    callers fail fast (the eval stops, a sleep cycle fails and backs off)."""


# Gemini quota ids ("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), Groq messages
# ("tokens per day (TPD)"), OpenRouter ("free-models-per-day").
_DAILY_QUOTA_RE = re.compile(r"per[\s_-]?day|\((?:TPD|RPD)\)", re.IGNORECASE)
_QUOTA_FIELD_RE = re.compile(r'"(quotaId|quotaValue)"\s*:\s*"([^"]+)"')


def daily_quota_detail(body: str) -> str | None:
    """None unless `body` (a 429 response) is about a per-day limit; otherwise a short
    description, with Gemini's quota id and value when present."""
    if not _DAILY_QUOTA_RE.search(body):
        return None
    fields = dict(_QUOTA_FIELD_RE.findall(body))
    if fields:
        return ", ".join(f"{k}={v}" for k, v in fields.items())
    m = _DAILY_QUOTA_RE.search(body)
    assert m is not None
    start = max(0, m.start() - 120)
    return " ".join(body[start : m.end() + 80].split())


class LLMClient(Protocol):
    name: str
    model: str

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        trace_name: str = "llm.chat",
    ) -> LLMResponse: ...

    async def aclose(self) -> None: ...


RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')

Sleep = Callable[[float], Awaitable[None]]


def parse_retry_after(resp: httpx.Response) -> float | None:
    header = resp.headers.get("retry-after")
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass  # HTTP-date form; fall through to backoff
    match = _RETRY_DELAY_RE.search(resp.text)
    return float(match.group(1)) if match else None


class AsyncRateLimiter:
    """Reserves evenly spaced start times. Single-process only, which is all v1 needs."""

    def __init__(
        self, rpm: int, *, clock: Callable[[], float] = time.monotonic, sleep: Sleep = asyncio.sleep
    ) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._clock = clock
        self._sleep = sleep

    async def acquire(self) -> float:
        """Wait for the next slot; returns how long it waited."""
        if self._interval == 0.0:
            return 0.0
        async with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self._interval
        wait = start - now
        if wait > 0:
            await self._sleep(wait)
        return wait


class _BackoffWait:
    def __init__(self, base: float = 1.0, cap: float = 60.0) -> None:
        self.base, self.cap = base, cap

    def __call__(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, RetryableLLMError) and exc.retry_after is not None:
            return min(self.cap, exc.retry_after + random.uniform(0.0, 0.5))
        return min(self.cap, random.uniform(0.0, self.base * 2 ** (state.attempt_number - 1)))


class OpenAICompatClient:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        rpm: int,
        tracer: Tracer | None = None,
        max_retries: int = 5,
        timeout_s: float = 60.0,
        http: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        backoff_base_s: float = 1.0,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        """`extra_params` are merged into every request body, for provider-specific
        options such as Groq's `reasoning_effort`."""
        self.name = name
        self.model = model
        self._extra = dict(extra_params or {})
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._tracer = tracer or Tracer()
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient()
        self._sleep = sleep
        self._limiter = AsyncRateLimiter(rpm, sleep=sleep)
        self._wait = _BackoffWait(base=backoff_base_s)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        trace_name: str = "llm.chat",
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self._extra,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with self._tracer.observe(
            trace_name,
            as_type="generation",
            model=self.model,
            input=list(messages),
            model_parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
                **{k: v for k, v in self._extra.items() if isinstance(v, str | int | float | bool)},
            },
            metadata={"provider": self.name},
        ) as gen:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(self._max_retries + 1),
                wait=self._wait,
                retry=retry_if_exception_type(RetryableLLMError),
                sleep=self._sleep,
                reraise=True,
            )
            async for attempt in retrying:
                with attempt:
                    resp = await self._post(payload)
            gen.update(
                output=resp.text,
                usage_details={"input": resp.input_tokens, "output": resp.output_tokens},
                metadata={"provider": self.name, "latency_ms": round(resp.latency_ms, 1)},
            )
            return resp

    async def _post(self, payload: dict[str, Any]) -> LLMResponse:
        await self._limiter.acquire()
        t0 = time.perf_counter()
        try:
            r = await self._http.post(
                self._url, json=payload, headers=self._headers, timeout=self._timeout
            )
        except httpx.TransportError as exc:
            raise RetryableLLMError(f"{self.name}: transport error: {exc!r}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        if r.status_code == 429 and (detail := daily_quota_detail(r.text)):
            raise QuotaExhaustedError(
                f"{self.name}: daily quota used up for {self.model} ({detail}). Wait for it to "
                "reset (Gemini: midnight Pacific time) or switch to a model with its own quota."
            )
        if r.status_code in RETRYABLE_STATUS:
            raise RetryableLLMError(
                f"{self.name}: HTTP {r.status_code}: {r.text[:300]}", parse_retry_after(r)
            )
        if r.status_code >= 400:
            raise LLMError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")

        try:
            data = r.json()
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: unexpected response: {r.text[:300]}") from exc
        if not text.strip() and choice.get("finish_reason") == "length":
            # Reasoning models can spend the whole token budget thinking and never answer.
            raise LLMError(
                f"{self.name}: hit max_tokens before producing an answer; raise max_tokens "
                "or lower the reasoning effort"
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
        )
