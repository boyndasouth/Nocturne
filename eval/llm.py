"""LLM wrappers for evaluation: a response cache and a token meter.

Cache (spec §11 "cache LLM responses keyed by prompt hash"): a rerun after a rate limit, a
crash, or a scoring-only change replays finished calls for free and gets identical text, so
sleep cycles rebuild the same memory. The key covers everything that changes the output:
provider, model, messages, temperature, max_tokens, json_mode, and provider extras.

Meter: counts tokens per role (answer / sleep / judge), with cache hits counted at their
original token cost. That keeps "tokens online vs offline" comparable between a cold run
and a cached rerun; `cache_hits` says how much was actually free.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.llm.client import ChatMessage, LLMClient, LLMResponse


@dataclass
class Meter:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def snapshot(self) -> Meter:
        return Meter(self.calls, self.cache_hits, self.input_tokens, self.output_tokens)

    def __sub__(self, other: Meter) -> Meter:
        return Meter(
            self.calls - other.calls,
            self.cache_hits - other.cache_hits,
            self.input_tokens - other.input_tokens,
            self.output_tokens - other.output_tokens,
        )


class ResponseCache:
    """SQLite, stdlib only. One table, keyed by a SHA-256 of the canonical request."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "create table if not exists responses (key text primary key, text text not null,"
            " model text not null, input_tokens int not null, output_tokens int not null,"
            " created_at real not null)"
        )
        self._db.commit()

    @staticmethod
    def key(**request: Any) -> str:
        blob = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        row = self._db.execute(
            "select text, model, input_tokens, output_tokens from responses where key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return LLMResponse(
            text=row[0], model=row[1], input_tokens=row[2], output_tokens=row[3], latency_ms=0.0
        )

    def put(self, key: str, resp: LLMResponse) -> None:
        self._db.execute(
            "insert or replace into responses values (?, ?, ?, ?, ?, ?)",
            (key, resp.text, resp.model, resp.input_tokens, resp.output_tokens, time.time()),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class EvalLLM:
    """Wraps any LLMClient with the cache (optional) and a meter. Implements LLMClient, so
    the Responder, the sleep graph and the judge use it unchanged."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        cache: ResponseCache | None,
        extra_key: dict[str, Any] | None = None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._extra_key = extra_key or {}
        self.name = inner.name
        self.model = inner.model
        self.meter = Meter()

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        trace_name: str = "llm.chat",
    ) -> LLMResponse:
        key = ResponseCache.key(
            provider=self.name,
            model=self.model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            extra=self._extra_key,
        )
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self._count(hit, cached=True)
                return hit
        resp = await self._inner.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            trace_name=trace_name,
        )
        if self._cache is not None and resp.text.strip():
            self._cache.put(key, resp)
        self._count(resp, cached=False)
        return resp

    def _count(self, resp: LLMResponse, *, cached: bool) -> None:
        self.meter.calls += 1
        self.meter.cache_hits += int(cached)
        self.meter.input_tokens += resp.input_tokens
        self.meter.output_tokens += resp.output_tokens

    async def aclose(self) -> None:
        await self._inner.aclose()
