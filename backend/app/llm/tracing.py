"""Thin wrapper over Langfuse (SDK v3+ / OpenTelemetry API). No-ops when keys are absent,
so tests and local runs never need a Langfuse account."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from app.config import Settings

log = logging.getLogger(__name__)

ObservationType = Literal["span", "generation", "agent", "retriever", "tool", "chain", "embedding"]


class Observation(Protocol):
    def update(self, **kwargs: Any) -> Any: ...


class _NoopObservation:
    def update(self, **kwargs: Any) -> _NoopObservation:
        return self


class Tracer:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def observe(
        self, name: str, *, as_type: ObservationType = "span", **attrs: Any
    ) -> Iterator[Observation]:
        """Open a nested observation. Errors are recorded on the observation and re-raised."""
        if self._client is None:
            yield _NoopObservation()
            return
        with self._client.start_as_current_observation(name=name, as_type=as_type, **attrs) as obs:
            try:
                yield obs
            except Exception as exc:
                obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])
                raise

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()


def build_tracer(settings: Settings) -> Tracer:
    pk, sk = settings.langfuse_public_key, settings.langfuse_secret_key
    if pk is None or sk is None:
        log.info("Langfuse keys not set; tracing disabled")
        return Tracer()
    from langfuse import Langfuse

    client = Langfuse(
        public_key=pk.get_secret_value(),
        secret_key=sk.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )
    return Tracer(client)
