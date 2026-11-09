"""Pieces every system shares: result types, the System protocol, the common answer prompt,
and an embedding cache."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from app.llm.client import LLMClient
from app.llm.embeddings import Embedder
from app.llm.prompts import RESPONDER_SYSTEM, RESPONDER_USER
from eval.datasets.schema import Conversation, Question


@dataclass
class PrepStats:
    episodes: int = 0  # raw turns/utterances fed in
    memory_items: int = 0  # what the system searches over (active notes, chunks, turns)
    memory_chars: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    sleep_cycles: int = 0
    facts_created: int = 0
    merges: int = 0
    supersessions: int = 0
    links: int = 0  # Linker links between facts (not supersedes / theme membership)
    archived: int = 0  # notes the Forgetter (and Abstractor) archived
    audit_score: float | None = None  # mean Auditor score over this memory's cycles
    audit_items: int = 0
    # Extraction recall: episodes the benchmark marks as holding an answer, and how many of
    # them are a source of at least one fact (any status). Sleeping systems only.
    evidence_episodes: int = 0
    evidence_covered: int = 0
    seconds: float = 0.0


@dataclass
class SystemAnswer:
    text: str
    retrieval_ms: float | None = None  # None: no retrieval step (full context)
    retrieved: int = 0
    context_chars: int = 0
    retrieved_via: str = ""  # nocturne only: what retrieval returned, by tier and route


class System(Protocol):
    name: str

    async def prepare(self, conv: Conversation) -> PrepStats: ...

    async def answer(self, q: Question) -> SystemAnswer: ...

    async def aclose(self) -> None: ...


async def answer_with_context(
    llm: LLMClient, question: str, memories: str, *, now: datetime, max_tokens: int
) -> str:
    """The Responder's prompt and sampling, for baselines that build their own context."""
    resp = await llm.chat(
        [
            {"role": "system", "content": RESPONDER_SYSTEM},
            {
                "role": "user",
                "content": RESPONDER_USER.format(
                    now=f"{now:%Y-%m-%d %H:%M} UTC", memories=memories, message=question
                ),
            },
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        trace_name="eval.answer",
    )
    return resp.text


class CachedEmbedder:
    """Every system embeds the same episodes; compute each text once per process."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.dim = inner.dim
        self._docs: dict[str, list[float]] = {}
        self._queries: dict[str, list[float]] = {}

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._docs))
        if missing:
            for t, v in zip(missing, self._inner.embed_documents(missing), strict=True):
                self._docs[t] = v
        return [self._docs[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if text not in self._queries:
            self._queries[text] = self._inner.embed_query(text)
        return self._queries[text]

    def warmup(self) -> None:
        warm = getattr(self._inner, "warmup", None)
        if callable(warm):
            warm()
