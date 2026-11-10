"""Systems under evaluation (spec §9).

Every system builds a memory from a Conversation (`prepare`), then answers its questions
(`answer`). All of them answer with the same LLM and the same Responder prompt, so the only
thing that differs between rows is the memory: what's in the context and how it got there.

    full-context        eval/baselines/full_context.py
    naive-rag           eval/baselines/naive_rag.py
    nocturne-no-sleep   eval/baselines/no_sleep.py   (NocturneSystem, sleep=False)
    nocturne            NocturneSystem below         (all sleep sub-agents)
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from langgraph.checkpoint.memory import InMemorySaver
from psycopg_pool import AsyncConnectionPool

from app.agents.episode_writer import EpisodeWriter
from app.agents.responder import Responder
from app.agents.sleep.graph import SleepConfig
from app.agents.sleep.runner import RunnerLimits, SleepRunner
from app.config import Settings
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder, aembed_documents
from app.memory.cycles import CycleStore
from app.memory.models import NewNote, ScoredNote, Tier
from app.memory.retriever import HierarchicalRetriever
from app.memory.store import MemoryStore
from eval.base import PrepStats, SystemAnswer
from eval.datasets.schema import Conversation, Question
from eval.memdb import EvalDatabase


class TimedRetriever(HierarchicalRetriever):
    """Records how long the last retrieve() took: the online-path latency (spec §9)."""

    last_ms: float = 0.0
    last_count: int = 0
    last_via: str = ""

    async def retrieve(self, query: str, **kwargs: Any) -> list[ScoredNote]:
        t0 = time.perf_counter()
        out = await super().retrieve(query, **kwargs)
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.last_count = len(out)
        self.last_via = via_summary(out)
        return out


def via_summary(notes: list[ScoredNote]) -> str:
    """'theme:1 fact/search:3 fact/link:1 episode/search:3', for the results CSV."""
    from collections import Counter

    counts = Counter(
        s.note.tier.value if s.note.tier.value == "theme" else f"{s.note.tier.value}/{s.via}"
        for s in notes
    )
    return " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def _in_memory_checkpointer() -> Callable[[], Any]:
    @asynccontextmanager
    async def factory():  # type: ignore[no-untyped-def]
        yield InMemorySaver()

    return factory


def episode_id(conversation_id: str, index: int) -> UUID:
    """Deterministic episode ids that sort in dataset order. The sleep runner takes episodes
    oldest first and breaks timestamp ties by id; LongMemEval-S has sessions sharing a date,
    so with random ids their turns interleaved differently on every run, which changed the
    extraction prompts and defeated the LLM cache."""
    prefix = int.from_bytes(hashlib.sha256(conversation_id.encode()).digest()[:8], "big")
    return UUID(int=(prefix << 64) | index)


def sleep_points(session_ends: list, every: int) -> list:
    """Every `every`-th session end, plus the last one (so nothing is left unconsolidated)."""
    points = session_ends[every - 1 :: every]
    if session_ends and (not points or points[-1] != session_ends[-1]):
        points.append(session_ends[-1])
    return points


class NocturneSystem:
    """The real backend components against a dedicated eval database, reset before every
    conversation. Ingest writes episodes exactly as the Episode Writer would (same format,
    valid time = transaction time = turn time) but in one batch, with no LLM calls.

    With sleep on, a cycle runs at the end of every `sleep_every`-th session (and after the
    last one), with the cycle clock set to that session's last turn. sleep_every=1 is the
    spec §9 protocol, a sleep between sessions; a larger value is the same protocol with
    longer days, which keeps 50-session LongMemEval-S haystacks within a free daily quota.
    Ingesting everything first and then sleeping in session order is equivalent to
    interleaving: a cycle only consolidates episodes recorded by its clock, and it only reads
    facts and themes written by earlier cycles."""

    def __init__(
        self,
        name: str,
        *,
        settings: Settings,
        db: EvalDatabase,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        answer_llm: LLMClient,
        sleep_llm: LLMClient | None,
        sleep: bool,
        sleep_every: int = 1,
    ) -> None:
        if sleep and sleep_llm is None:
            raise ValueError(f"{name}: sleep=True needs a sleep LLM")
        self.name = name
        self._settings = settings
        self._db = db
        self._embedder = embedder
        self._sleep = sleep
        self._sleep_every = max(1, sleep_every)
        self._store = MemoryStore(pool)
        self._retriever = TimedRetriever(
            self._store,
            embedder,
            weights=settings.score_weights,
            recency_tau_days=settings.recency_tau_days,
            top_k=settings.retrieval_top_k,
            candidate_pool=settings.retrieval_candidate_pool,
            config=settings.retrieval_config,
        )
        self._responder = Responder(
            llm=answer_llm,
            retriever=self._retriever,
            store=self._store,
            writer=EpisodeWriter(self._store, embedder),
            recent_turns=settings.recent_turns,
            reinforce_eta=settings.reinforce_eta,
            reinforce_kappa=settings.reinforce_kappa,
            context_budget_chars=settings.context_budget_chars,
            context_item_max_chars=settings.context_item_max_chars,
            max_tokens=settings.online_max_tokens,
        )
        self._runner = (
            SleepRunner(
                store=self._store,
                cycles=CycleStore(pool),
                embedder=embedder,
                llm_factory=lambda: sleep_llm,  # type: ignore[arg-type,return-value]
                checkpointer=_in_memory_checkpointer(),
                config=SleepConfig.from_settings(settings),
                limits=RunnerLimits(
                    max_episodes=settings.sleep_max_episodes,
                    max_attempts=1,
                    recursion_limit=settings.sleep_recursion_limit,
                ),
            )
            if sleep
            else None
        )

    async def prepare(self, conv: Conversation) -> PrepStats:
        t0 = time.perf_counter()
        await self._db.reset()
        episodes = conv.episodes()
        vecs = await aembed_documents(self._embedder, [e.content for e in episodes])
        notes = await self._store.insert_notes(
            [
                NewNote(
                    id=episode_id(conv.id, i),
                    tier=Tier.EPISODE,
                    content=e.content,
                    embedding=v,
                    valid_from=e.at,
                    recorded_at=e.at,
                )
                for i, (e, v) in enumerate(zip(episodes, vecs, strict=True))
            ]
        )
        if self._runner is not None:
            for end in sleep_points(conv.session_ends(), self._sleep_every):
                # A session over sleep_max_episodes is consolidated over several cycles.
                while await self._runner.run_once(clock=end) is not None:
                    pass
        stats = await self._db.stats()
        stats.episodes = len(episodes)
        if self._runner is not None:
            evidence = [n.id for n, e in zip(notes, episodes, strict=True) if e.evidence]
            stats.evidence_episodes = len(evidence)
            stats.evidence_covered = await self._db.covered(evidence)
        stats.seconds = time.perf_counter() - t0
        return stats

    async def answer(self, q: Question) -> SystemAnswer:
        result = await self._responder.respond(q.question, occurred_at=q.asked_at, remember=False)
        return SystemAnswer(
            text=result.answer,
            retrieval_ms=self._retriever.last_ms,
            retrieved=self._retriever.last_count,
            context_chars=len(result.context),
            retrieved_via=self._retriever.last_via,
        )

    async def aclose(self) -> None:
        pass  # the pool and LLM clients belong to the runner
