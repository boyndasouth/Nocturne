"""Runs sleep cycles: picks the work, owns concurrency, resumes failed cycles.

A cycle runs in two halves so `/sleep` can return a cycle id before the slow part:
    begin()   - claim a failed/stale cycle to resume, or create one for pending episodes
    execute() - run the graph to completion (or record the failure for the next attempt)

Failed cycles are always resumed before new work is picked, so the "prepare" steps that read
memory see the same state on resume as they did the first time (only sleep writes facts).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row

from app.agents.sleep.graph import SleepConfig, SleepDeps, build_sleep_graph, initial_state
from app.agents.sleep.state import EpisodeIn
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder
from app.llm.tracing import Tracer
from app.memory.cycles import CycleStore, SleepBusy, SleepCycle
from app.memory.store import MemoryStore

log = logging.getLogger("nocturne.sleep")

CheckpointerFactory = Callable[[], AbstractAsyncContextManager[BaseCheckpointSaver]]


def postgres_checkpointer(database_url: str) -> CheckpointerFactory:
    """A dedicated autocommit connection per cycle, as LangGraph's saver requires.
    Prepared statements stay off, like the app pool, so transaction poolers work."""

    @asynccontextmanager
    async def factory() -> AsyncIterator[BaseCheckpointSaver]:
        async with await psycopg.AsyncConnection.connect(
            database_url, autocommit=True, prepare_threshold=None, row_factory=dict_row
        ) as conn:
            yield AsyncPostgresSaver(conn)  # type: ignore[arg-type]

    return factory


@dataclass(frozen=True)
class PreparedCycle:
    cycle: SleepCycle
    resumed: bool


@dataclass(frozen=True)
class RunnerLimits:
    max_episodes: int = 200
    max_attempts: int = 3
    stale_after: timedelta = timedelta(minutes=30)
    recursion_limit: int = 500


class SleepRunner:
    def __init__(
        self,
        *,
        store: MemoryStore,
        cycles: CycleStore,
        embedder: Embedder,
        llm_factory: Callable[[], LLMClient],
        checkpointer: CheckpointerFactory,
        config: SleepConfig,
        limits: RunnerLimits,
        tracer: Tracer | None = None,
    ) -> None:
        self._store = store
        self._cycles = cycles
        self._embedder = embedder
        self._llm_factory = llm_factory
        self._llm: LLMClient | None = None
        self._checkpointer = checkpointer
        self._config = config
        self._limits = limits
        self._tracer = tracer or Tracer()
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = self._llm_factory()  # raises ConfigError if the key is missing
        return self._llm

    async def aclose(self) -> None:
        if self._llm is not None:
            await self._llm.aclose()

    async def begin(self, *, clock: datetime | None = None) -> PreparedCycle | None:
        """Claim work. Returns None when there's nothing to consolidate. Raises SleepBusy if a
        cycle is already running, here or in another process. `clock` (benchmark replay) sets
        the cycle's notion of now; a resumed cycle keeps its original clock."""
        if self._busy:
            raise SleepBusy("a sleep cycle is already running in this process")
        self._busy = True
        try:
            self._get_llm()  # fail before claiming anything if the sleep LLM isn't configured
            stale = self._limits.stale_after
            resumable = await self._cycles.find_resumable(
                max_attempts=self._limits.max_attempts, stale_after=stale
            )
            if resumable is not None:
                cycle = await self._cycles.claim(resumable.cycle_id, stale_after=stale)
                log.info("resuming sleep cycle %s (attempt %d)", cycle.cycle_id, cycle.attempts)
                return PreparedCycle(cycle, resumed=True)

            now = clock or datetime.now(UTC)
            episodes = await self._store.pending_episodes(
                before=now, limit=self._limits.max_episodes
            )
            if not episodes:
                self._busy = False
                return None
            cycle = await self._cycles.create([e.id for e in episodes], now)
            return PreparedCycle(cycle, resumed=False)
        except BaseException:
            self._busy = False
            raise

    async def execute(self, prepared: PreparedCycle) -> SleepCycle:
        """Run the graph for a cycle from `begin`. On error the cycle is marked failed (with
        its checkpoint kept for resume) and the error is re-raised."""
        cycle = prepared.cycle
        cycle_id = cycle.cycle_id
        config: dict[str, Any] = {
            "configurable": {"thread_id": str(cycle_id)},
            "recursion_limit": self._limits.recursion_limit,
        }
        deps = SleepDeps(
            store=self._store,
            cycles=self._cycles,
            embedder=self._embedder,
            llm=self._get_llm(),
            tracer=self._tracer,
            config=self._config,
        )
        try:
            with self._tracer.observe(
                "sleep_cycle",
                as_type="agent",
                input={"cycle_id": str(cycle_id), "episodes": len(cycle.episode_ids)},
                metadata={"resumed": prepared.resumed, "attempt": cycle.attempts},
            ) as obs:
                async with self._checkpointer() as saver:
                    graph = build_sleep_graph(deps, saver)
                    snapshot = await graph.aget_state(config)
                    if prepared.resumed and snapshot.next:
                        graph_input = None  # continue from the last checkpoint
                    else:
                        notes = await self._store.notes_by_ids(cycle.episode_ids)
                        episodes = [
                            EpisodeIn(id=n.id, content=n.content, valid_from=n.valid_from)
                            for n in notes
                        ]
                        graph_input = initial_state(cycle_id, cycle.clock, episodes)
                    final = await graph.ainvoke(graph_input, config)
                    await saver.adelete_thread(str(cycle_id))  # keep free-tier storage flat
                obs.update(output=final.get("result"))
        except BaseException as exc:
            log.exception("sleep cycle %s failed", cycle_id)
            try:
                await self._cycles.fail(cycle_id, f"{type(exc).__name__}: {exc}")
            except Exception:  # e.g. pool closed at shutdown; the stale-heartbeat rule recovers
                log.exception("could not mark cycle %s failed", cycle_id)
            raise
        finally:
            self._busy = False
            self._tracer.flush()
        done = await self._cycles.get(cycle_id)
        assert done is not None
        log.info(
            "sleep cycle %s: %d episodes -> %d facts, %d merges, %d supersessions, %d links,"
            " %d new / %d updated themes, %d archived, audit %s",
            cycle_id,
            done.episodes_processed,
            done.facts_created,
            done.merges,
            done.supersessions,
            done.links_created,
            done.themes_created,
            done.themes_updated,
            done.archived,
            "-" if done.audit_score is None else f"{done.audit_score:.2f}",
        )
        return done

    async def run_once(self, *, clock: datetime | None = None) -> SleepCycle | None:
        prepared = await self.begin(clock=clock)
        return None if prepared is None else await self.execute(prepared)
