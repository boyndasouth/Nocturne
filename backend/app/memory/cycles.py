"""Sleep-cycle bookkeeping (spec §4.4): one row per cycle, resumable after a crash.

Concurrency: a partial unique index allows at most one `running` row, so two processes (the
idle trigger and the GitHub Actions cron, say) can never consolidate at the same time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import psycopg
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel


class SleepBusy(RuntimeError):
    """Another sleep cycle is running."""


class SleepCycle(BaseModel):
    cycle_id: UUID
    started_at: datetime
    finished_at: datetime | None
    status: str
    clock: datetime
    episode_ids: list[UUID]
    episodes_processed: int
    facts_created: int
    merges: int
    supersessions: int
    archived: int
    tokens_used: int
    llm_calls: int
    attempts: int
    heartbeat_at: datetime
    error: str | None


_COLS = (
    "cycle_id, started_at, finished_at, status::text as status, clock, episode_ids,"
    " episodes_processed, facts_created, merges, supersessions, archived, tokens_used,"
    " llm_calls, attempts, heartbeat_at, error"
)


def _cycle(row: dict[str, Any]) -> SleepCycle:
    return SleepCycle.model_validate(row)


class CycleStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create(self, episode_ids: Sequence[UUID], clock: datetime) -> SleepCycle:
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    f"insert into sleep_cycles (episode_ids, clock) values (%s, %s)"
                    f" returning {_COLS}",
                    (list(episode_ids), clock),
                )
                row = await cur.fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise SleepBusy("another sleep cycle is running") from exc
        assert row is not None
        return _cycle(row)

    async def find_resumable(
        self, *, max_attempts: int, stale_after: timedelta
    ) -> SleepCycle | None:
        """The oldest cycle that died part-way: failed with attempts left, or 'running' with no
        heartbeat for `stale_after` (its process crashed)."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                select {_COLS} from sleep_cycles
                where (status = 'failed' and attempts < %(max)s)
                   or (status = 'running' and heartbeat_at < now() - %(stale)s)
                order by started_at limit 1
                """,
                {"max": max_attempts, "stale": stale_after},
            )
            row = await cur.fetchone()
        return _cycle(row) if row else None

    async def claim(self, cycle_id: UUID, *, stale_after: timedelta) -> SleepCycle:
        """Take over a failed or stale cycle. Raises SleepBusy if someone else holds it or
        another cycle is running."""
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(
                    f"""
                    update sleep_cycles
                    set status = 'running', attempts = attempts + 1, heartbeat_at = now(),
                        error = null
                    where cycle_id = %(id)s
                      and (status = 'failed'
                           or (status = 'running' and heartbeat_at < now() - %(stale)s))
                    returning {_COLS}
                    """,
                    {"id": cycle_id, "stale": stale_after},
                )
                row = await cur.fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise SleepBusy("another sleep cycle is running") from exc
        if row is None:
            raise SleepBusy(f"cycle {cycle_id} was claimed by another process")
        return _cycle(row)

    async def heartbeat(self, cycle_id: UUID) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "update sleep_cycles set heartbeat_at = now()"
                " where cycle_id = %s and status = 'running'",
                (cycle_id,),
            )

    async def fail(self, cycle_id: UUID, error: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                # Only a running cycle can fail: if apply already committed 'succeeded', an
                # error raised afterwards must not make it look resumable (it'd re-apply).
                "update sleep_cycles set status = 'failed', error = %s, finished_at = now()"
                " where cycle_id = %s and status = 'running'",
                (error[:2000], cycle_id),
            )

    async def finish_tx(
        self,
        conn: psycopg.AsyncConnection,
        cycle_id: UUID,
        *,
        episodes_processed: int,
        facts_created: int,
        merges: int,
        supersessions: int,
        tokens_used: int,
        llm_calls: int,
    ) -> None:
        """Mark success inside the apply transaction, so memory changes and the cycle's
        status commit together."""
        await conn.execute(
            """
            update sleep_cycles set status = 'succeeded', finished_at = now(), error = null,
                episodes_processed = %s, facts_created = %s, merges = %s, supersessions = %s,
                tokens_used = %s, llm_calls = %s
            where cycle_id = %s
            """,
            (
                episodes_processed,
                facts_created,
                merges,
                supersessions,
                tokens_used,
                llm_calls,
                cycle_id,
            ),
        )

    async def get(self, cycle_id: UUID) -> SleepCycle | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"select {_COLS} from sleep_cycles where cycle_id = %s", (cycle_id,)
            )
            row = await cur.fetchone()
        return _cycle(row) if row else None

    async def list(self, *, limit: int = 50) -> list[SleepCycle]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"select {_COLS} from sleep_cycles order by started_at desc limit %s", (limit,)
            )
            rows = await cur.fetchall()
        return [_cycle(r) for r in rows]
