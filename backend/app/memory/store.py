"""Data access for memory notes. All SQL lives here."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
from psycopg_pool import AsyncConnectionPool

from app.memory.decay import reinforce
from app.memory.models import Candidate, MemoryNote, NewNote, Status, Tier

_COLUMNS = (
    "id", "tier", "content", "keywords", "strength", "access_count", "last_accessed",
    "valid_from", "valid_to", "recorded_at", "status", "source_episode_ids", "version",
)  # fmt: skip
_ENUM_COLUMNS = {"tier", "status"}


def _select_columns(prefix: str = "") -> str:
    return ", ".join(
        f"{prefix}{c}::text as {c}" if c in _ENUM_COLUMNS else f"{prefix}{c}" for c in _COLUMNS
    )


NOTE_COLUMNS = _select_columns()
_N_COLUMNS = _select_columns("n.")

# Full-text query with OR semantics: plainto_tsquery ANDs every lexeme, which makes
# natural-language questions match almost nothing. Stop words are already removed.
_OR_TSQUERY = "replace(plainto_tsquery('english', %(qtext)s)::text, ' & ', ' | ')::tsquery"

_HYBRID_SQL = f"""
with q as (
    select %(qvec)s::vector as v, {_OR_TSQUERY} as tsq
),
filtered as (
    select n.* from memory_notes n
    where n.tier = any(%(tiers)s::note_tier[]) and n.status = any(%(statuses)s::note_status[])
      -- as-of transaction time: only what the system knew by then (benchmark replay)
      and (%(as_of)s::timestamptz is null or n.recorded_at <= %(as_of)s)
),
by_vector as (
    select f.id from filtered f, q
    where f.embedding is not null
    order by f.embedding <=> q.v
    limit %(pool)s
),
by_keyword as (
    select f.id from filtered f, q
    where f.search_tsv @@ q.tsq
    order by ts_rank_cd(f.search_tsv, q.tsq) desc
    limit %(pool)s
),
cand as (select id from by_vector union select id from by_keyword)
select {_N_COLUMNS},
       coalesce(1 - (n.embedding <=> q.v), 0) as semantic_sim,
       ts_rank_cd(n.search_tsv, q.tsq, 32) as keyword_score
from memory_notes n join cand using (id), q
"""


def _note(row: dict[str, Any]) -> MemoryNote:
    return MemoryNote.model_validate(row)


def _vec(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


class MemoryStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ping(self) -> bool:
        try:
            async with self._pool.connection(timeout=5) as conn:
                await conn.execute("select 1")
            return True
        except Exception:
            return False

    async def insert_notes(self, notes: Sequence[NewNote]) -> list[MemoryNote]:
        if not notes:
            return []
        out: list[MemoryNote] = []
        async with self._pool.connection() as conn, conn.transaction():
            for n in notes:
                cur = await conn.execute(
                    f"""
                    insert into memory_notes
                        (tier, content, keywords, embedding, strength,
                         valid_from, recorded_at, source_episode_ids)
                    values (%s::note_tier, %s, %s, %s, %s,
                            coalesce(%s, now()), coalesce(%s, now()), %s)
                    returning {NOTE_COLUMNS}
                    """,
                    (
                        n.tier.value,
                        n.content,
                        n.keywords,
                        _vec(n.embedding),
                        n.strength,
                        n.valid_from,
                        n.recorded_at,
                        n.source_episode_ids,
                    ),
                )
                row = await cur.fetchone()
                assert row is not None
                out.append(_note(row))
        return out

    async def get_note(self, note_id: UUID) -> MemoryNote | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"select {NOTE_COLUMNS} from memory_notes where id = %s", (note_id,)
            )
            row = await cur.fetchone()
        return _note(row) if row else None

    async def recent_episodes(self, limit: int, before: datetime | None = None) -> list[MemoryNote]:
        """Most recent active episodes, returned oldest first."""
        if limit <= 0:
            return []
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                select {NOTE_COLUMNS} from memory_notes
                where tier = 'episode' and status = 'active'
                  and (%(before)s::timestamptz is null or recorded_at <= %(before)s)
                order by valid_from desc, recorded_at desc
                limit %(limit)s
                """,
                {"limit": limit, "before": before},
            )
            rows = await cur.fetchall()
        return [_note(r) for r in reversed(rows)]

    async def hybrid_candidates(
        self,
        query_vec: Sequence[float],
        query_text: str,
        *,
        tiers: Sequence[Tier],
        statuses: Sequence[Status] = (Status.ACTIVE,),
        pool_size: int = 40,
        as_of: datetime | None = None,
    ) -> list[Candidate]:
        """Union of the top `pool_size` notes by vector distance and by full-text rank.
        `as_of` hides notes recorded after that time."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                _HYBRID_SQL,
                {
                    "qvec": _vec(query_vec),
                    "qtext": query_text,
                    "tiers": [t.value for t in tiers],
                    "statuses": [s.value for s in statuses],
                    "pool": pool_size,
                    "as_of": as_of,
                },
            )
            rows = await cur.fetchall()
        return [
            Candidate(
                note=_note(r),
                semantic_sim=float(r["semantic_sim"]),
                keyword_score=float(r["keyword_score"]),
            )
            for r in rows
        ]

    async def reinforce(self, note_ids: Sequence[UUID], *, now: datetime, eta: float) -> None:
        """Access reinforcement (spec §3.1). Access metadata isn't a content edit, so
        `version` isn't bumped and no memory_versions snapshot is written."""
        ids = list(dict.fromkeys(note_ids))
        if not ids:
            return
        async with self._pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                "select id, strength from memory_notes where id = any(%s) for update", (ids,)
            )
            rows = await cur.fetchall()
            async with conn.cursor() as c:
                await c.executemany(
                    "update memory_notes set strength = %s, access_count = access_count + 1,"
                    " last_accessed = greatest(coalesce(last_accessed, %s), %s) where id = %s",
                    [(reinforce(r["strength"], eta), now, now, r["id"]) for r in rows],
                )
