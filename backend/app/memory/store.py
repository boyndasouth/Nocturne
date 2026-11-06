"""Data access for memory notes. All SQL lives here."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
import psycopg
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from app.memory.decay import reinforce
from app.memory.models import (
    Candidate,
    LinkType,
    MemoryNote,
    Neighbor,
    NewNote,
    Status,
    Tier,
    VersionRecord,
)

_COLUMNS = (
    "id", "tier", "content", "keywords", "strength", "access_count", "last_accessed",
    "valid_from", "valid_to", "recorded_at", "status", "source_episode_ids", "version",
    "decay_tau_days",
)  # fmt: skip

# Columns a sleep cycle may change. Every change goes through `edit_note`, which snapshots the
# row into memory_versions first.
EDITABLE_COLUMNS = frozenset(
    {"content", "keywords", "embedding", "strength", "status", "valid_to",
     "source_episode_ids", "decay_tau_days"}
)  # fmt: skip
_CASTS = {"status": "::note_status"}
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


def _floats(v: Any) -> list[float]:
    """A pgvector value as floats: a Vector object, a numpy array, or a list."""
    if hasattr(v, "to_list"):
        v = v.to_list()
    return [float(x) for x in v]


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

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[psycopg.AsyncConnection]:
        """A connection inside one transaction, for multi-step writes (sleep cycle apply)."""
        async with self._pool.connection() as conn, conn.transaction():
            yield conn

    async def insert_notes(self, notes: Sequence[NewNote]) -> list[MemoryNote]:
        if not notes:
            return []
        async with self.transaction() as conn:
            return await self.insert_notes_tx(conn, notes)

    async def insert_notes_tx(
        self, conn: psycopg.AsyncConnection, notes: Sequence[NewNote]
    ) -> list[MemoryNote]:
        out: list[MemoryNote] = []
        for n in notes:
            cur = await conn.execute(
                f"""
                insert into memory_notes
                    (tier, content, keywords, embedding, strength,
                     valid_from, valid_to, recorded_at, status, source_episode_ids)
                values (%s::note_tier, %s, %s, %s, %s,
                        coalesce(%s, now()), %s, coalesce(%s, now()), %s::note_status, %s)
                returning {NOTE_COLUMNS}
                """,
                (
                    n.tier.value,
                    n.content,
                    n.keywords,
                    _vec(n.embedding),
                    n.strength,
                    n.valid_from,
                    n.valid_to,
                    n.recorded_at,
                    n.status.value,
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

    # --- Versioned edits (spec §3.2 design rule, §4.3) ----------------------------------------

    async def edit_note(
        self,
        conn: psycopg.AsyncConnection,
        note_id: UUID,
        *,
        reason: str,
        cycle_id: UUID | None = None,
        **changes: Any,
    ) -> MemoryNote:
        """Snapshot the full current row into memory_versions, then apply `changes` and bump
        `version`. Must run inside the caller's transaction so the snapshot and the edit
        commit together."""
        if not changes:
            raise ValueError("edit_note needs at least one change")
        unknown = set(changes) - EDITABLE_COLUMNS
        if unknown:
            raise ValueError(f"not editable: {sorted(unknown)}")

        await self._snapshot(conn, note_id, cycle_id=cycle_id, reason=reason)
        assignments = [
            sql.SQL("{} = {}" + _CASTS.get(col, "")).format(sql.Identifier(col), sql.Placeholder())
            for col in changes
        ]
        values = [
            _vec(v) if col == "embedding" else (v.value if isinstance(v, Status) else v)
            for col, v in changes.items()
        ]
        query = sql.SQL(
            "update memory_notes set {sets}, version = version + 1 where id = %s returning "
        ).format(sets=sql.SQL(", ").join(assignments)) + sql.SQL(NOTE_COLUMNS)  # type: ignore[arg-type]
        cur = await conn.execute(query, [*values, note_id])
        row = await cur.fetchone()
        assert row is not None
        return _note(row)

    async def _snapshot(
        self, conn: psycopg.AsyncConnection, note_id: UUID, *, cycle_id: UUID | None, reason: str
    ) -> None:
        """Lock the row and copy it, as it is now, into memory_versions."""
        cur = await conn.execute(
            "select version from memory_notes where id = %s for update", (note_id,)
        )
        if await cur.fetchone() is None:
            raise LookupError(f"note {note_id} not found")
        await conn.execute(
            """
            insert into memory_versions
                (note_id, version, content, snapshot, changed_by_cycle, reason)
            select id, version, content, to_jsonb(n) - 'search_tsv', %s, %s
            from memory_notes n where id = %s
            """,
            (cycle_id, reason, note_id),
        )

    async def merge_evidence_tx(
        self,
        conn: psycopg.AsyncConnection,
        note_id: UUID,
        episode_ids: Sequence[UUID],
        *,
        eta: float,
        cycle_id: UUID | None,
        reason: str,
    ) -> MemoryNote:
        """A duplicate was found: add its source episodes (order kept, no repeats) and give
        the fact a reinforcement, computed in SQL so it composes with concurrent online
        reinforcement instead of overwriting it."""
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        await self._snapshot(conn, note_id, cycle_id=cycle_id, reason=reason)
        cur = await conn.execute(
            f"""
            update memory_notes set
                source_episode_ids = array(
                    select e from unnest(source_episode_ids || %(ids)s::uuid[])
                        with ordinality as t(e, o)
                    group by e order by min(o)
                ),
                strength = least(1.0, strength + %(eta)s * (1.0 - strength)),
                version = version + 1
            where id = %(id)s
            returning {NOTE_COLUMNS}
            """,
            {"ids": list(episode_ids), "eta": eta, "id": note_id},
        )
        row = await cur.fetchone()
        assert row is not None
        return _note(row)

    async def add_link(
        self,
        conn: psycopg.AsyncConnection,
        src_id: UUID,
        dst_id: UUID,
        link_type: LinkType,
        *,
        cycle_id: UUID | None,
        weight: float = 1.0,
    ) -> bool:
        """True if the link is new (an existing identical link is left as it is)."""
        cur = await conn.execute(
            """
            insert into memory_links (src_id, dst_id, link_type, weight, created_by_cycle)
            values (%s, %s, %s::link_type, %s, %s)
            on conflict do nothing
            """,
            (src_id, dst_id, link_type.value, weight, cycle_id),
        )
        return cur.rowcount == 1

    async def replace_theme_members_tx(
        self,
        conn: psycopg.AsyncConnection,
        theme_id: UUID,
        member_ids: Sequence[UUID],
        *,
        cycle_id: UUID | None,
    ) -> None:
        """Make `member_ids` exactly the facts this theme summarises. Links are derived
        structure, not memory content: the theme's own text is versioned by edit_note."""
        await conn.execute(
            "delete from memory_links where src_id = %s and link_type = 'summarizes'",
            (theme_id,),
        )
        for m in dict.fromkeys(member_ids):
            await self.add_link(conn, theme_id, m, LinkType.SUMMARIZES, cycle_id=cycle_id)

    async def links_of(
        self, note_ids: Sequence[UUID], *, types: Sequence[LinkType] | None = None
    ) -> list[dict[str, Any]]:
        """Links touching any of `note_ids`, in either direction."""
        if not note_ids:
            return []
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select src_id, dst_id, link_type::text as link_type, weight, created_by_cycle"
                " from memory_links where (src_id = any(%(ids)s) or dst_id = any(%(ids)s))"
                " and (%(types)s::text[] is null or link_type::text = any(%(types)s))"
                " order by src_id, dst_id, link_type",
                {"ids": list(note_ids), "types": [t.value for t in types] if types else None},
            )
            return list(await cur.fetchall())

    async def versions(self, note_id: UUID) -> list[VersionRecord]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select note_id, version, snapshot, changed_by_cycle, reason, created_at"
                " from memory_versions where note_id = %s order by version",
                (note_id,),
            )
            rows = await cur.fetchall()
        return [VersionRecord.model_validate(r) for r in rows]

    # --- Sleep-cycle inputs -------------------------------------------------------------------

    async def pending_episodes(self, *, before: datetime, limit: int) -> list[MemoryNote]:
        """Episodes no cycle has consolidated yet, oldest first."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                select {NOTE_COLUMNS} from memory_notes
                where tier = 'episode' and consolidated_in is null and recorded_at <= %s
                order by recorded_at, valid_from, id
                limit %s
                """,
                (before, limit),
            )
            rows = await cur.fetchall()
        return [_note(r) for r in rows]

    async def count_pending_episodes(self) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select count(*) as n from memory_notes"
                " where tier = 'episode' and consolidated_in is null"
            )
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def notes_by_ids(self, ids: Sequence[UUID]) -> list[MemoryNote]:
        """In the order given; missing ids are skipped."""
        if not ids:
            return []
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"select {NOTE_COLUMNS} from memory_notes where id = any(%s)", (list(ids),)
            )
            rows = {r["id"]: _note(r) for r in await cur.fetchall()}
        return [rows[i] for i in ids if i in rows]

    async def mark_consolidated(
        self, conn: psycopg.AsyncConnection, episode_ids: Sequence[UUID], cycle_id: UUID
    ) -> None:
        await conn.execute(
            "update memory_notes set consolidated_in = %s where id = any(%s)",
            (cycle_id, list(episode_ids)),
        )

    async def nearest_active_facts(
        self, vectors: Sequence[Sequence[float]], *, k: int, min_sim: float
    ) -> list[list[Neighbor]]:
        """For each probe vector, up to k active facts with cosine similarity ≥ min_sim."""
        out: list[list[Neighbor]] = []
        async with self._pool.connection() as conn:
            for v in vectors:
                cur = await conn.execute(
                    f"""
                    select * from (
                        select {NOTE_COLUMNS}, 1 - (embedding <=> %(v)s) as sim
                        from memory_notes
                        where tier = 'fact' and status = 'active' and embedding is not null
                        order by embedding <=> %(v)s
                        limit %(k)s
                    ) t where sim >= %(min)s
                    """,
                    {"v": _vec(v), "k": k, "min": min_sim},
                )
                rows = await cur.fetchall()
                out.append([Neighbor(note=_note(r), sim=float(r["sim"])) for r in rows])
        return out

    async def active_fact_vectors(self, *, limit: int) -> list[tuple[MemoryNote, list[float]]]:
        """Active facts with their embeddings, newest first, for clustering."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                select {NOTE_COLUMNS}, embedding from memory_notes
                where tier = 'fact' and status = 'active' and embedding is not null
                order by recorded_at desc, id
                limit %s
                """,
                (limit,),
            )
            rows = await cur.fetchall()
        return [(_note(r), _floats(r["embedding"])) for r in rows]

    async def active_themes(self) -> list[tuple[MemoryNote, list[UUID]]]:
        """Active themes with the ids of every fact they summarise."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                select {_N_COLUMNS},
                    coalesce(array_agg(l.dst_id order by l.dst_id)
                             filter (where l.dst_id is not null), '{{}}') as members
                from memory_notes n
                left join memory_links l on l.src_id = n.id and l.link_type = 'summarizes'
                where n.tier = 'theme' and n.status = 'active'
                group by n.id
                order by n.recorded_at, n.id
                """
            )
            rows = await cur.fetchall()
        return [(_note(r), list(r["members"])) for r in rows]

    async def superseded_history(
        self, note_ids: Sequence[UUID], *, depth: int = 3
    ) -> dict[UUID, list[MemoryNote]]:
        """For each note, the chain of notes it superseded (newest first), up to `depth`."""
        ids = list(dict.fromkeys(note_ids))
        if not ids:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                with recursive chain(root, id, depth) as (
                    select l.src_id, l.dst_id, 1 from memory_links l
                    where l.src_id = any(%(ids)s) and l.link_type = 'supersedes'
                    union all
                    select c.root, l.dst_id, c.depth + 1 from chain c
                    join memory_links l on l.src_id = c.id and l.link_type = 'supersedes'
                    where c.depth < %(depth)s
                )
                select c.root, c.depth, {_N_COLUMNS}
                from chain c join memory_notes n on n.id = c.id
                order by c.root, c.depth, n.valid_from desc
                """,
                {"ids": ids, "depth": depth},
            )
            rows = await cur.fetchall()
        history: dict[UUID, list[MemoryNote]] = {}
        seen: set[tuple[UUID, UUID]] = set()
        for r in rows:
            note = _note(r)
            if (r["root"], note.id) in seen:
                continue
            seen.add((r["root"], note.id))
            history.setdefault(r["root"], []).append(note)
        return history
