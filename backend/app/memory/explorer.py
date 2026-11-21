"""Read-only queries behind the memory explorer UI: note lists, a note's history and links,
the graph, and what one sleep cycle changed. Nothing here writes.

A note doesn't store which cycle created it. Sleep inserts facts and themes with
`recorded_at` = the cycle's clock, so that's the join; ingest and chat only write episodes,
whose cycle is `consolidated_in`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from app.memory.models import MemoryNote, Status, Tier
from app.memory.store import NOTE_COLUMNS, _select_columns

_N = _select_columns("n.")


def _note(row: dict[str, Any]) -> MemoryNote:
    return MemoryNote.model_validate({k: row[k] for k in MemoryNote.model_fields if k in row})


def _ilike(q: str | None) -> str | None:
    if not q or not q.strip():
        return None
    escaped = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def change_kind(tier: str, before_status: str, after_status: str, content_changed: bool) -> str:
    """What one version record did, from the states either side of it (more robust than
    parsing the free-text reason)."""
    if after_status == "superseded" and before_status != "superseded":
        return "superseded"
    if after_status == "archived" and before_status != "archived":
        return "archived"
    if tier == "theme" and content_changed:
        return "theme_updated"
    if tier == "fact" and not content_changed:
        return "merged"  # duplicate evidence folded in
    return "edited"


class MemoryExplorer:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def stats(self) -> dict[str, Any]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select tier::text as tier, status::text as status, count(*) as n"
                " from memory_notes group by 1, 2 order by 1, 2"
            )
            counts = await cur.fetchall()
            cur = await conn.execute(
                "select (select count(*) from memory_links) as links,"
                " (select count(*) from sleep_cycles) as cycles,"
                " (select count(*) from memory_notes where tier = 'episode'"
                "   and consolidated_in is null) as pending_episodes"
            )
            row = await cur.fetchone()
        assert row is not None
        by_tier: dict[str, dict[str, int]] = {t.value: {s.value: 0 for s in Status} for t in Tier}
        for r in counts:
            by_tier[r["tier"]][r["status"]] = int(r["n"])
        return {
            "notes": by_tier,
            "links": int(row["links"]),
            "cycles": int(row["cycles"]),
            "pending_episodes": int(row["pending_episodes"]),
        }

    async def list_notes(
        self,
        *,
        tiers: Sequence[Tier],
        statuses: Sequence[Status],
        q: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[MemoryNote], int]:
        """Newest first (by when the system learned it)."""
        where = """
            where tier = any(%(tiers)s::note_tier[])
              and status = any(%(statuses)s::note_status[])
              and (%(q)s::text is null or content ilike %(q)s or %(raw)s = any(keywords))
        """
        params = {
            "tiers": [t.value for t in tiers],
            "statuses": [s.value for s in statuses],
            "q": _ilike(q),
            "raw": (q or "").strip().lower(),
            "limit": limit,
            "offset": offset,
        }
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"select {NOTE_COLUMNS} from memory_notes {where}"
                " order by recorded_at desc, valid_from desc, id"
                " limit %(limit)s offset %(offset)s",
                params,
            )
            rows = await cur.fetchall()
            cur = await conn.execute(f"select count(*) as n from memory_notes {where}", params)
            count = await cur.fetchone()
        return [_note(r) for r in rows], int(count["n"]) if count else 0

    async def note_detail(self, note_id: UUID) -> dict[str, Any] | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                select {NOTE_COLUMNS},
                    case when tier = 'episode' then consolidated_in
                         else (select c.cycle_id from sleep_cycles c
                               where c.clock = memory_notes.recorded_at
                               order by c.started_at limit 1)
                    end as created_by_cycle
                from memory_notes where id = %s
                """,
                (note_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            note = _note(row)

            cur = await conn.execute(
                """
                select v.version, v.content, v.reason, v.changed_by_cycle, v.created_at,
                       v.snapshot->>'status' as status,
                       (v.snapshot->>'valid_from')::timestamptz as valid_from,
                       (v.snapshot->>'valid_to')::timestamptz as valid_to,
                       (v.snapshot->>'strength')::real as strength,
                       jsonb_array_length(coalesce(v.snapshot->'source_episode_ids', '[]'))
                           as source_count,
                       c.clock as cycle_clock
                from memory_versions v
                left join sleep_cycles c on c.cycle_id = v.changed_by_cycle
                where v.note_id = %s order by v.version
                """,
                (note_id,),
            )
            versions = await cur.fetchall()

            cur = await conn.execute(
                f"""
                select l.link_type::text as link_type, l.weight, l.created_by_cycle,
                       case when l.src_id = %(id)s then 'out' else 'in' end as direction,
                       {_N}
                from memory_links l
                join memory_notes n on n.id = case when l.src_id = %(id)s
                                                   then l.dst_id else l.src_id end
                where l.src_id = %(id)s or l.dst_id = %(id)s
                order by l.link_type, n.valid_from
                limit 200
                """,
                {"id": note_id},
            )
            links = await cur.fetchall()

            cur = await conn.execute(
                f"""
                select {_N} from unnest(%s::uuid[]) with ordinality as s(id, o)
                join memory_notes n on n.id = s.id
                order by s.o limit 50
                """,
                (note.source_episode_ids,),
            )
            sources = await cur.fetchall()

        # Each version record is the state *before* a change; the live row is the last state.
        history = [
            {
                "version": v["version"],
                "content": v["content"],
                "status": v["status"] or "active",
                "valid_from": v["valid_from"],
                "valid_to": v["valid_to"],
                "strength": v["strength"],
                "source_count": v["source_count"],
                "ended": {
                    "cycle_id": v["changed_by_cycle"],
                    "cycle_clock": v["cycle_clock"],
                    "at": v["created_at"],
                    "reason": v["reason"],
                },
            }
            for v in versions
        ]
        history.append(
            {
                "version": note.version,
                "content": note.content,
                "status": note.status.value,
                "valid_from": note.valid_from,
                "valid_to": note.valid_to,
                "strength": note.strength,
                "source_count": len(note.source_episode_ids),
                "ended": None,
            }
        )
        return {
            "note": note,
            "created_by_cycle": row["created_by_cycle"],
            "history": history,
            "links": [
                {
                    "link_type": r["link_type"],
                    "direction": r["direction"],
                    "weight": r["weight"],
                    "created_by_cycle": r["created_by_cycle"],
                    "other": _note(r),
                }
                for r in links
            ],
            "sources": [_note(r) for r in sources],
        }

    async def graph(
        self,
        *,
        tiers: Sequence[Tier],
        statuses: Sequence[Status],
        limit: int,
    ) -> dict[str, Any]:
        """The newest `limit` notes of these tiers/statuses and the links among them."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                f"""
                select {NOTE_COLUMNS}, count(*) over () as total from memory_notes
                where tier = any(%(tiers)s::note_tier[])
                  and status = any(%(statuses)s::note_status[])
                order by recorded_at desc, id
                limit %(limit)s
                """,
                {
                    "tiers": [t.value for t in tiers],
                    "statuses": [s.value for s in statuses],
                    "limit": limit,
                },
            )
            rows = await cur.fetchall()
            ids = [r["id"] for r in rows]
            cur = await conn.execute(
                "select src_id, dst_id, link_type::text as link_type, weight, created_by_cycle"
                " from memory_links where src_id = any(%(ids)s) and dst_id = any(%(ids)s)"
                " order by src_id, dst_id, link_type",
                {"ids": ids},
            )
            edges = await cur.fetchall()
        return {
            "nodes": [_note(r) for r in rows],
            "edges": list(edges),
            "total": int(rows[0]["total"]) if rows else 0,
        }

    async def cycle_changes(self, cycle_id: UUID, *, limit: int = 500) -> dict[str, Any] | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select clock, audit from sleep_cycles where cycle_id = %s", (cycle_id,)
            )
            cycle = await cur.fetchone()
            if cycle is None:
                return None
            clock: datetime = cycle["clock"]

            cur = await conn.execute(
                f"""
                select {NOTE_COLUMNS} from memory_notes
                where tier <> 'episode' and recorded_at = %s
                order by tier, valid_from, id limit %s
                """,
                (clock, limit),
            )
            created = [_note(r) for r in await cur.fetchall()]

            cur = await conn.execute(
                f"""
                select {NOTE_COLUMNS} from memory_notes
                where tier = 'episode' and consolidated_in = %s
                order by valid_from, id limit %s
                """,
                (cycle_id, limit),
            )
            episodes = [_note(r) for r in await cur.fetchall()]

            # One row per version record this cycle wrote, with the state after it (the next
            # version's snapshot, or the live row if it's the latest).
            cur = await conn.execute(
                """
                select v.note_id, v.version, v.reason, v.content as before_content,
                       coalesce(v.snapshot->>'status', 'active') as before_status,
                       n.tier::text as tier,
                       coalesce(nx.content, n.content) as after_content,
                       coalesce(nx.snapshot->>'status', n.status::text) as after_status,
                       n.status::text as status_now
                from memory_versions v
                join memory_notes n on n.id = v.note_id
                left join memory_versions nx on nx.note_id = v.note_id
                                            and nx.version = v.version + 1
                where v.changed_by_cycle = %s
                order by n.tier, v.created_at, v.note_id
                """,
                (cycle_id,),
            )
            edit_rows = await cur.fetchall()

            cur = await conn.execute(
                """
                select l.src_id, l.dst_id, l.link_type::text as link_type, l.weight,
                       s.content as src_content, d.content as dst_content
                from memory_links l
                join memory_notes s on s.id = l.src_id
                join memory_notes d on d.id = l.dst_id
                where l.created_by_cycle = %s and l.link_type <> 'summarizes'
                order by l.link_type, s.valid_from limit %s
                """,
                (cycle_id, limit),
            )
            links = list(await cur.fetchall())

        edits: list[dict[str, Any]] = []
        episodes_forgotten = 0
        for r in edit_rows:
            kind = change_kind(
                r["tier"],
                r["before_status"],
                r["after_status"],
                r["before_content"] != r["after_content"],
            )
            if r["tier"] == "episode":
                episodes_forgotten += kind == "archived"
                continue
            if len(edits) < limit:
                edits.append(
                    {
                        "note_id": r["note_id"],
                        "tier": r["tier"],
                        "kind": kind,
                        "version": r["version"],
                        "reason": r["reason"],
                        "before": r["before_content"],
                        "after": r["after_content"],
                        "status_now": r["status_now"],
                    }
                )
        audit = cycle["audit"] if isinstance(cycle["audit"], dict) else None
        return {
            "cycle_id": cycle_id,
            "clock": clock,
            "created": created,
            "episodes": episodes,
            "edits": edits,
            "episodes_forgotten": episodes_forgotten,
            "links": links,
            "audit": audit,
        }
