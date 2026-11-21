"""Changes a person makes to their own memory from the UI: correct a fact or theme, or forget
a memory. Both are versioned edits like the sleep agent's (the history shows them, with no
cycle), so every change can be traced.

Forgetting has to actually stop the agent using the memory, so it uses its own status,
`forgotten`, which no retrieval path reads (the Forgetter's `archived` notes can come back
as a fallback). Chat retrieval also reads raw conversation turns, so forgetting a fact also
forgets the turns it came from, unless another remembered fact still relies on them.
Themes that mention it are reported back rather than rewritten here; the person can
correct or forget those too.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from app.llm.embeddings import Embedder, aembed_documents
from app.memory.models import MemoryNote, Status, Tier
from app.memory.store import MemoryStore

FORGET_REASON = "forgotten at your request"
CORRECT_REASON = "corrected by you"
MAX_CONTENT = 1000


class CurationError(ValueError):
    """The change isn't allowed for this note (wrong tier or status, empty text)."""


class Curation:
    def __init__(self, pool: AsyncConnectionPool, store: MemoryStore, embedder: Embedder) -> None:
        self._pool = pool
        self._store = store
        self._embedder = embedder

    async def correct(self, note_id: UUID, content: str) -> MemoryNote:
        text = " ".join(content.split())
        if not text:
            raise CurationError("the corrected text is empty")
        if len(text) > MAX_CONTENT:
            raise CurationError(f"keep it under {MAX_CONTENT} characters")
        note = await self._store.get_note(note_id)
        if note is None:
            raise LookupError(note_id)
        if note.tier is Tier.EPISODE:
            raise CurationError("a conversation turn records what was said; forget it instead")
        if note.status is not Status.ACTIVE:
            raise CurationError("only current memories can be corrected")
        if text == note.content:
            return note
        [vec] = await aembed_documents(self._embedder, [text])
        async with self._store.transaction() as conn:
            return await self._store.edit_note(
                conn, note_id, reason=CORRECT_REASON, content=text, embedding=vec
            )

    async def forget(self, note_id: UUID) -> dict[str, Any]:
        note = await self._store.get_note(note_id)
        if note is None:
            raise LookupError(note_id)
        if note.status is Status.FORGOTTEN:
            raise CurationError("already forgotten")
        async with self._store.transaction() as conn:
            await self._store.edit_note(
                conn, note_id, reason=FORGET_REASON, status=Status.FORGOTTEN
            )
            turns: list[UUID] = []
            if note.tier is Tier.FACT and note.source_episode_ids:
                # Its turns, unless another fact the agent still uses cites them.
                cur = await conn.execute(
                    """
                    select e.id from memory_notes e
                    where e.id = any(%(src)s) and e.status <> 'forgotten'
                      and not exists (
                        select 1 from memory_notes f
                        where f.tier = 'fact' and f.status <> 'forgotten' and f.id <> %(id)s
                          and e.id = any(f.source_episode_ids))
                    order by e.valid_from
                    """,
                    {"src": note.source_episode_ids, "id": note_id},
                )
                turns = [r["id"] for r in await cur.fetchall()]
                for turn in turns:
                    await self._store.edit_note(
                        conn, turn, reason=FORGET_REASON, status=Status.FORGOTTEN
                    )
            cur = await conn.execute(
                """
                select t.id, t.content from memory_links l
                join memory_notes t on t.id = l.src_id
                where l.dst_id = %s and l.link_type = 'summarizes' and t.status = 'active'
                order by t.valid_from
                """,
                (note_id,),
            )
            themes = [dict(r) for r in await cur.fetchall()]
        return {"note_id": note_id, "turns_forgotten": len(turns), "themes": themes}
