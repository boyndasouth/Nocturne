"""Episode Writer (spec §3.1): append raw input as episode notes. No LLM calls."""

from __future__ import annotations

import re
from datetime import datetime

from app.llm.embeddings import Embedder, aembed_documents
from app.memory.models import MemoryNote, NewNote, Tier
from app.memory.store import MemoryStore

CITATION_RE = re.compile(r"\s?\[M\d+\]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def strip_citations(text: str) -> str:
    """[M#] refs only mean something inside one prompt, so don't persist them."""
    return CITATION_RE.sub("", text).strip()


def format_turn(user_message: str, assistant_message: str | None) -> str:
    parts = [f"User: {user_message.strip()}"]
    if assistant_message:
        parts.append(f"Assistant: {strip_citations(assistant_message)}")
    return "\n".join(parts)


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """Pack paragraphs greedily into chunks of at most `max_chars`. Oversized paragraphs are
    split at sentence boundaries, and as a last resort at max_chars."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    pieces: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sentence in _SENTENCE_END_RE.split(para):
            while len(sentence) > max_chars:
                pieces.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


class EpisodeWriter:
    def __init__(self, store: MemoryStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    async def write_turn(
        self, user_message: str, assistant_message: str | None, *, occurred_at: datetime
    ) -> MemoryNote:
        """One episode per chat turn. In a live chat (and a benchmark replay) the system learns
        the turn when it happens, so valid time and transaction time are the same."""
        content = format_turn(user_message, assistant_message)
        [vec] = await aembed_documents(self._embedder, [content])
        [note] = await self._store.insert_notes(
            [
                NewNote(
                    tier=Tier.EPISODE,
                    content=content,
                    embedding=vec,
                    valid_from=occurred_at,
                    recorded_at=occurred_at,
                )
            ]
        )
        return note

    async def write_document(
        self, text: str, *, source: str | None, occurred_at: datetime | None
    ) -> list[MemoryNote]:
        """Ingested notes: valid time is when the material is from (e.g. lecture date) and
        transaction time is now, when the system learned it."""
        header = f"[Source: {source}]\n" if source else ""
        contents = [header + chunk for chunk in chunk_text(text)]
        if not contents:
            return []
        vecs = await aembed_documents(self._embedder, contents)
        return await self._store.insert_notes(
            [
                NewNote(tier=Tier.EPISODE, content=c, embedding=v, valid_from=occurred_at)
                for c, v in zip(contents, vecs, strict=True)
            ]
        )
