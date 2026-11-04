"""Responder Agent (spec §3.1): retrieve, answer with citations, reinforce, log the turn.
It never rewrites long-term memory; only the sleep agent does that."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from app.agents.episode_writer import EpisodeWriter
from app.llm.client import ChatMessage, LLMClient, LLMError
from app.llm.prompts import RESPONDER_SYSTEM, RESPONDER_USER
from app.llm.tracing import Tracer
from app.memory.models import MemoryNote, ScoredNote, Tier
from app.memory.retriever import HierarchicalRetriever
from app.memory.store import MemoryStore

REF_RE = re.compile(r"\[M(\d+)\]")
_PREVIEW_CHARS = 600


@dataclass(frozen=True)
class Citation:
    ref: str  # "M3"
    note_id: UUID


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: list[Citation]
    retrieved: list[ScoredNote]
    episode_id: UUID | None  # None when respond(remember=False)
    context: str = ""  # the memory block the LLM saw (eval logs its size)


def _preview(note: MemoryNote) -> str:
    text = note.content
    if len(text) > _PREVIEW_CHARS:
        text = text[:_PREVIEW_CHARS].rstrip() + " …"
    return text


def _fmt_note(ref: int, note: MemoryNote) -> str:
    if note.tier is Tier.FACT:
        return f"[M{ref}] FACT (true since {note.valid_from:%Y-%m-%d}) {_preview(note)}"
    return f"[M{ref}] ({note.valid_from:%Y-%m-%d %H:%M} UTC) {_preview(note)}"


def _fmt_previous(ref: int, note: MemoryNote) -> str:
    until = f"{note.valid_to:%Y-%m-%d}" if note.valid_to else "?"
    return f"    [M{ref}] previously ({note.valid_from:%Y-%m-%d} to {until}) {_preview(note)}"


def build_memory_context(
    retrieved: Sequence[ScoredNote],
    recent: Sequence[MemoryNote],
    history: Mapping[UUID, Sequence[MemoryNote]] | None = None,
) -> tuple[str, dict[int, UUID]]:
    """Number memories with short refs (LLMs mangle UUIDs) and return the ref → id map.
    Retrieved notes come first, by relevance, each fact followed by the older facts it
    superseded; then recent turns not already shown."""
    history = history or {}
    refmap: dict[int, UUID] = {}
    seen: set[UUID] = set()
    relevant_lines: list[str] = []
    recent_lines: list[str] = []

    for s in retrieved:
        if s.note.id in seen:
            continue
        ref = len(refmap) + 1
        refmap[ref] = s.note.id
        seen.add(s.note.id)
        relevant_lines.append(_fmt_note(ref, s.note))
        for old in history.get(s.note.id, ()):
            if old.id in seen:
                continue
            ref = len(refmap) + 1
            refmap[ref] = old.id
            seen.add(old.id)
            relevant_lines.append(_fmt_previous(ref, old))
    for n in recent:
        if n.id in seen:
            continue
        ref = len(refmap) + 1
        refmap[ref] = n.id
        seen.add(n.id)
        recent_lines.append(_fmt_note(ref, n))

    if not refmap:
        return "(no memories yet)", refmap
    sections = []
    if relevant_lines:
        sections.append("Relevant memories (most relevant first):\n" + "\n".join(relevant_lines))
    if recent_lines:
        sections.append("Most recent conversation (oldest first):\n" + "\n".join(recent_lines))
    return "\n\n".join(sections), refmap


def extract_citations(answer: str, refmap: dict[int, UUID]) -> list[Citation]:
    """Valid refs in order of first appearance. Hallucinated refs are dropped."""
    out: list[Citation] = []
    seen: set[int] = set()
    for m in REF_RE.finditer(answer):
        n = int(m.group(1))
        if n in refmap and n not in seen:
            seen.add(n)
            out.append(Citation(ref=f"M{n}", note_id=refmap[n]))
    return out


class Responder:
    def __init__(
        self,
        *,
        llm: LLMClient,
        retriever: HierarchicalRetriever,
        store: MemoryStore,
        writer: EpisodeWriter,
        tracer: Tracer | None = None,
        recent_turns: int = 4,
        reinforce_eta: float = 0.2,
        max_tokens: int = 800,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._store = store
        self._writer = writer
        self._tracer = tracer or Tracer()
        self._recent_turns = recent_turns
        self._eta = reinforce_eta
        self._max_tokens = max_tokens

    async def respond(
        self, message: str, *, occurred_at: datetime | None = None, remember: bool = True
    ) -> ChatResult:
        """`remember=False` answers without side effects: no episode is written and no note
        is reinforced. The eval harness uses it so benchmark questions don't become memories
        (LoCoMo asks ~200 questions per conversation) or shift strength-based scoring."""
        now = occurred_at or datetime.now(UTC)
        with self._tracer.observe("chat", as_type="agent", input={"message": message}) as obs:
            retrieved = await self._retriever.retrieve(message, now=now, as_of=occurred_at)
            recent = await self._store.recent_episodes(self._recent_turns, before=occurred_at)
            history = await self._store.superseded_history(
                [s.note.id for s in retrieved if s.note.tier is Tier.FACT]
            )
            memories, refmap = build_memory_context(retrieved, recent, history)

            messages: list[ChatMessage] = [
                {"role": "system", "content": RESPONDER_SYSTEM},
                {
                    "role": "user",
                    "content": RESPONDER_USER.format(
                        now=f"{now:%Y-%m-%d %H:%M} UTC", memories=memories, message=message
                    ),
                },
            ]
            try:
                resp = await self._llm.chat(
                    messages, temperature=0.2, max_tokens=self._max_tokens, trace_name="responder"
                )
            except LLMError:
                # Never lose user input: log the turn without an answer, then surface the error.
                if remember:
                    await self._writer.write_turn(message, None, occurred_at=now)
                raise

            citations = extract_citations(resp.text, refmap)
            episode_id: UUID | None = None
            if remember:
                await self._store.reinforce([c.note_id for c in citations], now=now, eta=self._eta)
                episode = await self._writer.write_turn(message, resp.text, occurred_at=now)
                episode_id = episode.id

            obs.update(
                output={"answer": resp.text, "cited": [str(c.note_id) for c in citations]},
                metadata={
                    "retrieved": len(retrieved),
                    "episode_id": str(episode_id) if episode_id else None,
                },
            )
            return ChatResult(
                answer=resp.text,
                citations=citations,
                retrieved=retrieved,
                episode_id=episode_id,
                context=memories,
            )
