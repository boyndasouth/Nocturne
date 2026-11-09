"""Benchmark-neutral data model. Every loader converts its format into these classes, so
systems and the runner never see LongMemEval or LoCoMo specifics.

A Conversation is one memory to build (a LongMemEval haystack, a LoCoMo dialogue); it owns
the questions asked against it. LongMemEval has one question per haystack, LoCoMo ~200.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from app.agents.episode_writer import format_turn

# Spacing between turns inside a session. Sessions only carry a date (sometimes to the
# minute), but episodes need a strict order for recency, replay and `as_of` filtering.
TURN_SPACING = timedelta(seconds=30)

Style = Literal["chat", "dialogue"]


@dataclass(frozen=True)
class Turn:
    speaker: str  # "user" / "assistant" for chat; the person's name for dialogue
    text: str
    evidence: bool = False  # the benchmark marks this turn as holding (part of) an answer


@dataclass(frozen=True)
class Session:
    id: str
    date: datetime  # timezone-aware
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Episode:
    """One memory write: what the Episode Writer would store for a chat turn."""

    session_id: str
    content: str
    at: datetime
    # The user's turn (chat) or the utterance (dialogue) is marked as evidence. Evidence in
    # an assistant reply doesn't count: the Extractor never takes facts from replies.
    evidence: bool = False


@dataclass(frozen=True)
class Question:
    id: str
    conversation_id: str
    question: str
    answer: str  # gold answer, or the explanation/rubric for abstention and preference
    category: str  # reporting bucket, e.g. "knowledge-update", "abstention", "temporal"
    question_type: str  # the benchmark's own label, kept for traceability
    asked_at: datetime
    abstention: bool = False
    evidence: tuple[str, ...] = ()  # session or dialogue ids holding the answer


@dataclass(frozen=True)
class Conversation:
    id: str
    benchmark: str
    style: Style
    sessions: tuple[Session, ...]  # sorted by date
    questions: tuple[Question, ...] = field(default=())

    def episodes(self) -> list[Episode]:
        """Chat: one episode per user turn plus the assistant reply that follows, formatted
        exactly like the live Episode Writer. Dialogue: one episode per utterance,
        "Name: text". Times are the session date plus TURN_SPACING per episode."""
        out: list[Episode] = []
        for s in self.sessions:
            texts = _chat_pairs(s.turns) if self.style == "chat" else _utterances(s.turns)
            out.extend(
                Episode(session_id=s.id, content=t, at=s.date + i * TURN_SPACING, evidence=ev)
                for i, (t, ev) in enumerate(texts)
            )
        return out

    def session_ends(self) -> list[datetime]:
        """The time of each session's last episode: when a between-sessions sleep runs."""
        ends: dict[str, datetime] = {}
        for ep in self.episodes():
            ends[ep.session_id] = ep.at
        return [ends[s.id] for s in self.sessions if s.id in ends]


def _chat_pairs(turns: tuple[Turn, ...]) -> list[tuple[str, bool]]:
    """(episode text, user turn is evidence) per user turn + the replies that follow."""
    texts: list[tuple[str, bool]] = []
    user: Turn | None = None
    assistant: list[str] = []

    def flush() -> None:
        if user is not None:
            texts.append((format_turn(user.text, "\n".join(assistant) or None), user.evidence))
        elif assistant:  # an assistant turn with no user turn before it
            texts.append(("Assistant: " + "\n".join(assistant), False))

    for t in turns:
        if not t.text.strip():
            continue
        if t.speaker == "user":
            flush()
            user, assistant = t, []
        else:
            assistant.append(t.text.strip())
    flush()
    return texts


def _utterances(turns: tuple[Turn, ...]) -> list[tuple[str, bool]]:
    return [(f"{t.speaker}: {t.text.strip()}", t.evidence) for t in turns if t.text.strip()]
