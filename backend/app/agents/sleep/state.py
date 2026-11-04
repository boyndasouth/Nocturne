"""Sleep-graph state.

LangGraph checkpoints this state after every step, so a failed cycle resumes where it
stopped. It holds only JSON primitives (dumped Pydantic models): checkpoints then survive code
changes and need no custom serializer. Embeddings are recomputed instead of stored, which keeps
checkpoint rows small on a free-tier database.

Refs: "n:<uuid>" is an existing note; "c:<key>" is a fact candidate from this cycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field


def note_ref(note_id: UUID | str) -> str:
    return f"n:{note_id}"


def cand_ref(key: str) -> str:
    return f"c:{key}"


def parse_ref(ref: str) -> tuple[str, str]:
    kind, _, value = ref.partition(":")
    if kind not in ("n", "c") or not value:
        raise ValueError(f"bad ref {ref!r}")
    return kind, value


class EpisodeIn(BaseModel):
    id: UUID
    content: str
    valid_from: datetime


class FactCandidate(BaseModel):
    key: str  # "<batch>.<index>", stable across resumes
    content: str
    keywords: list[str] = Field(default_factory=list)
    valid_from: datetime
    source_episode_ids: list[UUID]


class NeighborRef(BaseModel):
    ref: str
    content: str
    valid_from: datetime
    sim: float


class Group(BaseModel):
    """A candidate plus the similar facts an LLM judge should compare it with."""

    key: str
    content: str
    valid_from: datetime
    neighbors: list[NeighborRef]


class Conflict(BaseModel):
    new: str  # ref
    old: str  # ref
    reason: str = ""
    # Valid-from of both sides, so the apply step can order them without another lookup.
    new_valid_from: datetime
    old_valid_from: datetime


class SleepState(TypedDict, total=False):
    cycle_id: str
    clock: str  # ISO; the cycle's notion of "now" (simulated during benchmark replay)
    episodes: list[dict[str, Any]]  # EpisodeIn
    extract_plan: list[list[int]]  # episode indexes per batch, frozen on the first step
    extract_cursor: int  # next extraction batch
    candidates: list[dict[str, Any]]  # FactCandidate
    dedup_groups: list[dict[str, Any]]  # Group, judged in chunks
    dedup_cursor: int
    duplicates: dict[str, str]  # candidate key -> ref it duplicates
    conflict_groups: list[dict[str, Any]]
    conflict_cursor: int
    conflicts: list[dict[str, Any]]  # Conflict
    tokens: int
    llm_calls: int
    result: dict[str, Any]
