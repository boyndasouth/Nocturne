"""Pydantic models mirroring the schema in spec §4."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Tier(StrEnum):
    EPISODE = "episode"
    FACT = "fact"
    THEME = "theme"


class Status(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class LinkType(StrEnum):
    RELATED = "related"
    ELABORATES = "elaborates"
    EXAMPLE_OF = "example_of"
    CAUSED_BY = "caused_by"
    SUPERSEDES = "supersedes"


class MemoryNote(BaseModel):
    """A row of memory_notes, without the embedding (which is large and rarely needed)."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    tier: Tier
    content: str
    keywords: list[str] = Field(default_factory=list)
    strength: float
    access_count: int
    last_accessed: datetime | None
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    status: Status
    source_episode_ids: list[UUID] = Field(default_factory=list)
    version: int
    decay_tau_days: float = 30.0

    @field_validator("last_accessed", "valid_from", "valid_to", "recorded_at")
    @classmethod
    def _utc(cls, v: datetime | None) -> datetime | None:
        """psycopg returns timestamptz in the *session* time zone. Prompts label times as
        UTC and show dates, so normalise here, whatever the server or pooler is set to."""
        return v.astimezone(UTC) if v is not None and v.tzinfo is not None else v


class NewNote(BaseModel):
    """Input for inserting a note. Timestamps default to now() in the database."""

    tier: Tier
    content: str = Field(min_length=1)
    embedding: list[float] = Field(repr=False)
    keywords: list[str] = Field(default_factory=list)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime | None = None
    recorded_at: datetime | None = None
    source_episode_ids: list[UUID] = Field(default_factory=list)
    valid_to: datetime | None = None
    status: Status = Status.ACTIVE


class Candidate(BaseModel):
    """A note returned by hybrid search, before final scoring."""

    note: MemoryNote
    semantic_sim: float  # cosine similarity, in [-1, 1]
    keyword_score: float  # ts_rank_cd normalised to [0, 1)


class ScoredNote(BaseModel):
    note: MemoryNote
    score: float
    semantic_sim: float
    keyword_score: float
    recency: float


class Neighbor(BaseModel):
    """An existing note found by vector search, with its cosine similarity to a probe."""

    note: MemoryNote
    sim: float


class VersionRecord(BaseModel):
    note_id: UUID
    version: int
    snapshot: dict
    changed_by_cycle: UUID | None
    reason: str | None
    created_at: datetime
