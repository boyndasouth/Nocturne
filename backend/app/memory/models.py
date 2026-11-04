"""Pydantic models mirroring the schema in spec §4."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
