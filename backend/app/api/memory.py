"""Memory explorer endpoints for the frontend: notes, a note's history and links, the memory
graph, what a sleep cycle changed ("what changed overnight"), and the two changes a person
can make themselves: correct a memory, or forget it."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import ContainerDep, require_api_key
from app.memory.curation import Curation, CurationError
from app.memory.explorer import MemoryExplorer
from app.memory.models import LinkType, MemoryNote, Status, Tier

router = APIRouter(tags=["memory"], dependencies=[Depends(require_api_key)])


def get_explorer(c: ContainerDep) -> MemoryExplorer:
    return MemoryExplorer(c.pool)


ExplorerDep = Annotated[MemoryExplorer, Depends(get_explorer)]


class NoteOut(BaseModel):
    id: UUID
    tier: Tier
    status: Status
    content: str
    keywords: list[str]
    strength: float
    access_count: int
    last_accessed: datetime | None
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    version: int
    source_count: int

    @classmethod
    def of(cls, n: MemoryNote) -> NoteOut:
        return cls(
            **n.model_dump(exclude={"source_episode_ids", "decay_tau_days"}),
            source_count=len(n.source_episode_ids),
        )


class NotePage(BaseModel):
    items: list[NoteOut]
    total: int
    limit: int
    offset: int


class VersionEnd(BaseModel):
    cycle_id: UUID | None
    cycle_clock: datetime | None
    at: datetime
    reason: str | None


class HistoryEntry(BaseModel):
    """One state of the note. `ended` says what replaced it; None for the current state."""

    version: int
    content: str
    status: Status
    valid_from: datetime | None
    valid_to: datetime | None
    strength: float | None
    source_count: int | None
    ended: VersionEnd | None


class LinkOut(BaseModel):
    link_type: LinkType
    direction: Literal["out", "in"]
    weight: float
    created_by_cycle: UUID | None
    other: NoteOut


class NoteDetail(BaseModel):
    note: NoteOut
    created_by_cycle: UUID | None
    history: list[HistoryEntry]
    links: list[LinkOut]
    sources: list[NoteOut]


class EdgeOut(BaseModel):
    src_id: UUID
    dst_id: UUID
    link_type: LinkType
    weight: float
    created_by_cycle: UUID | None


class GraphOut(BaseModel):
    nodes: list[NoteOut]
    edges: list[EdgeOut]
    total: int  # matching notes; more than len(nodes) when the limit cut the graph


class StatsOut(BaseModel):
    notes: dict[str, dict[str, int]]
    links: int
    cycles: int
    pending_episodes: int


class EditOut(BaseModel):
    note_id: UUID
    tier: Tier
    kind: Literal["superseded", "archived", "theme_updated", "merged", "edited"]
    version: int
    reason: str | None
    before: str
    after: str
    status_now: Status


class CycleLinkOut(BaseModel):
    src_id: UUID
    dst_id: UUID
    link_type: LinkType
    weight: float
    src_content: str
    dst_content: str


class CycleChanges(BaseModel):
    cycle_id: UUID
    clock: datetime
    created: list[NoteOut]
    episodes: list[NoteOut]
    edits: list[EditOut]
    episodes_forgotten: int
    links: list[CycleLinkOut]
    audit: dict[str, Any] | None


def _csv(value: str, enum: type[Tier] | type[Status], name: str) -> list[Any]:
    try:
        out = [enum(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        allowed = ", ".join(e.value for e in enum)
        raise HTTPException(422, f"{name}: expected some of {allowed}") from exc
    if not out:
        raise HTTPException(422, f"{name} is empty")
    return out


@router.get("/memory/stats", response_model=StatsOut)
async def memory_stats(x: ExplorerDep) -> StatsOut:
    return StatsOut.model_validate(await x.stats())


@router.get("/memory/notes", response_model=NotePage)
async def list_notes(
    x: ExplorerDep,
    tier: str = Query("fact,theme", description="Comma-separated: episode,fact,theme"),
    status_: str = Query(
        "active", alias="status", description="Comma-separated: active,superseded,archived"
    ),
    q: str | None = Query(None, max_length=200, description="Substring or exact keyword"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> NotePage:
    notes, total = await x.list_notes(
        tiers=_csv(tier, Tier, "tier"),
        statuses=_csv(status_, Status, "status"),
        q=q,
        limit=limit,
        offset=offset,
    )
    return NotePage(items=[NoteOut.of(n) for n in notes], total=total, limit=limit, offset=offset)


@router.get("/memory/notes/{note_id}", response_model=NoteDetail)
async def get_note(note_id: UUID, x: ExplorerDep) -> NoteDetail:
    d = await x.note_detail(note_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "note not found")
    return NoteDetail(
        note=NoteOut.of(d["note"]),
        created_by_cycle=d["created_by_cycle"],
        history=[HistoryEntry.model_validate(h) for h in d["history"]],
        links=[
            LinkOut(
                **{k: v for k, v in link.items() if k != "other"}, other=NoteOut.of(link["other"])
            )
            for link in d["links"]
        ],
        sources=[NoteOut.of(n) for n in d["sources"]],
    )


@router.get("/memory/graph", response_model=GraphOut)
async def memory_graph(
    x: ExplorerDep,
    tier: str = Query("fact,theme"),
    status_: str = Query("active,superseded", alias="status"),
    limit: int = Query(400, ge=1, le=2000),
) -> GraphOut:
    g = await x.graph(
        tiers=_csv(tier, Tier, "tier"), statuses=_csv(status_, Status, "status"), limit=limit
    )
    return GraphOut(
        nodes=[NoteOut.of(n) for n in g["nodes"]],
        edges=[EdgeOut.model_validate(e) for e in g["edges"]],
        total=g["total"],
    )


@router.get("/sleep/cycles/{cycle_id}/changes", response_model=CycleChanges, tags=["sleep"])
async def cycle_changes(
    cycle_id: UUID, x: ExplorerDep, limit: int = Query(500, ge=1, le=2000)
) -> CycleChanges:
    d = await x.cycle_changes(cycle_id, limit=limit)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "cycle not found")
    return CycleChanges(
        cycle_id=d["cycle_id"],
        clock=d["clock"],
        created=[NoteOut.of(n) for n in d["created"]],
        episodes=[NoteOut.of(n) for n in d["episodes"]],
        edits=[EditOut.model_validate(e) for e in d["edits"]],
        episodes_forgotten=d["episodes_forgotten"],
        links=[CycleLinkOut.model_validate(r) for r in d["links"]],
        audit=d["audit"],
    )


# --- changes made by the person ------------------------------------------------------------


def get_curation(c: ContainerDep) -> Curation:
    return Curation(c.pool, c.store, c.embedder)


CurationDep = Annotated[Curation, Depends(get_curation)]


class CorrectIn(BaseModel):
    content: str = Field(min_length=1, max_length=1000)


class ThemeRef(BaseModel):
    id: UUID
    content: str


class ForgetOut(BaseModel):
    note_id: UUID
    turns_forgotten: int
    themes: list[ThemeRef]  # active themes that still mention it


@router.post("/memory/notes/{note_id}/correct", response_model=NoteOut)
async def correct_note(note_id: UUID, body: CorrectIn, cur: CurationDep) -> NoteOut:
    try:
        return NoteOut.of(await cur.correct(note_id, body.content))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "note not found") from exc
    except CurationError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/memory/notes/{note_id}/forget", response_model=ForgetOut)
async def forget_note(note_id: UUID, cur: CurationDep) -> ForgetOut:
    try:
        return ForgetOut.model_validate(await cur.forget(note_id))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "note not found") from exc
    except CurationError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
