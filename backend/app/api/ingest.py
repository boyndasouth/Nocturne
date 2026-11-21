from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import AwareDatetime, BaseModel, Field

from app.api.deps import ContainerDep, require_user

router = APIRouter(tags=["ingest"], dependencies=[Depends(require_user)])


class IngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    source: str | None = Field(None, max_length=200, description="e.g. 'Lecture 4: Attention'")
    occurred_at: AwareDatetime | None = Field(
        None, description="When the material is from (valid time). Defaults to now."
    )


class IngestResponse(BaseModel):
    episode_ids: list[UUID]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, c: ContainerDep) -> IngestResponse:
    c.activity.touch()
    notes = await c.writer.write_document(req.text, source=req.source, occurred_at=req.occurred_at)
    return IngestResponse(episode_ids=[n.id for n in notes])
