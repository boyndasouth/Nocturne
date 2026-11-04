from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import AwareDatetime, BaseModel, Field

from app.api.deps import ContainerDep, require_api_key
from app.llm.client import LLMError
from app.memory.models import Tier

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    occurred_at: AwareDatetime | None = Field(
        None, description="Override the turn's timestamp (benchmark replay). Defaults to now."
    )


class CitationOut(BaseModel):
    ref: str
    note_id: UUID


class RetrievedOut(BaseModel):
    id: UUID
    tier: Tier
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved: list[RetrievedOut]
    episode_id: UUID


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, c: ContainerDep) -> ChatResponse:
    try:
        result = await c.responder.respond(req.message, occurred_at=req.occurred_at)
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM provider error: {exc}") from exc
    return ChatResponse(
        answer=result.answer,
        citations=[CitationOut(ref=ci.ref, note_id=ci.note_id) for ci in result.citations],
        retrieved=[
            RetrievedOut(
                id=s.note.id,
                tier=s.note.tier,
                score=round(s.score, 4),
                preview=s.note.content[:200],
            )
            for s in result.retrieved
        ],
        episode_id=result.episode_id,
    )
