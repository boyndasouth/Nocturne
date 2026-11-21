from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import AwareDatetime, BaseModel, Field

from app.api.deps import ContainerDep, require_user
from app.llm.client import LLMError
from app.memory.models import Status, Tier

router = APIRouter(tags=["chat"], dependencies=[Depends(require_user)])


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


class TracedMemory(BaseModel):
    """One retrieved memory: its score parts, how retrieval reached it, and the ref it had
    in the prompt (None if the context budget left it out)."""

    id: UUID
    tier: Tier
    status: Status
    content: str
    valid_from: datetime
    valid_to: datetime | None
    score: float
    semantic: float
    keyword: float
    recency: float
    via: str  # search | theme | link | fallback
    ref: str | None
    cited: bool


class PromptMessage(BaseModel):
    role: str
    content: str


class AnswerTrace(BaseModel):
    """Why the assistant answered as it did: for developers debugging memory."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    memories: list[TracedMemory]
    recent_turns: int = Field(description="Latest conversation turns added for continuity")
    history_notes: int = Field(description="Older versions shown beside current facts")
    messages: list[PromptMessage] = Field(description="Exactly what the model was sent")


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieved: list[RetrievedOut]
    episode_id: UUID
    trace: AnswerTrace


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, c: ContainerDep) -> ChatResponse:
    c.activity.touch()
    try:
        result = await c.responder.respond(req.message, occurred_at=req.occurred_at)
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM provider error: {exc}") from exc
    assert result.episode_id is not None  # remember=True always writes the turn
    cited = {ci.note_id for ci in result.citations}
    trace = AnswerTrace(
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=round(result.latency_ms, 1),
        memories=[
            TracedMemory(
                id=s.note.id,
                tier=s.note.tier,
                status=s.note.status,
                content=s.note.content,
                valid_from=s.note.valid_from,
                valid_to=s.note.valid_to,
                score=round(s.score, 4),
                semantic=round(s.semantic_sim, 4),
                keyword=round(s.keyword_score, 4),
                recency=round(s.recency, 4),
                via=s.via,
                ref=result.refs.get(s.note.id),
                cited=s.note.id in cited,
            )
            for s in result.retrieved
        ],
        recent_turns=result.recent_turns,
        history_notes=result.history_notes,
        messages=[PromptMessage(role=m["role"], content=m["content"]) for m in result.messages],
    )
    return ChatResponse(
        trace=trace,
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
