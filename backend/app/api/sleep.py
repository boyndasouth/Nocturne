from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import AwareDatetime, BaseModel, Field

from app.api.deps import ContainerDep, require_api_key
from app.llm.providers import ConfigError
from app.memory.cycles import SleepBusy, SleepCycle

log = logging.getLogger("nocturne.sleep")

router = APIRouter(tags=["sleep"], dependencies=[Depends(require_api_key)])

# Strong references to background cycles; asyncio only keeps weak ones.
_background: set[asyncio.Task[object]] = set()


class SleepRequest(BaseModel):
    occurred_at: AwareDatetime | None = Field(
        None,
        description="The cycle's notion of now (benchmark replay). Facts are recorded at this "
        "time and only episodes recorded by then are consolidated. Defaults to now.",
    )


class CycleOut(BaseModel):
    cycle_id: UUID
    status: str
    started_at: datetime
    finished_at: datetime | None
    clock: datetime
    episodes_processed: int
    facts_created: int
    merges: int
    supersessions: int
    links_created: int
    themes_created: int
    themes_updated: int
    audit_score: float | None
    audit_items: int
    archived: int
    tokens_used: int
    llm_calls: int
    attempts: int
    error: str | None

    @classmethod
    def of(cls, c: SleepCycle) -> CycleOut:
        return cls.model_validate(c.model_dump())


class SleepResponse(BaseModel):
    status: Literal["nothing_to_do", "started", "succeeded", "failed"]
    resumed: bool = False
    cycle: CycleOut | None = None


async def cancel_background() -> None:
    """Called at shutdown, before the pool closes. A cancelled cycle is marked failed (or,
    failing that, goes stale) and resumes on the next trigger."""
    tasks = list(_background)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@router.post("/sleep", response_model=SleepResponse)
async def sleep(
    c: ContainerDep,
    response: Response,
    req: SleepRequest | None = None,
    wait: bool = Query(False, description="Run to completion before responding (eval, tests)."),
) -> SleepResponse:
    """Trigger a sleep cycle. A failed cycle is resumed before any new work is picked up."""
    try:
        prepared = await c.sleep_runner.begin(clock=req.occurred_at if req else None)
    except SleepBusy as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if prepared is None:
        return SleepResponse(status="nothing_to_do")

    if not wait:
        task = asyncio.create_task(c.sleep_runner.execute(prepared))
        _background.add(task)
        task.add_done_callback(_background.discard)
        task.add_done_callback(lambda t: t.cancelled() or t.exception())  # error already logged
        response.status_code = status.HTTP_202_ACCEPTED
        return SleepResponse(
            status="started", resumed=prepared.resumed, cycle=CycleOut.of(prepared.cycle)
        )

    try:
        done = await c.sleep_runner.execute(prepared)
    except Exception:
        failed = await c.cycles.get(prepared.cycle.cycle_id)
        return SleepResponse(
            status="failed", resumed=prepared.resumed, cycle=CycleOut.of(failed) if failed else None
        )
    return SleepResponse(status="succeeded", resumed=prepared.resumed, cycle=CycleOut.of(done))


@router.get("/sleep/cycles", response_model=list[CycleOut])
async def list_cycles(c: ContainerDep, limit: int = Query(50, ge=1, le=500)) -> list[CycleOut]:
    return [CycleOut.of(x) for x in await c.cycles.list(limit=limit)]


@router.get("/sleep/cycles/{cycle_id}", response_model=CycleOut)
async def get_cycle(cycle_id: UUID, c: ContainerDep) -> CycleOut:
    cycle = await c.cycles.get(cycle_id)
    if cycle is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "cycle not found")
    return CycleOut.of(cycle)
