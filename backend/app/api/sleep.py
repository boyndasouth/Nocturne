from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import AwareDatetime, BaseModel, Field

from app.api.deps import Container, ContainerDep, require_service, require_user
from app.llm.providers import ConfigError
from app.memory.cycles import SleepBusy, SleepCycle
from app.owner import owner_scope

log = logging.getLogger("nocturne.sleep")

router = APIRouter(tags=["sleep"], dependencies=[Depends(require_user)])

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
    if c.guard is not None:
        if await c.guard.sleeps_left() == 0:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"you've started today's {c.guard.limits.sleeps_per_day} sleeps by hand. Your "
                "conversations will still be consolidated by the scheduled sleep",
            )
        if await c.guard.sleep_spent():
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "the demo has used today's free sleep quota. Your conversations are kept and "
                "will be consolidated tomorrow (the quota resets at midnight UTC)",
            )
    try:
        prepared = await c.sleep_runner.begin(clock=req.occurred_at if req else None)
    except SleepBusy as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    if prepared is None:
        return SleepResponse(status="nothing_to_do")
    if c.usage is not None:
        await c.usage.add_account(sleeps=1)

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


# --- across accounts (the GitHub Actions cron) ---------------------------------------------

service_router = APIRouter(tags=["sleep"])


class SleepAllResponse(BaseModel):
    waiting: int = Field(0, description="Accounts with conversation turns waiting for sleep")
    started: int = Field(description="Accounts whose sleep cycle ran (wait=true)")
    busy: int = Field(description="Accounts already sleeping")
    failed: int = Field(description="Accounts whose cycle couldn't start (see the server log)")


async def _sleep_owners(c: Container, owners: list[str]) -> SleepAllResponse:
    out = SleepAllResponse(waiting=len(owners), started=0, busy=0, failed=0)
    for owner in owners:
        if c.guard is not None and await c.guard.sleep_spent():
            break  # the rest wait for tomorrow's budget
        with owner_scope(owner):
            try:
                prepared = await c.sleep_runner.begin()
            except SleepBusy:
                out.busy += 1
                continue
            except Exception:
                log.exception("couldn't start a sleep cycle for an account")
                out.failed += 1
                continue
            if prepared is None:
                continue
            # One account at a time: free-tier rate limits, and a cycle is cheap to wait for.
            try:
                await c.sleep_runner.execute(prepared)
                out.started += 1
            except Exception:
                out.failed += 1
    return out


@service_router.post(
    "/sleep/all",
    response_model=SleepAllResponse,
    dependencies=[Depends(require_service)],
)
async def sleep_all(
    c: ContainerDep,
    response: Response,
    wait: bool = Query(False, description="Run every cycle before responding (tests)."),
) -> SleepAllResponse:
    """Sleep every account that has conversation turns waiting, one after another. For the
    scheduled job, with the service key (X-API-Key). By default it starts in the background
    and answers 202 at once, so a host's request timeout can't cut it short."""
    if c.guard is not None and await c.guard.sleep_spent():
        return SleepAllResponse(waiting=0, started=0, busy=0, failed=0)
    owners = sorted(await c.store.pending_by_owner())
    if wait:
        return await _sleep_owners(c, owners)
    if owners:
        task = asyncio.create_task(_sleep_owners(c, owners))
        _background.add(task)
        task.add_done_callback(_background.discard)
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        response.status_code = status.HTTP_202_ACCEPTED
    return SleepAllResponse(waiting=len(owners), started=0, busy=0, failed=0)
