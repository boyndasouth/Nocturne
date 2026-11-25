"""Who is signed in, for the frontend's account menu."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import ContainerDep, UserDep

router = APIRouter(tags=["account"])


class Me(BaseModel):
    auth_mode: Literal["none", "supabase"]
    owner: str
    email: str | None
    demo: bool = False  # exploring the read-only demo memory
    chats_left: int | None = None  # today's messages left; None = no limit
    chats_per_day: int | None = None
    sleeps_left: int | None = None  # today's manual sleeps left
    sleeps_per_day: int | None = None


@router.get("/me", response_model=Me)
async def me(principal: UserDep, c: ContainerDep) -> Me:
    out = Me(
        auth_mode=c.settings.auth_mode,
        owner=principal.owner,
        email=principal.email,
        demo=principal.read_only,
    )
    if c.guard is not None and not principal.read_only:
        out.chats_left = await c.guard.chats_left()
        out.sleeps_left = await c.guard.sleeps_left()
        if out.chats_left is not None:
            out.chats_per_day = c.guard.limits.chats_per_day
        if out.sleeps_left is not None:
            out.sleeps_per_day = c.guard.limits.sleeps_per_day
    return out
