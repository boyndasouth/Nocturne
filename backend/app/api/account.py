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


@router.get("/me", response_model=Me)
async def me(principal: UserDep, c: ContainerDep) -> Me:
    return Me(auth_mode=c.settings.auth_mode, owner=principal.owner, email=principal.email)
