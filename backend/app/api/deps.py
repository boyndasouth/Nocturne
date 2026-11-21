from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg_pool import AsyncConnectionPool

from app.agents.episode_writer import EpisodeWriter
from app.agents.responder import Responder
from app.agents.sleep.runner import SleepRunner
from app.agents.sleep.trigger import ActivityClock
from app.auth import AuthError, Principal, SupabaseVerifier
from app.config import Settings
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder
from app.llm.tracing import Tracer
from app.memory.cycles import CycleStore
from app.memory.retriever import HierarchicalRetriever
from app.memory.store import MemoryStore
from app.owner import LOCAL_OWNER, set_owner


@dataclass
class Container:
    settings: Settings
    pool: AsyncConnectionPool
    store: MemoryStore
    embedder: Embedder
    tracer: Tracer
    online_llm: LLMClient
    retriever: HierarchicalRetriever
    writer: EpisodeWriter
    responder: Responder
    cycles: CycleStore
    sleep_runner: SleepRunner
    activity: ActivityClock
    verifier: SupabaseVerifier | None = None  # set when AUTH_MODE=supabase


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


_bearer = HTTPBearer(auto_error=False)


def _key_matches(given: str | None, expected: str) -> bool:
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


async def authenticate(
    request: Request,
    container: ContainerDep,
    x_api_key: Annotated[str | None, Header()] = None,
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> Principal:
    """Who is calling, and whose memory every query in this request touches.

    AUTH_MODE=none: one memory ('local'); X-API-Key is required only if API_KEY is set.
    AUTH_MODE=supabase: a Supabase access token (Authorization: Bearer) selects the user's
    own memory. The service key (X-API-Key = API_KEY) is accepted only where an endpoint
    asks for it (the cron's POST /sleep/all).
    """
    s = container.settings
    if s.auth_mode == "none":
        expected = s.api_key
        if expected is not None and not _key_matches(x_api_key, expected.get_secret_value()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")
        principal = Principal(owner=LOCAL_OWNER)
    elif bearer is not None and container.verifier is not None:
        try:
            principal = await container.verifier.verify(bearer.credentials)
        except AuthError as exc:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, str(exc), headers={"WWW-Authenticate": "Bearer"}
            ) from exc
    elif s.api_key is not None and _key_matches(x_api_key, s.api_key.get_secret_value()):
        principal = Principal(owner="", service=True)
    else:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "sign in first", headers={"WWW-Authenticate": "Bearer"}
        )
    if not principal.service:
        set_owner(principal.owner)  # scopes every connection this request (and its tasks) use
    request.state.principal = principal
    return principal


async def require_user(principal: Annotated[Principal, Depends(authenticate)]) -> Principal:
    """A person's memory: the service key can't use these endpoints (it has no owner)."""
    if principal.service:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "the service key can't act as a user")
    return principal


async def require_service(principal: Annotated[Principal, Depends(authenticate)]) -> Principal:
    """Jobs across accounts. With accounts off, the local caller is the only account."""
    if not principal.service and principal.owner != LOCAL_OWNER:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "needs the service key")
    return principal


UserDep = Annotated[Principal, Depends(require_user)]
