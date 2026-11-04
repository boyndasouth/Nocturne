from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from psycopg_pool import AsyncConnectionPool

from app.agents.episode_writer import EpisodeWriter
from app.agents.responder import Responder
from app.agents.sleep.runner import SleepRunner
from app.agents.sleep.trigger import ActivityClock
from app.config import Settings
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder
from app.llm.tracing import Tracer
from app.memory.cycles import CycleStore
from app.memory.retriever import HierarchicalRetriever
from app.memory.store import MemoryStore


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


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


async def require_api_key(
    container: ContainerDep,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = container.settings.api_key
    if expected is None:  # auth disabled (local dev)
        return
    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")
