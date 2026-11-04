"""FastAPI app factory.

Run (from backend/):
    uvicorn app.main:create_app --factory --reload --loop app.eventloop:loop_factory
The --loop flag is required on Windows (see app/eventloop.py) and harmless elsewhere.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.agents.episode_writer import EpisodeWriter
from app.agents.responder import Responder
from app.api import chat, health, ingest
from app.api.deps import Container
from app.config import Settings, get_settings
from app.db import create_pool, verify_schema
from app.eventloop import check_loop_compatible
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder, build_embedder
from app.llm.providers import build_online_client
from app.llm.tracing import Tracer, build_tracer
from app.memory.retriever import HierarchicalRetriever
from app.memory.store import MemoryStore

log = logging.getLogger("nocturne")


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    tracer: Tracer | None = None,
) -> FastAPI:
    """Pass `llm`, `embedder` or `tracer` to override the configured ones (used by tests)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        check_loop_compatible()
        tracer_ = tracer or build_tracer(settings)
        embedder_ = embedder or build_embedder(settings)
        if embedder_.dim != settings.embedding_dim:
            raise ValueError(f"embedder dim {embedder_.dim} != EMBEDDING_DIM")

        owns_llm = llm is None
        online = llm or build_online_client(settings, tracer_)  # fails fast on a missing key

        await verify_schema(settings.database_url, settings.embedding_dim)
        pool = create_pool(settings)
        await pool.open(wait=True, timeout=30)

        store = MemoryStore(pool)
        retriever = HierarchicalRetriever(
            store,
            embedder_,
            weights=settings.score_weights,
            recency_tau_days=settings.recency_tau_days,
            top_k=settings.retrieval_top_k,
            candidate_pool=settings.retrieval_candidate_pool,
            tracer=tracer_,
        )
        writer = EpisodeWriter(store, embedder_)
        responder = Responder(
            llm=online,
            retriever=retriever,
            store=store,
            writer=writer,
            tracer=tracer_,
            recent_turns=settings.recent_turns,
            reinforce_eta=settings.reinforce_eta,
            max_tokens=settings.online_max_tokens,
        )
        app.state.container = Container(
            settings=settings,
            pool=pool,
            store=store,
            embedder=embedder_,
            tracer=tracer_,
            online_llm=online,
            retriever=retriever,
            writer=writer,
            responder=responder,
        )

        warmup = getattr(embedder_, "warmup", None)
        if callable(warmup):  # load the model now, not on the first chat
            await asyncio.to_thread(warmup)
        log.info("Nocturne %s ready (tracing=%s)", __version__, tracer_.enabled)
        try:
            yield
        finally:
            if owns_llm:
                await online.aclose()
            await pool.close()
            tracer_.flush()

    app = FastAPI(title="Nocturne", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    return app
