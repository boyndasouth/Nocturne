"""FastAPI app factory.

Run (from backend/):
    uvicorn app.main:create_app --factory --reload --loop app.eventloop:loop_factory
The --loop flag is required on Windows (see app/eventloop.py) and harmless elsewhere.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.agents.episode_writer import EpisodeWriter
from app.agents.responder import Responder
from app.agents.sleep.graph import SleepConfig
from app.agents.sleep.runner import (
    CheckpointerFactory,
    RunnerLimits,
    SleepRunner,
    postgres_checkpointer,
)
from app.agents.sleep.trigger import ActivityClock, IdleTrigger, TriggerPolicy
from app.api import account, chat, health, ingest, memory, sleep
from app.api.deps import Container
from app.auth import SupabaseVerifier
from app.config import Settings, get_settings
from app.db import create_pool, verify_schema
from app.eventloop import check_loop_compatible
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder, build_embedder
from app.llm.providers import ConfigError, build_online_client, build_sleep_client
from app.llm.tracing import Tracer, build_tracer
from app.memory.cycles import CycleStore
from app.memory.retriever import HierarchicalRetriever
from app.memory.store import MemoryStore
from app.usage import Guard, Limits, UsageStore

log = logging.getLogger("nocturne")


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    sleep_llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    tracer: Tracer | None = None,
    checkpointer: CheckpointerFactory | None = None,
    verifier: SupabaseVerifier | None = None,
) -> FastAPI:
    """Pass `llm`, `sleep_llm`, `embedder`, `tracer`, `checkpointer` or `verifier` to
    override the configured ones (used by tests)."""
    settings = settings or get_settings()
    if settings.auth_mode == "supabase" and verifier is None:
        if not settings.supabase_url:
            raise ConfigError("AUTH_MODE=supabase needs SUPABASE_URL")
        verifier = SupabaseVerifier(
            settings.supabase_url,
            audience=settings.supabase_audience,
            jwt_secret=settings.supabase_jwt_secret.get_secret_value()
            if settings.supabase_jwt_secret
            else None,
        )

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
            config=settings.retrieval_config,
            tracer=tracer_,
        )
        writer = EpisodeWriter(store, embedder_)
        # Built lazily: chat works without a sleep key; /sleep then returns 503.
        sleep_factory: Callable[[], LLMClient] = (
            (lambda: sleep_llm) if sleep_llm else (lambda: build_sleep_client(settings, tracer_))
        )
        usage = UsageStore(pool)
        guard = Guard(
            usage,
            Limits(
                chats_per_day=settings.limit_chats_per_day or None,
                sleeps_per_day=settings.limit_sleeps_per_day or None,
                answer_tokens_per_day=settings.answer_tokens_per_day or None,
                sleep_requests_per_day=settings.sleep_requests_per_day or None,
            ),
            per_account=settings.auth_mode == "supabase",
        )
        responder = Responder(
            llm=online,
            retriever=retriever,
            store=store,
            writer=writer,
            tracer=tracer_,
            recent_turns=settings.recent_turns,
            reinforce_eta=settings.reinforce_eta,
            reinforce_kappa=settings.reinforce_kappa,
            context_budget_chars=settings.context_budget_chars,
            context_item_max_chars=settings.context_item_max_chars,
            max_tokens=settings.online_max_tokens,
            prompt=settings.responder_prompt,
            temperature=settings.responder_temperature,
            fallback=sleep_factory if settings.answer_fallback else None,
        )
        cycles = CycleStore(pool)
        activity = ActivityClock()
        runner = SleepRunner(
            store=store,
            cycles=cycles,
            embedder=embedder_,
            llm_factory=sleep_factory,
            checkpointer=checkpointer or postgres_checkpointer(settings.database_url),
            config=SleepConfig.from_settings(settings),
            limits=RunnerLimits(
                max_episodes=settings.sleep_max_episodes,
                max_attempts=settings.sleep_max_attempts,
                stale_after=timedelta(minutes=settings.sleep_stale_minutes),
                recursion_limit=settings.sleep_recursion_limit,
            ),
            tracer=tracer_,
            usage=usage,
        )
        trigger = IdleTrigger(
            runner,
            store,
            activity,
            TriggerPolicy(
                idle_seconds=settings.sleep_idle_minutes * 60,
                idle_min_episodes=settings.sleep_idle_min_episodes,
                backlog_episodes=settings.sleep_backlog_episodes,
                poll_seconds=settings.sleep_poll_seconds,
                retry_backoff_seconds=settings.sleep_retry_backoff_minutes * 60,
            ),
            guard=guard,
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
            cycles=cycles,
            sleep_runner=runner,
            activity=activity,
            verifier=verifier if settings.auth_mode == "supabase" else None,
            usage=usage,
            guard=guard,
        )

        warmup = getattr(embedder_, "warmup", None)
        if callable(warmup):  # load the model now, not on the first chat
            await asyncio.to_thread(warmup)
        if settings.sleep_auto:
            trigger.start()
        log.info(
            "Nocturne %s ready (accounts=%s, tracing=%s, auto-sleep=%s)",
            __version__,
            settings.auth_mode,
            tracer_.enabled,
            settings.sleep_auto,
        )
        try:
            yield
        finally:
            await trigger.stop()
            await sleep.cancel_background()
            await runner.aclose()
            if owns_llm:
                await online.aclose()
            await pool.close()
            tracer_.flush()

    app = FastAPI(title="Nocturne", version=__version__, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(sleep.router)
    app.include_router(sleep.service_router)
    app.include_router(memory.router)
    app.include_router(account.router)
    origins = [o.strip().rstrip("/") for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            # Every header the frontend sends; a missing one fails the browser's preflight.
            allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Nocturne-Demo"],
            max_age=600,
        )
    return app
