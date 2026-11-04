"""Integration tests against a real Postgres + pgvector.

Set TEST_DATABASE_URL to a *disposable* database whose name contains "test", for example the
nocturne_test database that docker-compose creates. The public schema is wiped before each
test. Without TEST_DATABASE_URL these tests are skipped.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Sequence

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.eventloop import loop_factory
from app.llm.client import ChatMessage, LLMError, LLMResponse
from app.llm.embeddings import HashEmbedder
from app.llm.tracing import Tracer
from app.main import create_app
from scripts.migrate import migrate

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def db_url() -> str:
    if not TEST_DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    dbname = psycopg.conninfo.conninfo_to_dict(TEST_DB_URL).get("dbname", "")
    if "test" not in str(dbname):
        pytest.exit(f"refusing to wipe database {dbname!r}: name must contain 'test'")
    return TEST_DB_URL


@pytest.fixture
def fresh_db(db_url: str) -> str:
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("drop schema if exists public cascade")
        conn.execute("create schema public")
    migrate(db_url, verbose=False)
    return db_url


class FakeLLM:
    """Scripted LLM. `reply` receives the prompt messages and returns the answer text.
    `fail_on` makes calls with those trace names raise (consumed once per entry)."""

    name = "fake"
    model = "fake-model"

    def __init__(self, reply: Callable[[Sequence[ChatMessage]], str] | None = None) -> None:
        self.reply = reply or (lambda _messages: "Noted.")
        self.calls: list[list[ChatMessage]] = []
        self.trace_names: list[str] = []
        self.fail = False
        self.fail_on: list[str] = []

    async def chat(
        self, messages: Sequence[ChatMessage], *, trace_name: str = "llm.chat", **_: object
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.trace_names.append(trace_name)
        if self.fail:
            raise LLMError("provider down")
        if trace_name in self.fail_on:
            self.fail_on.remove(trace_name)
            raise LLMError(f"{trace_name}: rate limited")
        return LLMResponse(
            text=self.reply(messages), model=self.model, input_tokens=10, output_tokens=5,
            latency_ms=1.0,
        )  # fmt: skip

    async def aclose(self) -> None:
        pass


def ref_for(messages: Sequence[ChatMessage], needle: str) -> str | None:
    """Find the [M#] ref of the memory line containing `needle` in the prompt."""
    for line in messages[-1]["content"].splitlines():
        m = re.match(r"\s*\[(M\d+)\]", line)
        if m and needle in line:
            return m.group(1)
    return None


def make_test_client(app: FastAPI) -> TestClient:
    """TestClient on a psycopg-compatible event loop (matters on Windows)."""
    return TestClient(app, backend_options={"loop_factory": loop_factory})


@pytest.fixture
def make_client(fresh_db: str) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(
        llm: FakeLLM | None = None, sleep_llm: FakeLLM | None = None, **overrides: object
    ) -> TestClient:
        overrides.setdefault("sleep_auto", False)
        settings = Settings(
            database_url=fresh_db, embedding_backend="hash", _env_file=None, **overrides
        )
        app = create_app(
            settings,
            llm=llm or FakeLLM(),
            sleep_llm=sleep_llm,
            embedder=HashEmbedder(),
            tracer=Tracer(),
        )
        client = make_test_client(app)
        client.__enter__()  # run lifespan
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
