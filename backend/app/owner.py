"""Whose memory a request (or a background sleep cycle) is acting on.

The owner lives in a context variable, set once per request by the auth dependency and
inherited by any task the request starts. With accounts on, every connection taken from
the pool is switched to the restricted role `nocturne_app` and tagged with that owner, and
Postgres row-level security (migration 006) does the rest: no query in the codebase needs
an owner filter, and a connection with no owner sees nothing.

With accounts off, the same happens for the single account 'local'. Only the eval harness
uses unscoped connections (MEMORY_ISOLATION=false).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

import psycopg
from psycopg_pool import AsyncConnectionPool

LOCAL_OWNER = "local"
APP_ROLE = "nocturne_app"

_owner: ContextVar[str | None] = ContextVar("nocturne_owner", default=None)


class OwnerMissing(RuntimeError):
    """A scoped connection was requested outside any owner's context (a bug: fail closed)."""


def current_owner() -> str:
    """The owner this code is acting for; 'local' when accounts are off."""
    return _owner.get() or LOCAL_OWNER


@contextmanager
def owner_scope(owner: str) -> Iterator[None]:
    token = _owner.set(owner)
    try:
        yield
    finally:
        _owner.reset(token)


def set_owner(owner: str) -> None:
    """For the request dependency: the request's context is discarded afterwards."""
    _owner.set(owner)


class ScopedPool(AsyncConnectionPool):
    """A pool whose connections act for the current owner when `scoped` is true."""

    def __init__(
        self,
        *args: object,
        scoped: bool = False,
        default_owner: str | None = None,
        **kwargs: object,
    ) -> None:
        """`default_owner`: who a connection acts for outside any request (accounts off:
        'local'). None means no default: such a connection is refused (accounts on)."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.scoped = scoped
        self.default_owner = default_owner

    @asynccontextmanager
    async def connection(  # type: ignore[override]
        self,
        timeout: float | None = None,  # noqa: ASYNC109 (psycopg_pool's signature)
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        async with super().connection(timeout) as conn:
            if self.scoped:
                owner = _owner.get() or self.default_owner
                if not owner:
                    raise OwnerMissing("no owner in context for a scoped connection")
                await conn.execute(
                    "select set_config('role', %s, false), set_config('nocturne.owner', %s, false)",
                    (APP_ROLE, owner),
                )
            yield conn

    @asynccontextmanager
    async def admin_connection(
        self,
        timeout: float | None = None,  # noqa: ASYNC109 (psycopg_pool's signature)
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        """An unscoped connection (the login role, all owners' rows) for the few
        cross-account jobs: finding who has conversations waiting for sleep."""
        async with super().connection(timeout) as conn:
            if self.scoped:
                await conn.execute(
                    "select set_config('role', 'none', false),"
                    " set_config('nocturne.owner', '', false)"
                )
            yield conn
