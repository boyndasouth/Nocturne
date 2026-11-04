"""Event loop setup for Windows.

psycopg's async mode can't run on Windows' default ProactorEventLoop. Uvicorn only picks a
compatible loop when it runs with --reload, so pass this factory explicitly:

    uvicorn app.main:create_app --factory --loop app.eventloop:loop_factory

On Linux and macOS the factory returns the normal default loop.
"""

from __future__ import annotations

import asyncio
import selectors
import sys


def loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def check_loop_compatible() -> None:
    """Raise early, with the fix, instead of failing deep inside psycopg."""
    proactor = getattr(asyncio, "ProactorEventLoop", None)  # only exists on Windows
    if (
        sys.platform == "win32"
        and proactor is not None
        and isinstance(asyncio.get_running_loop(), proactor)
    ):
        raise RuntimeError(
            "psycopg can't run on Windows' ProactorEventLoop. Start the server with:\n"
            "  uvicorn app.main:create_app --factory --loop app.eventloop:loop_factory"
        )
