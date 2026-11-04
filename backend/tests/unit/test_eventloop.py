import asyncio
import selectors
import sys

import pytest

from app import eventloop


def run_check(loop_factory=None) -> None:
    async def main() -> None:
        eventloop.check_loop_compatible()

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(main())


def test_loop_factory_uses_select_selector_on_windows(monkeypatch):
    monkeypatch.setattr(eventloop.sys, "platform", "win32")
    loop = eventloop.loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert isinstance(loop._selector, selectors.SelectSelector)  # type: ignore[attr-defined]
    finally:
        loop.close()


def test_loop_from_factory_passes_the_check():
    run_check(eventloop.loop_factory)  # real check on every OS, including Windows CI


def test_check_rejects_proactor_loop(monkeypatch):
    # Simulated on any OS: pretend the running loop's class is Windows' ProactorEventLoop.
    async def main() -> None:
        monkeypatch.setattr(eventloop.sys, "platform", "win32")
        monkeypatch.setattr(
            asyncio, "ProactorEventLoop", type(asyncio.get_running_loop()), raising=False
        )
        with pytest.raises(RuntimeError, match="--loop app.eventloop:loop_factory"):
            eventloop.check_loop_compatible()

    asyncio.run(main())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
def test_windows_default_loop_is_rejected():
    with pytest.raises(RuntimeError, match="ProactorEventLoop"):
        run_check()  # Python's default loop on Windows is the Proactor
