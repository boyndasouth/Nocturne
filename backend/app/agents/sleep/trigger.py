"""In-process sleep trigger (spec §3.2 triggers a and b; the cron is c).

Runs a cycle when the user has been idle for N minutes with some pending episodes, or as
soon as M episodes have piled up. A plain asyncio loop: no scheduler dependency, and it
stops cleanly with the app. When the free host sleeps, the GitHub Actions cron covers it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.agents.sleep.runner import SleepRunner
from app.memory.cycles import SleepBusy
from app.memory.store import MemoryStore

log = logging.getLogger("nocturne.sleep")


@dataclass(frozen=True)
class TriggerPolicy:
    idle_seconds: float = 600.0
    idle_min_episodes: int = 3
    backlog_episodes: int = 30
    poll_seconds: float = 60.0
    retry_backoff_seconds: float = 600.0


def should_sleep(pending: int, idle_seconds: float, p: TriggerPolicy) -> bool:
    if pending >= p.backlog_episodes:
        return True
    return pending >= max(1, p.idle_min_episodes) and idle_seconds >= p.idle_seconds


class ActivityClock:
    """Last time the user did something. Updated by /chat and /ingest."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last = clock()

    def touch(self) -> None:
        self._last = self._clock()

    def idle_seconds(self) -> float:
        return self._clock() - self._last


class IdleTrigger:
    def __init__(
        self,
        runner: SleepRunner,
        store: MemoryStore,
        activity: ActivityClock,
        policy: TriggerPolicy,
    ) -> None:
        self._runner = runner
        self._store = store
        self._activity = activity
        self._policy = policy
        self._task: asyncio.Task[None] | None = None
        self._failures = 0
        self._next_try = 0.0

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="nocturne-idle-trigger")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def tick(self) -> bool:
        """One check. Returns True if a cycle ran (successfully or not)."""
        if self._runner.busy or time.monotonic() < self._next_try:
            return False
        pending = await self._store.count_pending_episodes()
        if not should_sleep(pending, self._activity.idle_seconds(), self._policy):
            return False
        try:
            await self._runner.run_once()
            self._failures = 0
        except SleepBusy:
            return False
        except Exception:  # logged by the runner; back off so a quota error isn't hammered
            self._failures += 1
            delay = self._policy.retry_backoff_seconds * 2 ** min(self._failures - 1, 5)
            self._next_try = time.monotonic() + delay
            log.warning("idle-triggered sleep failed; next attempt in %.0fs", delay)
        return True

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._policy.poll_seconds)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("idle trigger check failed")
