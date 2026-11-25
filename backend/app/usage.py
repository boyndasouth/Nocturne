"""Daily usage limits for a public deployment.

Two kinds of protection for the free LLM quotas:
- per account (accounts on only): messages and manually started sleeps per UTC day, so one
  person can't use up the day for everyone;
- per model budget, across everyone: chat-model tokens and sleep-model requests per UTC day,
  set a little under the provider's free quota. When the chat model's budget is spent,
  answers come from the sleep model instead; when that is spent too, the demo says so.

Days are UTC. Groq's token quota is a rolling 24 hours, so keep the answer budget a margin
under it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from psycopg_pool import AsyncConnectionPool

ANSWER = "answer"  # the chat model's budget
SLEEP = "sleep"  # the consolidation model's (also the fallback answerer)


def today() -> date:
    return datetime.now(UTC).date()


@dataclass(frozen=True)
class Limits:
    chats_per_day: int | None = None  # per account; None = unlimited
    sleeps_per_day: int | None = None  # per account, manual sleeps only
    answer_tokens_per_day: int | None = None  # across everyone
    sleep_requests_per_day: int | None = None  # across everyone


@dataclass(frozen=True)
class AccountUsage:
    chats: int
    sleeps: int


@dataclass(frozen=True)
class BudgetUsage:
    requests: int
    tokens: int


class UsageStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def account(self, day: date | None = None) -> AccountUsage:
        """The current owner's usage (row-level security picks the owner)."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select chats, sleeps from usage_daily where day = %s", (day or today(),)
            )
            row = await cur.fetchone()
        return AccountUsage(row["chats"], row["sleeps"]) if row else AccountUsage(0, 0)

    async def add_account(self, *, chats: int = 0, sleeps: int = 0) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                insert into usage_daily (day, chats, sleeps) values (%s, %s, %s)
                on conflict (owner_id, day) do update
                    set chats = usage_daily.chats + excluded.chats,
                        sleeps = usage_daily.sleeps + excluded.sleeps
                """,
                (today(), chats, sleeps),
            )

    def _shared(self):  # type: ignore[no-untyped-def]
        """The budget counters are shared and hold no personal data; they're also read by the
        scheduled job, which acts for no account. So: an unscoped connection."""
        return getattr(self._pool, "admin_connection", self._pool.connection)()

    async def budget(self, name: str, day: date | None = None) -> BudgetUsage:
        async with self._shared() as conn:
            cur = await conn.execute(
                "select requests, tokens from llm_usage_daily where day = %s and budget = %s",
                (day or today(), name),
            )
            row = await cur.fetchone()
        return BudgetUsage(row["requests"], int(row["tokens"])) if row else BudgetUsage(0, 0)

    async def add_budget(self, name: str, *, requests: int, tokens: int) -> None:
        if requests <= 0 and tokens <= 0:
            return
        async with self._shared() as conn:
            await conn.execute(
                """
                insert into llm_usage_daily (day, budget, requests, tokens)
                values (%s, %s, %s, %s)
                on conflict (day, budget) do update
                    set requests = llm_usage_daily.requests + excluded.requests,
                        tokens = llm_usage_daily.tokens + excluded.tokens
                """,
                (today(), name, requests, tokens),
            )


class Guard:
    """Decides whether a message or a sleep may run now, and which model answers."""

    def __init__(self, usage: UsageStore, limits: Limits, *, per_account: bool) -> None:
        self.usage = usage
        self.limits = limits
        self.per_account = per_account  # accounts on: per-account limits apply

    async def answer_spent(self) -> bool:
        cap = self.limits.answer_tokens_per_day
        return cap is not None and (await self.usage.budget(ANSWER)).tokens >= cap

    async def sleep_spent(self) -> bool:
        cap = self.limits.sleep_requests_per_day
        return cap is not None and (await self.usage.budget(SLEEP)).requests >= cap

    async def chats_left(self) -> int | None:
        cap = self.limits.chats_per_day
        if not self.per_account or cap is None:
            return None
        return max(0, cap - (await self.usage.account()).chats)

    async def sleeps_left(self) -> int | None:
        cap = self.limits.sleeps_per_day
        if not self.per_account or cap is None:
            return None
        return max(0, cap - (await self.usage.account()).sleeps)
