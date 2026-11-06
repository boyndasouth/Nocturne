"""The eval database: migrate, reset between conversations, and measure memory size.

Benchmarks need one fresh memory per conversation, so reset() TRUNCATEs the memory tables.
That bypasses the no-hard-delete rule on purpose (TRUNCATE doesn't fire row triggers), which
is why the database name must contain "eval" or "test": never point this at real memory.
"""

from __future__ import annotations

import psycopg
from psycopg_pool import AsyncConnectionPool

from eval.base import PrepStats
from scripts.migrate import migrate

_TABLES = "memory_links, memory_versions, memory_notes, sleep_cycles"


class UnsafeDatabaseError(RuntimeError):
    pass


def check_disposable(database_url: str) -> str:
    dbname = str(psycopg.conninfo.conninfo_to_dict(database_url).get("dbname", ""))
    if "eval" not in dbname and "test" not in dbname:
        raise UnsafeDatabaseError(
            f"refusing to use database {dbname!r} for evaluation: it is wiped before every "
            "conversation. Use a database whose name contains 'eval' (e.g. nocturne_eval)."
        )
    return dbname


class EvalDatabase:
    def __init__(self, database_url: str, pool: AsyncConnectionPool) -> None:
        check_disposable(database_url)
        self.url = database_url
        self._pool = pool

    @staticmethod
    def prepare_schema(database_url: str) -> None:
        check_disposable(database_url)
        migrate(database_url, verbose=False)

    async def reset(self) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(f"truncate {_TABLES} restart identity cascade")

    async def stats(self) -> PrepStats:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select tier::text as tier, count(*) as n, coalesce(sum(length(content)), 0) as c"
                " from memory_notes where status = 'active' group by tier"
            )
            rows = await cur.fetchall()
            cur = await conn.execute(
                "select count(*) as n, coalesce(sum(facts_created), 0) as facts,"
                " coalesce(sum(merges), 0) as merges,"
                " coalesce(sum(supersessions), 0) as sups"
                " from sleep_cycles where status = 'succeeded'"
            )
            cyc = await cur.fetchone()
            cur = await conn.execute(
                "select count(*) as n from memory_links"
                " where link_type not in ('supersedes', 'summarizes')"
            )
            links = await cur.fetchone()
        by_tier = {r["tier"]: int(r["n"]) for r in rows}
        assert cyc is not None and links is not None
        return PrepStats(
            memory_items=sum(by_tier.values()),
            memory_chars=sum(int(r["c"]) for r in rows),
            by_tier=by_tier,
            sleep_cycles=int(cyc["n"]),
            facts_created=int(cyc["facts"]),
            merges=int(cyc["merges"]),
            supersessions=int(cyc["sups"]),
            links=int(links["n"]),
        )
