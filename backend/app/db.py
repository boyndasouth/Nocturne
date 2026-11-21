"""Postgres connection pool and startup schema checks."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import Settings


class SchemaError(RuntimeError):
    pass


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)
    await conn.commit()  # the pool requires connections to be returned idle


def create_pool(settings: Settings) -> AsyncConnectionPool:
    # prepare_threshold=None: server-side prepared statements break behind PgBouncer-style
    # transaction poolers (Supabase port 6543, Neon "-pooler" hosts). Disabling them costs
    # little at this scale and makes any connection string safe.
    return AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        open=False,
        configure=_configure,
        check=AsyncConnectionPool.check_connection,  # free tiers drop idle connections
        kwargs={"prepare_threshold": None, "row_factory": dict_row},
        name="nocturne",
    )


async def verify_schema(database_url: str, expected_dim: int) -> None:
    """Fail fast with an actionable message before the pool starts retrying in the dark."""
    async with await psycopg.AsyncConnection.connect(
        database_url, prepare_threshold=None, row_factory=dict_row
    ) as conn:
        ext = await (
            await conn.execute("select 1 from pg_extension where extname = 'vector'")
        ).fetchone()
        if ext is None:
            raise SchemaError(
                "pgvector isn't enabled in this database. Run `python -m scripts.migrate`."
            )
        row = await (
            await conn.execute(
                "select atttypmod as dim from pg_attribute "
                "where attrelid = to_regclass('memory_notes') and attname = 'embedding'"
            )
        ).fetchone()
        if row is None:
            raise SchemaError("memory_notes table not found. Run `python -m scripts.migrate`.")
        phase2 = await (
            await conn.execute(
                "select to_regclass('checkpoints') is not null as ok,"
                " exists (select 1 from pg_attribute where attrelid ="
                " to_regclass('memory_notes') and attname = 'consolidated_in') as ok2"
            )
        ).fetchone()
        if phase2 is None or not (phase2["ok"] and phase2["ok2"]):
            raise SchemaError(
                "The database is missing the Phase 2 sleep-cycle schema. "
                "Run `python -m scripts.migrate`."
            )
        forgotten = await (
            await conn.execute(
                "select 1 from pg_enum e join pg_type t on t.oid = e.enumtypid"
                " where t.typname = 'note_status' and e.enumlabel = 'forgotten'"
            )
        ).fetchone()
        if forgotten is None:
            raise SchemaError(
                "The database is missing migration 005 (forgetting from the UI). "
                "Run `python -m scripts.migrate`."
            )
        if row["dim"] != expected_dim:
            raise SchemaError(
                f"memory_notes.embedding is vector({row['dim']}) but EMBEDDING_DIM={expected_dim}."
                " Use an embedding model with that dimension or add a migration."
            )
