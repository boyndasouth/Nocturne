"""Apply SQL migrations in backend/migrations/ in filename order.

Usage (from backend/):  python -m scripts.migrate
Each file runs in its own transaction and is recorded in schema_migrations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def migrate(database_url: str, *, verbose: bool = True) -> list[str]:
    applied_now: list[str] = []
    # prepare_threshold=None keeps this compatible with Supabase/Neon transaction poolers.
    with psycopg.connect(database_url, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(
            "create table if not exists schema_migrations ("
            " filename text primary key, applied_at timestamptz not null default now())"
        )
        done = {row[0] for row in conn.execute("select filename from schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
                conn.execute("insert into schema_migrations (filename) values (%s)", (path.name,))
            applied_now.append(path.name)
            if verbose:
                print(f"applied {path.name}")
    if verbose and not applied_now:
        print("schema is up to date")
    setup_checkpointer(database_url)
    return applied_now


def setup_checkpointer(database_url: str) -> None:
    """Create/upgrade LangGraph's checkpoint tables (sleep-cycle resume). Idempotent."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    with psycopg.connect(
        database_url, autocommit=True, prepare_threshold=None, row_factory=dict_row
    ) as conn:
        PostgresSaver(conn).setup()  # type: ignore[arg-type]


def main() -> int:
    from app.config import get_settings

    migrate(get_settings().database_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
