"""Phase 1 exit check: can chat, episodes persist across restarts, retrieval works."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import psycopg
import pytest

from tests.integration.conftest import FakeLLM, make_test_client, ref_for


def note_row(db_url: str, note_id: str | UUID) -> dict:
    with psycopg.connect(db_url, row_factory=psycopg.rows.dict_row) as conn:
        row = conn.execute("select * from memory_notes where id = %s", (note_id,)).fetchone()
    assert row is not None
    return row


def count_episodes(db_url: str) -> int:
    with psycopg.connect(db_url) as conn:
        return conn.execute("select count(*) from memory_notes where tier = 'episode'").fetchone()[
            0
        ]


def test_health(make_client):
    r = make_client().get("/health")
    assert r.status_code == 200
    assert r.json()["database"] is True


def test_chat_recalls_fact_across_restart_and_reinforces_cited_note(make_client, fresh_db):
    def reply(messages):
        ref = ref_for(messages, "exam is on Friday")
        return f"Your exam is on Friday [{ref}]." if ref else "Noted."

    llm = FakeLLM(reply)
    first = make_client(llm).post("/chat", json={"message": "My linear algebra exam is on Friday"})
    assert first.status_code == 200, first.text
    first_episode = first.json()["episode_id"]
    assert first.json()["citations"] == []

    # A brand-new app instance (simulated restart) must still find the episode.
    second = make_client(llm).post("/chat", json={"message": "When is my exam?"})
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["answer"].startswith("Your exam is on Friday")
    assert body["citations"] == [{"ref": "M1", "note_id": first_episode}]
    assert body["retrieved"][0]["id"] == first_episode

    row = note_row(fresh_db, first_episode)
    assert row["access_count"] == 1 and row["last_accessed"] is not None
    assert row["version"] == 1  # access metadata is not a content edit

    stored_answer = note_row(fresh_db, body["episode_id"])["content"]
    assert "[M1]" not in stored_answer and stored_answer.startswith("User: When is my exam?")
    assert count_episodes(fresh_db) == 2


def test_keyword_and_vector_search_rank_relevant_episode_first(make_client):
    client = make_client()
    for msg in [
        "I had pasta for dinner",
        "HNSW indexes in pgvector trade recall for speed",
        "The weather is cold today",
    ]:
        assert client.post("/chat", json={"message": msg}).status_code == 200

    r = client.post("/chat", json={"message": "how do pgvector HNSW indexes work?"})
    top = r.json()["retrieved"][0]["preview"]
    assert "HNSW indexes in pgvector" in top


def test_occurred_at_sets_valid_and_recorded_time_for_replay(make_client, fresh_db):
    t = datetime(2026, 1, 15, 9, 30, tzinfo=UTC)
    r = make_client().post("/chat", json={"message": "hi", "occurred_at": t.isoformat()})
    row = note_row(fresh_db, r.json()["episode_id"])
    assert row["valid_from"] == t and row["recorded_at"] == t


def test_replay_never_sees_memories_from_the_future(make_client):
    llm = FakeLLM()
    client = make_client(llm, recent_turns=5)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    later = (t0 + timedelta(days=9)).isoformat()
    client.post("/chat", json={"message": "future secret about exams", "occurred_at": later})
    r = client.post("/chat", json={"message": "what about exams?", "occurred_at": t0.isoformat()})
    assert r.json()["retrieved"] == []
    assert "future secret" not in llm.calls[-1][-1]["content"]

    # Without a simulated time (live mode), everything already written is visible.
    r = client.post("/chat", json={"message": "what about exams?"})
    assert any("future secret" in x["preview"] for x in r.json()["retrieved"])


def test_ingest_chunks_document_into_episodes(make_client, fresh_db):
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 150 for i in range(4))
    r = make_client().post(
        "/ingest",
        json={"text": text, "source": "Lecture 4", "occurred_at": "2026-02-01T10:00:00Z"},
    )
    assert r.status_code == 200, r.text
    ids = r.json()["episode_ids"]
    assert len(ids) >= 2
    row = note_row(fresh_db, ids[0])
    assert row["content"].startswith("[Source: Lecture 4]")
    assert row["valid_from"] == datetime(2026, 2, 1, 10, tzinfo=UTC)
    assert row["recorded_at"] > row["valid_from"]


def test_llm_failure_returns_502_but_still_logs_user_message(make_client, fresh_db):
    llm = FakeLLM()
    llm.fail = True
    r = make_client(llm).post("/chat", json={"message": "remember: lab report due Monday"})
    assert r.status_code == 502
    with psycopg.connect(fresh_db) as conn:
        contents = [c for (c,) in conn.execute("select content from memory_notes")]
    assert contents == ["User: remember: lab report due Monday"]


def test_api_key_is_enforced_when_configured(make_client):
    client = make_client(api_key="s3cret")
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    assert (
        client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "nope"}).status_code
        == 401
    )  # noqa: E501
    assert (
        client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "s3cret"}).status_code
        == 200
    )  # noqa: E501
    assert client.get("/health").status_code == 200  # health stays public


def test_hard_delete_is_blocked_by_the_database(make_client, fresh_db):
    episode = make_client().post("/chat", json={"message": "hi"}).json()["episode_id"]
    with psycopg.connect(fresh_db) as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="never hard-deleted"):
            conn.execute("delete from memory_notes where id = %s", (episode,))
        conn.rollback()
        with conn.transaction():
            conn.execute("set local nocturne.allow_delete = 'on'")
            conn.execute("delete from memory_notes where id = %s", (episode,))


def test_migrations_are_idempotent(fresh_db):
    from scripts.migrate import migrate

    assert migrate(fresh_db, verbose=False) == []


def test_startup_fails_with_clear_error_when_schema_missing(db_url):
    from app.config import Settings
    from app.db import SchemaError
    from app.llm.embeddings import HashEmbedder
    from app.main import create_app

    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("drop schema if exists public cascade")
        conn.execute("create schema public")
    app = create_app(
        Settings(database_url=db_url, embedding_backend="hash", _env_file=None),
        llm=FakeLLM(),
        embedder=HashEmbedder(),
    )
    with pytest.raises(SchemaError, match="migrate"), make_test_client(app):
        pass
