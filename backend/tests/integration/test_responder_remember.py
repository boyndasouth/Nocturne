"""Responder.respond(remember=False), the side-effect-free path the eval harness uses."""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
from psycopg.rows import dict_row

from tests.integration.conftest import FakeLLM, ref_for

T0 = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)
T1 = datetime(2026, 2, 2, 9, 0, tzinfo=UTC)


def _notes(url: str) -> list[dict]:
    with psycopg.connect(url, row_factory=dict_row) as conn:
        return conn.execute(
            "select id, content, strength, access_count from memory_notes order by recorded_at"
        ).fetchall()


def test_remember_false_answers_without_writing_or_reinforcing(make_client, fresh_db):
    def reply(messages):
        ref = ref_for(messages, "exam")
        return f"Your exam is on Friday [{ref}]." if ref else "Noted."

    client = make_client(llm=FakeLLM(reply))
    client.post(
        "/chat", json={"message": "My exam is on Friday", "occurred_at": T0.isoformat()}
    ).raise_for_status()
    before = _notes(fresh_db)
    assert len(before) == 1

    responder = client.app.state.container.responder
    result = client.portal.call(
        lambda: responder.respond("When is my exam?", occurred_at=T1, remember=False)
    )

    assert result.episode_id is None
    assert [c.note_id for c in result.citations] == [before[0]["id"]]  # it still cites
    assert "exam is on Friday" in result.context
    assert _notes(fresh_db) == before  # no new episode, no reinforcement
