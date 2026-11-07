"""Phase 3 step 4 against real Postgres: the Forgetter and the Auditor.

Notes start at τ = 30 days. Untouched, a consolidated episode is archived after ~1 τ and an
unused fact outside any theme after ~5 τ (~150 days).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import psycopg

from app.llm.prompts import AUDITOR_SYSTEM
from app.memory.models import LinkType, NewNote, Status, Tier
from tests.integration.conftest import FakeLLM
from tests.integration.test_phase2 import chat, db, sleep
from tests.integration.test_phase3_structure import FACTS
from tests.integration.test_phase3_structure import brain as structure_brain

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)


def brain(messages) -> str:
    if messages[0]["content"] == AUDITOR_SYSTEM:
        items = re.findall(r"^Item (\d+)\n(\w+)", messages[-1]["content"], re.M)
        verdict = {"MERGE": "good", "SUPERSEDE": "good", "LINK": "questionable", "THEME": "good"}
        return json.dumps(
            {
                "verdicts": [
                    {"item": int(n), "verdict": verdict[kind], "note": kind.lower()}
                    for n, kind in items
                ]
            }
        )
    return structure_brain(messages)


def status_of(url: str, content: str) -> str:
    """Status of the one note whose content starts with `content` (episodes add the reply)."""
    [row] = db(url, "select status::text from memory_notes where content like %s", content + "%")
    return row["status"]


def add_fact(client, content: str, at: datetime, **kw):
    c = client.app.state.container
    [vec] = c.embedder.embed_documents([content])
    note = NewNote(tier=Tier.FACT, content=content, embedding=vec, valid_from=at,
                   recorded_at=at, **kw)  # fmt: skip
    [out] = client.portal.call(lambda: c.store.insert_notes([note]))
    return out.id


# --- Forgetter ------------------------------------------------------------------------------


def test_consolidated_turns_are_archived_after_about_a_month_unconsolidated_never(
    make_client, fresh_db
):
    client = make_client(sleep_llm=FakeLLM(brain))
    chat(client, "My exam is on Friday", T0)
    sleep(client, T0 + timedelta(hours=8))  # consolidates the turn
    assert status_of(fresh_db, "User: My exam is on Friday") == "active"

    chat(client, "My road bike is a Specialized", T0 + timedelta(days=20))
    cycle = sleep(client, T0 + timedelta(days=20, hours=8))["cycle"]
    assert cycle["archived"] == 0  # 20 days < ~1 τ
    assert status_of(fresh_db, "User: My exam is on Friday") == "active"

    chat(client, "My commuter bike is a Trek", T0 + timedelta(days=40))
    cycle = sleep(client, T0 + timedelta(days=40, hours=8))["cycle"]
    assert status_of(fresh_db, "User: My exam is on Friday") == "archived"  # 40 days
    assert status_of(fresh_db, "User: My road bike is a Specialized") == "active"  # 20 days
    assert cycle["archived"] == 1
    [v] = db(
        fresh_db,
        "select v.reason from memory_versions v join memory_notes n on n.id = v.note_id"
        " where n.content like %s",
        "User: My exam is on Friday%",
    )
    assert v["reason"].startswith("forgotten")


def test_unused_facts_fade_but_used_themed_and_superseded_ones_stay(make_client, fresh_db):
    client = make_client(sleep_llm=FakeLLM(brain))
    old = T0 - timedelta(days=200)
    unused = add_fact(client, "The user once tried pottery.", old)
    snack = add_fact(client, "The user's go-to snack is almonds.", old)
    with psycopg.connect(fresh_db, autocommit=True) as conn:  # used twice, long ago
        conn.execute("update memory_notes set access_count = 2 where id = %s", (snack,))
    add_fact(client, "The user lived in Austin.", old, status=Status.SUPERSEDED, valid_to=T0)
    # A theme keeps its facts only while it has at least THEME_MIN_FACTS (3) of them active.
    themed = [
        add_fact(client, text, old)
        for text in (
            "The user's car is blue.",
            "The user drives a Toyota.",
            "The user bought a car.",
        )
    ]
    c = client.app.state.container
    [vec] = c.embedder.embed_documents(["Car: blue"])
    [theme] = client.portal.call(
        lambda: c.store.insert_notes(
            [
                NewNote(
                    tier=Tier.THEME,
                    content="Car: blue",
                    embedding=vec,
                    valid_from=old,
                    recorded_at=old,
                )
            ]  # fmt: skip
        )
    )

    async def link():
        async with c.store.transaction() as conn:
            for fact in themed:
                await c.store.add_link(conn, theme.id, fact, LinkType.SUMMARIZES, cycle_id=None)

    client.portal.call(link)

    chat(client, "My exam is on Friday", T0)
    sleep(client, T0 + timedelta(hours=8))
    assert status_of(fresh_db, "The user once tried pottery.") == "archived"
    assert status_of(fresh_db, "The user's go-to snack is almonds.") == "active"  # used twice
    assert status_of(fresh_db, "The user's car is blue.") == "active"  # in a theme
    assert status_of(fresh_db, "The user drives a Toyota.") == "active"
    assert status_of(fresh_db, "The user lived in Austin.") == "superseded"  # history: exempt
    assert status_of(fresh_db, "Car: blue") == "active"  # themes are exempt

    # Archived is not gone: the retriever's fallback still finds it.
    out = client.portal.call(
        lambda: c.retriever.retrieve("Did I ever try pottery?", now=T0 + timedelta(days=1))
    )
    hit = [s for s in out if s.note.id == unused]
    assert hit and hit[0].via == "fallback"


def test_forgetter_can_be_switched_off(make_client, fresh_db):
    client = make_client(sleep_llm=FakeLLM(brain), enable_forgetter=False)
    add_fact(client, "The user once tried pottery.", T0 - timedelta(days=400))
    chat(client, "My exam is on Friday", T0)
    assert sleep(client, T0 + timedelta(hours=8))["cycle"]["archived"] == 0
    assert status_of(fresh_db, "The user once tried pottery.") == "active"


def test_access_grows_tau(make_client, fresh_db):
    def reply(messages):
        m = re.search(r"\[(M\d+)\]", messages[-1]["content"])
        return f"Noted [{m.group(1)}]." if m else "Noted."

    client = make_client(llm=FakeLLM(reply))
    fact = add_fact(client, "The user's exam is on Friday.", T0)
    chat(client, "When is my exam?", T0 + timedelta(days=1))
    [row] = db(fresh_db, "select decay_tau_days, access_count from memory_notes where id = %s",
               fact)  # fmt: skip
    assert (row["decay_tau_days"], row["access_count"]) == (45.0, 1)  # 30 · κ, κ = 1.5


# --- Auditor --------------------------------------------------------------------------------


def _bikes(client):
    for i, phrase in enumerate(list(FACTS)[:3]):
        chat(client, phrase, T0 + timedelta(minutes=i))
    return sleep(client, T0 + timedelta(hours=8))


def test_auditor_scores_a_sample_of_the_cycles_decisions(make_client, fresh_db):
    sleep_llm = FakeLLM(brain)
    client = make_client(sleep_llm=sleep_llm)
    cycle = _bikes(client)["cycle"]
    # 2 links ("questionable") + 1 theme ("good")
    assert cycle["audit_items"] == 3
    assert cycle["audit_score"] == round((0.5 + 0.5 + 1.0) / 3, 4)
    [row] = db(fresh_db, "select audit, tokens_used, llm_calls from sleep_cycles")
    assert row["audit"]["by_kind"] == {"link": 0.5, "theme": 1.0}
    assert {v["kind"] for v in row["audit"]["verdicts"]} == {"link", "theme"}
    assert sleep_llm.trace_names.count("sleep.auditor") == 1
    assert row["llm_calls"] == len(sleep_llm.trace_names)  # the audit call is counted


def test_a_failed_audit_never_fails_the_cycle(make_client, fresh_db):
    sleep_llm = FakeLLM(brain)
    sleep_llm.fail_on = ["sleep.auditor"]
    client = make_client(sleep_llm=sleep_llm)
    out = _bikes(client)
    assert out["status"] == "succeeded" and out["cycle"]["audit_score"] is None
    [row] = db(fresh_db, "select audit from sleep_cycles")
    assert "error" in row["audit"]
    assert db(fresh_db, "select count(*) as n from memory_notes where tier = 'theme'")[0]["n"] == 1


def test_auditor_off_or_nothing_to_audit(make_client, fresh_db):
    sleep_llm = FakeLLM(brain)
    client = make_client(sleep_llm=sleep_llm, enable_auditor=False)
    assert _bikes(client)["cycle"]["audit_score"] is None
    assert "sleep.auditor" not in sleep_llm.trace_names
