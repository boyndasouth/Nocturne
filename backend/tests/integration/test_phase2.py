"""Phase 2 exit check and sleep-cycle behaviour against real Postgres, with the real LangGraph
Postgres checkpointer and a scripted sleep LLM.

Exit check (spec §10): tell it a fact, then a contradicting fact; after sleep it answers with
the new fact and can report the old one with dates.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime

import psycopg
from psycopg.rows import dict_row

from app.llm.prompts import CONTRADICTION_SYSTEM, DEDUP_SYSTEM, EXTRACTOR_SYSTEM
from tests.integration.conftest import FakeLLM, ref_for

JAN = datetime(2026, 1, 10, 9, 0, tzinfo=UTC)
JAN_SLEEP = datetime(2026, 1, 11, 3, 0, tzinfo=UTC)
MAR = datetime(2026, 3, 2, 18, 0, tzinfo=UTC)
MAR_SLEEP = datetime(2026, 3, 3, 3, 0, tzinfo=UTC)
ASK = datetime(2026, 3, 5, 12, 0, tzinfo=UTC)

# What a good extractor would say for each user message in these tests.
FACTS = {
    "I live in Austin": "The user lives in Austin.",
    "I moved to Dallas": "The user lives in Dallas.",
    "Back in January I lived in Austin": "The user lives in Austin.",
    "My exam is on Friday": "The user's linear algebra exam is on Friday.",
}


def sleep_brain(messages) -> str:
    """Scripted sleep LLM: extraction by phrase table; no LLM-judged duplicates; 'lives in'
    facts about different cities conflict."""
    system, user = messages[0]["content"], messages[-1]["content"]
    if system == EXTRACTOR_SYSTEM:
        facts = []
        for n, body in re.findall(
            r"^\[(\d+)\] \([^)]*\)\n(.*?)(?=^\[\d+\] |\Z)", user, re.M | re.S
        ):
            for phrase, statement in FACTS.items():
                if f"User: {phrase}" in body:
                    facts.append({"statement": statement, "keywords": [], "sources": [int(n)]})
        return json.dumps({"facts": facts})
    groups = re.findall(r"Group (\d+)\nNEW: (.*?)\nEXISTING:\n(.*?)(?=\n\nGroup |\Z)", user, re.S)
    if system == DEDUP_SYSTEM:
        return json.dumps(
            {"decisions": [{"group": int(g), "duplicate_of": None} for g, *_ in groups]}
        )
    assert system == CONTRADICTION_SYSTEM
    decisions = []
    for g, new, existing in groups:
        city = re.search(r"lives in (\w+)", new)
        hits = [
            letter
            for letter, line in re.findall(r"^\[([A-Z])\] (.*)$", existing, re.M)
            if city and "lives in" in line and city.group(1) not in line
        ]
        decisions.append({"group": int(g), "conflicts_with": hits, "reason": "moved"})
    return json.dumps({"decisions": decisions})


def db(url: str, query: str, *args):
    with psycopg.connect(url, row_factory=dict_row) as conn:
        return conn.execute(query, args).fetchall()


def facts(url: str):
    return db(
        url,
        "select id, content, status::text, valid_from, valid_to, recorded_at, version,"
        " source_episode_ids, strength from memory_notes where tier = 'fact' order by valid_from",
    )


def chat(client, message: str, when: datetime):
    r = client.post("/chat", json={"message": message, "occurred_at": when.isoformat()})
    assert r.status_code == 200, r.text
    return r.json()


def sleep(client, when: datetime | None = None):
    body = {"occurred_at": when.isoformat()} if when else None
    r = client.post("/sleep", params={"wait": "true"}, json=body)
    assert r.status_code == 200, r.text
    return r.json()


# --- exit check -----------------------------------------------------------------------------


def test_exit_check_new_fact_wins_and_old_one_is_reported_with_dates(make_client, fresh_db):
    def answer(messages):
        dallas = ref_for(messages, "FACT (true since 2026-03-02) The user lives in Dallas")
        austin = ref_for(messages, "previously (2026-01-10 to 2026-03-02) The user lives in Austin")
        if dallas and austin:
            return (
                f"You live in Dallas [{dallas}]; before that, Austin (Jan 10 to Mar 2) [{austin}]."
            )
        return "Noted."

    client = make_client(FakeLLM(answer), sleep_llm=FakeLLM(sleep_brain))

    chat(client, "I live in Austin", JAN)
    first = sleep(client, JAN_SLEEP)
    assert first["status"] == "succeeded"
    assert first["cycle"]["facts_created"] == 1 and first["cycle"]["episodes_processed"] == 1

    chat(client, "I moved to Dallas", MAR)
    second = sleep(client, MAR_SLEEP)
    assert second["status"] == "succeeded"
    assert second["cycle"]["supersessions"] == 1

    austin, dallas = facts(fresh_db)
    assert (austin["content"], austin["status"]) == ("The user lives in Austin.", "superseded")
    assert austin["valid_from"] == JAN and austin["valid_to"] == MAR  # bitemporal: valid time
    assert austin["recorded_at"] == JAN_SLEEP  # transaction time: when the system learned it
    assert austin["version"] == 2
    assert (dallas["status"], dallas["valid_from"], dallas["valid_to"]) == ("active", MAR, None)
    [link] = db(fresh_db, "select src_id, dst_id, link_type::text from memory_links")
    assert (link["src_id"], link["dst_id"], link["link_type"]) == (
        dallas["id"],
        austin["id"],
        "supersedes",
    )

    # The pre-edit row is kept in full (rollback material), including the embedding.
    [snap] = db(fresh_db, "select * from memory_versions where note_id = %s", austin["id"])
    assert snap["version"] == 1 and str(snap["changed_by_cycle"]) == second["cycle"]["cycle_id"]
    assert snap["snapshot"]["status"] == "active" and snap["snapshot"]["valid_to"] is None
    assert snap["snapshot"]["embedding"].startswith("[") and "search_tsv" not in snap["snapshot"]
    assert snap["reason"].startswith(f"superseded by {dallas['id']}")

    r = chat(client, "Where do I live?", ASK)
    assert r["answer"].startswith("You live in Dallas")
    cited = {c["note_id"] for c in r["citations"]}
    assert cited == {str(dallas["id"]), str(austin["id"])}


# --- merges, ordering, provenance ---------------------------------------------------------


def test_repeated_fact_is_merged_not_duplicated(make_client, fresh_db):
    client = make_client(sleep_llm=FakeLLM(sleep_brain))
    e1 = chat(client, "I live in Austin", JAN)["episode_id"]
    sleep(client, JAN_SLEEP)
    e2 = chat(client, "I live in Austin", MAR)["episode_id"]
    cycle = sleep(client, MAR_SLEEP)["cycle"]

    assert (cycle["facts_created"], cycle["merges"]) == (0, 1)
    [f] = facts(fresh_db)
    assert [str(x) for x in f["source_episode_ids"]] == [e1, e2]
    assert f["version"] == 2 and f["valid_from"] == JAN
    [snap] = db(fresh_db, "select reason, snapshot from memory_versions")
    assert snap["reason"].startswith("merged") and len(snap["snapshot"]["source_episode_ids"]) == 1


def test_backdated_material_does_not_override_a_newer_fact(make_client, fresh_db):
    """Notes from January, ingested after the user moved, become history, not the present."""
    client = make_client(sleep_llm=FakeLLM(sleep_brain))
    chat(client, "I moved to Dallas", MAR)
    sleep(client, MAR_SLEEP)
    chat(client, "Back in January I lived in Austin", JAN)  # valid & recorded in January...
    sleep(client, ASK)  # ...but consolidated in March

    austin, dallas = facts(fresh_db)
    assert dallas["status"] == "active" and dallas["version"] == 1
    assert (austin["status"], austin["valid_from"], austin["valid_to"]) == ("superseded", JAN, MAR)


def test_assistant_text_and_unrelated_turns_produce_no_facts(make_client, fresh_db):
    client = make_client(FakeLLM(lambda _m: "I live in Austin"), sleep_llm=FakeLLM(sleep_brain))
    chat(client, "hello there", JAN)  # the assistant "said" a fact; the user didn't
    r = sleep(client, JAN_SLEEP)
    assert r["cycle"]["facts_created"] == 0 and r["cycle"]["episodes_processed"] == 1
    assert facts(fresh_db) == []
    assert sleep(client)["status"] == "nothing_to_do"  # episode is consolidated


# --- resume, concurrency, operations ------------------------------------------------------


def test_failed_cycle_resumes_from_checkpoint_without_redoing_extraction(make_client, fresh_db):
    sleep_llm = FakeLLM(sleep_brain)
    client = make_client(sleep_llm=sleep_llm)
    chat(client, "I live in Austin", JAN)
    sleep(client, JAN_SLEEP)
    chat(client, "I moved to Dallas", MAR)

    sleep_llm.fail_on = ["sleep.contradiction"]  # e.g. daily quota hit mid-cycle
    sleep_llm.trace_names.clear()
    failed = sleep(client, MAR_SLEEP)
    assert failed["status"] == "failed" and "rate limited" in failed["cycle"]["error"]
    assert [f["status"] for f in facts(fresh_db)] == ["active"]  # nothing half-applied
    assert db(fresh_db, "select count(*) as n from checkpoints")[0]["n"] > 0

    resumed = sleep(client)
    assert resumed["status"] == "succeeded" and resumed["resumed"] is True
    assert resumed["cycle"]["cycle_id"] == failed["cycle"]["cycle_id"]
    assert resumed["cycle"]["attempts"] == 2 and resumed["cycle"]["supersessions"] == 1
    assert sleep_llm.trace_names.count("sleep.extract") == 1  # not re-extracted
    assert sleep_llm.trace_names.count("sleep.contradiction") == 2  # failed one + retry
    assert db(fresh_db, "select count(*) as n from checkpoints")[0]["n"] == 0  # cleaned up


def test_only_one_cycle_runs_at_a_time(make_client, fresh_db):
    client = make_client(sleep_llm=FakeLLM(sleep_brain))
    chat(client, "I live in Austin", JAN)
    db_exec = "insert into sleep_cycles (status) values ('running') returning cycle_id"
    with psycopg.connect(fresh_db) as conn:
        conn.execute(db_exec)
    r = client.post("/sleep", params={"wait": "true"})
    assert r.status_code == 409


def test_background_sleep_and_cycle_history(make_client):
    client = make_client(sleep_llm=FakeLLM(sleep_brain))
    chat(client, "My exam is on Friday", JAN)
    r = client.post("/sleep")
    assert r.status_code == 202 and r.json()["status"] == "started"
    cycle_id = r.json()["cycle"]["cycle_id"]

    deadline = time.monotonic() + 10
    while (c := client.get(f"/sleep/cycles/{cycle_id}").json())["status"] == "running":
        assert time.monotonic() < deadline, "background cycle didn't finish"
        time.sleep(0.05)
    assert c["status"] == "succeeded" and c["facts_created"] == 1 and c["llm_calls"] >= 1
    assert [x["cycle_id"] for x in client.get("/sleep/cycles").json()] == [cycle_id]
    assert client.get(f"/sleep/cycles/{'0' * 8}-0000-0000-0000-{'0' * 12}").status_code == 404


def test_sleep_without_a_sleep_llm_key_is_503_and_chat_still_works(make_client):
    client = make_client(gemini_api_key=None, sleep_provider="gemini")
    chat(client, "I live in Austin", JAN)
    r = client.post("/sleep", params={"wait": "true"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
    assert client.get("/sleep/cycles").json() == []  # nothing claimed


async def _tick(client, **policy) -> bool:
    from app.agents.sleep.trigger import IdleTrigger, TriggerPolicy

    c = client.app.state.container
    trigger = IdleTrigger(c.sleep_runner, c.store, c.activity, TriggerPolicy(**policy))
    return await trigger.tick()


def test_idle_trigger_sleeps_on_backlog_but_not_mid_conversation(make_client, fresh_db):
    client = make_client(sleep_llm=FakeLLM(sleep_brain))
    chat(client, "I live in Austin", JAN)
    chat(client, "My exam is on Friday", JAN)
    run = client.portal.call  # run on the app's event loop

    assert run(lambda: _tick(client, idle_seconds=600, idle_min_episodes=1)) is False  # active
    assert run(lambda: _tick(client, idle_seconds=0, idle_min_episodes=3)) is False  # too few
    assert run(lambda: _tick(client, idle_seconds=600, backlog_episodes=2)) is True  # backlog
    assert len(facts(fresh_db)) == 2
    assert run(lambda: _tick(client, idle_seconds=0, idle_min_episodes=1)) is False  # all done


def test_reembed_rewrites_vectors_without_versioning(make_client, fresh_db, monkeypatch):
    from app.config import get_settings
    from scripts.reembed import reembed

    chat(make_client(), "I live in Austin", JAN)
    with psycopg.connect(fresh_db) as conn:
        conn.execute("update memory_notes set embedding = null")
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    get_settings.cache_clear()
    try:
        assert reembed(fresh_db, batch_size=1) == 1
    finally:
        get_settings.cache_clear()
    [row] = db(fresh_db, "select embedding is not null as has, version from memory_notes")
    assert row["has"] and row["version"] == 1
