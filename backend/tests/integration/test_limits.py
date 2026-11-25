"""Protecting the free LLM quotas in a public deployment: per-account daily limits, the
daily budgets shared by everyone (with the sleep model answering once the chat model's is
spent), and the read-only demo memory."""

from __future__ import annotations

import psycopg
import pytest

from app.auth import SupabaseVerifier
from app.llm.client import QuotaExhaustedError
from tests.integration.conftest import FakeLLM
from tests.integration.test_accounts import ALICE, BOB, JWKS, URL, auth
from tests.integration.test_phase2 import JAN, JAN_SLEEP, sleep_brain


def sleep_and_answer(messages):
    """The sleep model's script, plus chat answers for when it stands in as the fallback."""
    if "long-term memory" in messages[0]["content"] and "[M" in messages[0]["content"]:
        return "Answered by the fallback model."
    return sleep_brain(messages)


def accounts_client(make_client, **overrides):
    settings = dict(
        sleep_llm=FakeLLM(sleep_brain),
        enable_auditor=False,
        auth_mode="supabase",
        supabase_url=URL,
        api_key="service-key",
        verifier=SupabaseVerifier(URL, jwks=JWKS),
    )
    settings.update(overrides)
    return make_client(**settings)


def say(client, who: str, message: str = "I live in Austin", minute: int = 0):
    return client.post(
        "/chat",
        json={"message": message, "occurred_at": JAN.replace(minute=minute).isoformat()},
        headers=auth(who),
    )


def test_each_account_has_a_daily_message_limit(make_client):
    client = accounts_client(make_client, limit_chats_per_day=2)
    assert say(client, ALICE).json()["chats_left"] == 1
    assert say(client, ALICE, minute=1).json()["chats_left"] == 0
    r = say(client, ALICE, minute=2)
    assert r.status_code == 429 and "today's 2 messages" in r.json()["detail"]
    # reading still works, and other people aren't affected
    assert client.get("/memory/notes", headers=auth(ALICE)).status_code == 200
    assert say(client, BOB).status_code == 200
    me = client.get("/me", headers=auth(ALICE)).json()
    assert (me["chats_left"], me["chats_per_day"]) == (0, 2)


def test_manual_sleeps_are_limited_per_account(make_client):
    client = accounts_client(make_client, limit_sleeps_per_day=1)
    say(client, ALICE)
    body = {"occurred_at": JAN_SLEEP.isoformat()}
    first = client.post("/sleep", params={"wait": "true"}, json=body, headers=auth(ALICE))
    assert first.json()["status"] == "succeeded"
    say(client, ALICE, "My exam is on Friday", minute=5)
    again = client.post("/sleep", params={"wait": "true"}, json=body, headers=auth(ALICE))
    assert again.status_code == 429 and "scheduled sleep" in again.json()["detail"]
    # the scheduled job still consolidates it
    r = client.post("/sleep/all", params={"wait": "true"}, headers={"X-API-Key": "service-key"})
    assert r.json()["started"] == 1


def test_spent_answer_budget_switches_to_the_sleep_model_then_stops(make_client):
    sleep_llm = FakeLLM(sleep_and_answer)
    client = accounts_client(
        make_client, sleep_llm=sleep_llm, answer_tokens_per_day=1, sleep_requests_per_day=1
    )
    first = say(client, ALICE).json()
    assert first["trace"]["fallback"] is False  # 15 tokens: the answer budget is now spent
    second = say(client, ALICE, "Where do I live?", minute=1).json()
    assert second["trace"]["fallback"] is True
    assert sleep_llm.trace_names[-1] == "responder.fallback"
    # the fallback's one request spent the sleep budget too: the demo says so plainly
    third = say(client, ALICE, "And now?", minute=2)
    assert third.status_code == 503 and "today's free model quota" in third.json()["detail"]
    r = client.post("/sleep", json={"occurred_at": JAN_SLEEP.isoformat()}, headers=auth(ALICE))
    assert r.status_code == 503 and "sleep quota" in r.json()["detail"]
    r = client.post("/sleep/all", headers={"X-API-Key": "service-key"})
    assert r.json()["waiting"] == 0  # nothing starts once the budget is spent


def test_a_failing_chat_model_falls_back_and_a_spent_quota_is_explained(make_client):
    class OutOfQuota(FakeLLM):
        async def chat(self, messages, **kw):  # type: ignore[override]
            raise QuotaExhaustedError("daily tokens")

    sleep_llm = FakeLLM(sleep_and_answer)
    client = accounts_client(make_client, llm=OutOfQuota(), sleep_llm=sleep_llm)
    r = say(client, ALICE)
    assert r.status_code == 200 and r.json()["trace"]["fallback"] is True
    assert r.json()["answer"] == "Answered by the fallback model."

    sleep_llm.fail = True  # both down: a clear 503, and the message is still remembered
    r = say(client, ALICE, "My exam is on Friday", minute=1)
    assert r.status_code == 502  # the fallback's own error is a provider outage, not quota
    turns = client.get("/memory/notes", params={"tier": "episode"}, headers=auth(ALICE)).json()
    assert turns["total"] == 2


def test_sleep_usage_is_counted_against_the_budget(make_client, fresh_db):
    client = accounts_client(make_client)
    say(client, ALICE)
    r = client.post(
        "/sleep", params={"wait": "true"}, json={"occurred_at": JAN_SLEEP.isoformat()},
        headers=auth(ALICE),
    )  # fmt: skip
    calls = r.json()["cycle"]["llm_calls"]
    assert calls > 0
    with psycopg.connect(fresh_db) as conn:
        row = conn.execute("select requests from llm_usage_daily where budget = 'sleep'").fetchone()
    assert row is not None and row[0] == calls


@pytest.fixture
def demo(make_client, fresh_db):
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        conn.execute("select set_config('nocturne.owner', 'demo', false)")
        conn.execute(
            "insert into memory_notes (tier, content, valid_from) values ('fact', %s, now())",
            ("The user lives in Dallas.",),
        )
    return accounts_client(make_client, demo_owner="demo")


DEMO = {"X-Nocturne-Demo": "1"}


def test_visitors_can_explore_the_demo_memory_read_only(demo):
    notes = demo.get("/memory/notes", headers=DEMO).json()
    assert [n["content"] for n in notes["items"]] == ["The user lives in Dallas."]
    me = demo.get("/me", headers=DEMO).json()
    assert me["demo"] is True and me["owner"] == "demo" and me["chats_left"] is None
    note = notes["items"][0]["id"]
    for method, path, body in [
        ("post", "/chat", {"message": "hi"}),
        ("post", "/sleep", None),
        ("post", f"/memory/notes/{note}/forget", None),
        ("post", f"/memory/notes/{note}/correct", {"content": "x"}),
        ("post", "/ingest", {"messages": [{"role": "user", "content": "x"}]}),
    ]:
        r = getattr(demo, method)(path, json=body, headers=DEMO)
        assert r.status_code == 403, (path, r.status_code)
        assert "read-only" in r.json()["detail"]
    # signed-in people don't see the demo, and the demo doesn't see them
    assert demo.get("/memory/notes", headers=auth(ALICE)).json()["total"] == 0


def test_no_demo_unless_configured(make_client):
    client = accounts_client(make_client)
    assert client.get("/memory/notes", headers=DEMO).status_code == 401
