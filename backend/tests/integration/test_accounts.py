"""Accounts (AUTH_MODE=supabase): each Supabase user has their own memory, enforced by
Postgres row-level security, and the cron's service key sleeps every account.

Tokens are real ES256 JWTs signed with a key generated here and served to the verifier as
its JWKS, so the production verification path runs end to end."""

from __future__ import annotations

import time

import jwt
import psycopg
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from app.agents.sleep.trigger import IdleTrigger, TriggerPolicy
from app.auth import SupabaseVerifier
from tests.integration.conftest import FakeLLM
from tests.integration.test_phase2 import JAN, JAN_SLEEP, sleep_brain

URL = "https://nocturne-test.supabase.co"
ISSUER = URL + "/auth/v1"
KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())
JWKS = {
    "keys": [
        {**jwt.algorithms.ECAlgorithm.to_jwk(KEY.public_key(), as_dict=True), "kid": "k1",
         "alg": "ES256", "use": "sig"}
    ]
}  # fmt: skip
ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


def token(sub: str, *, key=KEY, kid: str = "k1", **claims: object) -> str:
    now = int(time.time())
    body = {"sub": sub, "aud": "authenticated", "iss": ISSUER, "role": "authenticated",
            "email": f"{sub[:4]}@example.com", "iat": now, "exp": now + 3600, **claims}  # fmt: skip
    return jwt.encode(body, key, algorithm="ES256", headers={"kid": kid})


def auth(sub: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(sub)}"}


@pytest.fixture
def accounts(make_client):
    return make_client(
        sleep_llm=FakeLLM(sleep_brain),
        enable_auditor=False,
        auth_mode="supabase",
        supabase_url=URL,
        api_key="service-key",
        verifier=SupabaseVerifier(URL, jwks=JWKS),
    )


def say(client, who: str, message: str, minute: int = 0):
    r = client.post(
        "/chat",
        json={"message": message, "occurred_at": JAN.replace(minute=minute).isoformat()},
        headers=auth(who),
    )
    assert r.status_code == 200, r.text
    return r.json()


def sleep_now(client, who: str):
    r = client.post(
        "/sleep",
        params={"wait": "true"},
        json={"occurred_at": JAN_SLEEP.isoformat()},
        headers=auth(who),
    )
    assert r.status_code == 200, r.text
    return r.json()


def facts(client, who: str) -> list[str]:
    r = client.get("/memory/notes", headers=auth(who))
    assert r.status_code == 200, r.text
    return [n["content"] for n in r.json()["items"]]


def test_requests_need_a_valid_supabase_token(accounts):
    assert accounts.get("/health").status_code == 200
    for path in ("/memory/notes", "/memory/stats", "/sleep/cycles", "/me"):
        assert accounts.get(path).status_code == 401, path
    assert accounts.post("/chat", json={"message": "hi"}).status_code == 401

    def get(tok: str) -> int:
        return accounts.get("/memory/notes", headers={"Authorization": f"Bearer {tok}"}).status_code

    assert get(token(ALICE)) == 200
    assert get(token(ALICE, key=OTHER_KEY)) == 401  # signed by someone else
    assert get(token(ALICE, kid="nope")) == 401  # unknown key
    assert get(token(ALICE, exp=int(time.time()) - 3600)) == 401  # expired
    assert get(token(ALICE, aud="anon-app")) == 401
    assert get(token(ALICE, iss="https://evil.supabase.co/auth/v1")) == 401
    assert get(token(ALICE, role="anon")) == 401
    assert get("not-a-jwt") == 401
    # the service key isn't a person
    assert accounts.get("/memory/notes", headers={"X-API-Key": "service-key"}).status_code == 403

    me = accounts.get("/me", headers=auth(ALICE)).json()
    assert me == {"auth_mode": "supabase", "owner": ALICE, "email": "1111@example.com"}


def test_each_account_sees_only_its_own_memory(accounts):
    say(accounts, ALICE, "I live in Austin")
    a = sleep_now(accounts, ALICE)
    assert a["status"] == "succeeded"
    assert facts(accounts, ALICE) == ["The user lives in Austin."]

    # Bob sees nothing of Alice's: notes, turns, cycles, the graph, stats
    assert facts(accounts, BOB) == []
    bob = auth(BOB)
    assert (
        accounts.get("/memory/notes", params={"tier": "episode"}, headers=bob).json()["total"] == 0
    )
    assert accounts.get("/sleep/cycles", headers=bob).json() == []
    assert accounts.get("/memory/graph", headers=bob).json()["nodes"] == []
    assert accounts.get("/memory/stats", headers=bob).json()["notes"]["fact"]["active"] == 0
    # ...even by id
    austin = accounts.get("/memory/notes", headers=auth(ALICE)).json()["items"][0]["id"]
    cycle = a["cycle"]["cycle_id"]
    assert accounts.get(f"/memory/notes/{austin}", headers=bob).status_code == 404
    assert accounts.get(f"/sleep/cycles/{cycle}", headers=bob).status_code == 404
    assert accounts.get(f"/sleep/cycles/{cycle}/changes", headers=bob).status_code == 404
    assert accounts.post(f"/memory/notes/{austin}/forget", headers=bob).status_code == 404
    r = accounts.post(f"/memory/notes/{austin}/correct", json={"content": "x"}, headers=bob)
    assert r.status_code == 404
    # ...and Bob's assistant can't recall it
    answer = say(accounts, BOB, "Where do I live? Austin?", minute=5)
    assert answer["retrieved"] == [] or all(
        "Austin" not in m["preview"] for m in answer["retrieved"]
    )
    # Bob's sleep only reads Bob's turn
    b = sleep_now(accounts, BOB)
    assert b["cycle"]["episodes_processed"] == 1
    assert facts(accounts, ALICE) == ["The user lives in Austin."]


def test_service_key_sleeps_every_account(accounts):
    say(accounts, ALICE, "I live in Austin")
    say(accounts, BOB, "My exam is on Friday")
    assert accounts.post("/sleep/all", headers=auth(ALICE)).status_code == 403
    assert accounts.post("/sleep/all").status_code == 401

    r = accounts.post("/sleep/all", params={"wait": "true"}, headers={"X-API-Key": "service-key"})
    assert r.status_code == 200, r.text
    assert r.json() == {"waiting": 2, "started": 2, "busy": 0, "failed": 0}
    assert facts(accounts, ALICE) == ["The user lives in Austin."]
    assert facts(accounts, BOB) == ["The user's linear algebra exam is on Friday."]
    again = accounts.post("/sleep/all", headers={"X-API-Key": "service-key"})
    assert again.status_code == 200  # nothing waiting: nothing started
    assert again.json() == {"waiting": 0, "started": 0, "busy": 0, "failed": 0}


async def test_idle_trigger_sleeps_each_account_separately(accounts):
    say(accounts, ALICE, "I live in Austin")
    say(accounts, BOB, "My exam is on Friday")
    c = accounts.app.state.container
    policy = TriggerPolicy(idle_seconds=0, idle_min_episodes=1, backlog_episodes=99)
    ran = await IdleTrigger(c.sleep_runner, c.store, c.activity, policy).tick()
    assert ran
    assert facts(accounts, ALICE) == ["The user lives in Austin."]
    assert facts(accounts, BOB) == ["The user's linear algebra exam is on Friday."]


def test_the_database_itself_enforces_it(accounts, fresh_db):
    say(accounts, ALICE, "I live in Austin")
    say(accounts, BOB, "My exam is on Friday")
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        conn.execute("set role nocturne_app")
        # no owner: nothing visible (fails closed)
        assert conn.execute("select count(*) from memory_notes").fetchone()[0] == 0
        conn.execute("select set_config('nocturne.owner', %s, false)", (ALICE,))
        rows = conn.execute("select owner_id, content from memory_notes").fetchall()
        assert [r[0] for r in rows] == [ALICE] and "Austin" in rows[0][1]
        # can't write as someone else, or move a row to them
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                "insert into memory_notes (tier, content, valid_from, owner_id)"
                " values ('episode', 'x', now(), %s)",
                (BOB,),
            )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("update memory_notes set owner_id = %s", (BOB,))
        # and the guard on deletes still applies
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("delete from memory_notes")


def test_accounts_off_is_one_local_memory(make_client):
    client = make_client(sleep_llm=FakeLLM(sleep_brain), enable_auditor=False)
    r = client.post("/chat", json={"message": "I live in Austin", "occurred_at": JAN.isoformat()})
    assert r.status_code == 200
    assert client.get("/me").json() == {"auth_mode": "none", "owner": "local", "email": None}
    r = client.post("/sleep/all", params={"wait": "true"})  # local is the only account
    assert r.status_code == 200 and r.json()["started"] == 1
    assert [n["content"] for n in client.get("/memory/notes").json()["items"]] == [
        "The user lives in Austin."
    ]


def test_accounts_off_never_shows_other_accounts(make_client, fresh_db):
    """A database that once had accounts on: with accounts off you see only 'local'."""
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        conn.execute("select set_config('nocturne.owner', %s, false)", (ALICE,))
        conn.execute(
            "insert into memory_notes (tier, content, valid_from) values ('fact', %s, now())",
            ("The user lives in Austin.",),
        )
    client = make_client(enable_auditor=False)
    assert client.get("/memory/notes").json()["total"] == 0
    assert client.get("/memory/stats").json()["notes"]["fact"]["active"] == 0
