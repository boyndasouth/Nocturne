"""Memory explorer API (frontend backing): notes, a note's history, the graph, and what a
sleep cycle changed. Scenario: Austin, sleep, move to Dallas, sleep."""

from __future__ import annotations

from tests.integration.conftest import FakeLLM
from tests.integration.test_phase2 import JAN, JAN_SLEEP, MAR, MAR_SLEEP, chat, sleep, sleep_brain


def _two_nights(make_client):
    client = make_client(sleep_llm=FakeLLM(sleep_brain), enable_auditor=False)
    chat(client, "I live in Austin", JAN)
    first = sleep(client, JAN_SLEEP)["cycle"]
    chat(client, "I moved to Dallas", MAR)
    second = sleep(client, MAR_SLEEP)["cycle"]
    return client, first, second


def test_notes_list_filters_and_detail_history(make_client, fresh_db):
    client, first, second = _two_nights(make_client)

    active = client.get("/memory/notes").json()
    assert [n["content"] for n in active["items"]] == ["The user lives in Dallas."]
    assert active["total"] == 1 and active["items"][0]["source_count"] == 1

    both = client.get("/memory/notes", params={"status": "active,superseded"}).json()
    assert both["total"] == 2
    assert [n["content"] for n in both["items"]] == [  # newest first
        "The user lives in Dallas.",
        "The user lives in Austin.",
    ]
    searched = client.get("/memory/notes", params={"status": "superseded", "q": "austin"}).json()
    assert [n["content"] for n in searched["items"]] == ["The user lives in Austin."]
    episodes = client.get(
        "/memory/notes", params={"tier": "episode", "status": "active,archived"}
    ).json()
    assert episodes["total"] == 2
    paged = client.get(
        "/memory/notes", params={"status": "active,superseded", "limit": 1, "offset": 5}
    ).json()
    assert paged["items"] == [] and paged["total"] == 2
    assert client.get("/memory/notes", params={"tier": "nope"}).status_code == 422

    austin_id = searched["items"][0]["id"]
    d = client.get(f"/memory/notes/{austin_id}").json()
    assert d["created_by_cycle"] == first["cycle_id"]
    # v1 (active, open-ended) ended by the second cycle -> v2 superseded, valid to March
    v1, v2 = d["history"]
    assert (v1["version"], v1["status"], v1["valid_to"]) == (1, "active", None)
    assert v1["ended"]["cycle_id"] == second["cycle_id"]
    assert v1["ended"]["reason"].startswith("superseded by")
    assert (v2["version"], v2["status"], v2["ended"]) == (2, "superseded", None)
    assert v2["valid_to"].startswith("2026-03-02")
    [link] = d["links"]
    assert (link["link_type"], link["direction"]) == ("supersedes", "in")
    assert link["other"]["content"] == "The user lives in Dallas."
    [source] = d["sources"]
    assert source["tier"] == "episode" and "I live in Austin" in source["content"]

    assert client.get("/memory/notes/00000000-0000-0000-0000-000000000000").status_code == 404


def test_graph_and_cycle_changes(make_client, fresh_db):
    client, first, second = _two_nights(make_client)

    g = client.get("/memory/graph").json()
    assert g["total"] == 2 and len(g["nodes"]) == 2
    [edge] = g["edges"]
    by_id = {n["id"]: n["content"] for n in g["nodes"]}
    assert by_id[edge["src_id"]] == "The user lives in Dallas."
    assert edge["link_type"] == "supersedes" and edge["created_by_cycle"] == second["cycle_id"]
    only_active = client.get("/memory/graph", params={"status": "active"}).json()
    assert len(only_active["nodes"]) == 1 and only_active["edges"] == []

    c1 = client.get(f"/sleep/cycles/{first['cycle_id']}/changes").json()
    assert [n["content"] for n in c1["created"]] == ["The user lives in Austin."]
    assert [e["tier"] for e in c1["episodes"]] == ["episode"] and c1["edits"] == []

    c2 = client.get(f"/sleep/cycles/{second['cycle_id']}/changes").json()
    assert [n["content"] for n in c2["created"]] == ["The user lives in Dallas."]
    [edit] = c2["edits"]
    assert edit["kind"] == "superseded" and edit["status_now"] == "superseded"
    assert edit["before"] == edit["after"] == "The user lives in Austin."
    [link] = c2["links"]
    assert link["link_type"] == "supersedes"
    assert (link["src_content"], link["dst_content"]) == (
        "The user lives in Dallas.",
        "The user lives in Austin.",
    )

    stats = client.get("/memory/stats").json()
    assert stats["notes"]["fact"] == {"active": 1, "superseded": 1, "archived": 0, "forgotten": 0}
    assert stats["cycles"] == 2 and stats["pending_episodes"] == 0

    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/sleep/cycles/{missing}/changes").status_code == 404


def test_cors_allows_the_configured_frontend_only(make_client, fresh_db):
    client = make_client(cors_origins="https://nocturne.example.app")
    pre = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "x-api-key"}
    ok = client.options("/memory/notes", headers={"Origin": "https://nocturne.example.app", **pre})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "https://nocturne.example.app"
    bad = client.options("/memory/notes", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in bad.headers


def test_memory_endpoints_require_the_api_key(make_client, fresh_db):
    client = make_client(api_key="s3cret")
    assert client.get("/memory/notes").status_code == 401
    assert client.get("/memory/graph", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_correcting_a_fact_is_a_versioned_edit(make_client, fresh_db):
    client, _, _ = _two_nights(make_client)
    dallas = client.get("/memory/notes", params={"q": "dallas"}).json()["items"][0]

    r = client.post(
        f"/memory/notes/{dallas['id']}/correct", json={"content": "  I live in Plano. "}
    )
    assert r.status_code == 200
    assert r.json()["content"] == "I live in Plano." and r.json()["version"] == 2

    d = client.get(f"/memory/notes/{dallas['id']}").json()
    v1, v2 = d["history"]
    assert v1["content"] == "The user lives in Dallas." and v2["content"] == "I live in Plano."
    assert v1["ended"]["reason"] == "corrected by you" and v1["ended"]["cycle_id"] is None
    # the new wording is what search finds
    assert client.get("/memory/notes", params={"q": "plano"}).json()["total"] == 1

    austin = client.get("/memory/notes", params={"status": "superseded"}).json()["items"][0]
    assert (
        client.post(f"/memory/notes/{austin['id']}/correct", json={"content": "x"}).status_code
        == 409
    )
    turn = client.get("/memory/notes", params={"tier": "episode"}).json()["items"][0]
    assert (
        client.post(f"/memory/notes/{turn['id']}/correct", json={"content": "x"}).status_code == 409
    )
    assert (
        client.post(f"/memory/notes/{dallas['id']}/correct", json={"content": ""}).status_code
        == 422
    )


def test_forgetting_a_fact_forgets_its_turns_and_chat_stops_using_them(make_client, fresh_db):
    client, _, _ = _two_nights(make_client)
    dallas = client.get("/memory/notes", params={"q": "dallas"}).json()["items"][0]

    r = client.post(f"/memory/notes/{dallas['id']}/forget").json()
    assert r["turns_forgotten"] == 1 and r["themes"] == []
    assert client.get("/memory/notes", params={"q": "dallas"}).json()["total"] == 0
    forgotten = client.get(
        "/memory/notes", params={"tier": "fact,episode", "status": "forgotten", "q": "dallas"}
    ).json()
    assert forgotten["total"] == 2  # the fact and the turn it came from
    assert client.post(f"/memory/notes/{dallas['id']}/forget").status_code == 409

    answer = chat(client, "Where do I live? Dallas?", MAR.replace(day=20))
    used = {m["id"] for m in answer["retrieved"]}
    assert dallas["id"] not in used
    assert not any("Dallas" in m["preview"] for m in answer["retrieved"])


def test_a_turn_forgotten_before_sleep_is_never_learned(make_client, fresh_db):
    client = make_client(sleep_llm=FakeLLM(sleep_brain), enable_auditor=False)
    chat(client, "I live in Austin", JAN)
    turn = client.get("/memory/notes", params={"tier": "episode"}).json()["items"][0]
    assert client.get("/memory/stats").json()["pending_episodes"] == 1

    client.post(f"/memory/notes/{turn['id']}/forget")

    assert client.get("/memory/stats").json()["pending_episodes"] == 0
    assert sleep(client, JAN_SLEEP)["status"] == "nothing_to_do"
    assert client.get("/memory/notes").json()["total"] == 0


def test_chat_returns_an_answer_trace(make_client, fresh_db):
    client, _, _ = _two_nights(make_client)
    r = chat(client, "Where do I live?", MAR.replace(day=20))
    t = r["trace"]
    assert t["model"] == "fake-model" and t["input_tokens"] >= 0
    [system, user] = t["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    assert "Where do I live?" in user["content"]
    dallas = next(m for m in t["memories"] if m["content"] == "The user lives in Dallas.")
    assert dallas["via"] == "search" and dallas["status"] == "active"
    assert dallas["ref"] and f"[{dallas['ref']}]" in user["content"]  # it was in the prompt
    assert dallas["score"] > 0 and 0 <= dallas["semantic"] <= 1
    assert [m["id"] for m in t["memories"]] == [m["id"] for m in r["retrieved"]]
