"""Phase 3 step 2 against real Postgres: the Linker and Abstractor inside a sleep cycle.

Hash embeddings put the "The user's <kind> bike is a <brand>" facts at ~0.75 cosine of each
other and well below that of everything else, so the default thresholds link and cluster
exactly the bike facts.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

from app.llm.prompts import (
    ABSTRACTOR_SYSTEM,
    CONTRADICTION_SYSTEM,
    DEDUP_SYSTEM,
    EXTRACTOR_SYSTEMS,
    LINKER_SYSTEM,
)
from tests.integration.conftest import FakeLLM
from tests.integration.test_phase2 import chat, db, sleep

FACTS = {
    "My road bike is a Specialized": "The user's road bike is a Specialized.",
    "My commuter bike is a Trek": "The user's commuter bike is a Trek.",
    "My mountain bike is a Giant": "The user's mountain bike is a Giant.",
    "My gravel bike is a Canyon": "The user's gravel bike is a Canyon.",
    "My exam is on Friday": "The user's linear algebra exam is on Friday.",
}

T = {
    "jan": datetime(2026, 1, 10, 9, tzinfo=UTC),
    "jan_sleep": datetime(2026, 1, 11, 3, tzinfo=UTC),
    "mar": datetime(2026, 3, 2, 18, tzinfo=UTC),
    "mar_sleep": datetime(2026, 3, 3, 3, tzinfo=UTC),
    "apr": datetime(2026, 4, 6, 18, tzinfo=UTC),
    "apr_sleep": datetime(2026, 4, 7, 3, tzinfo=UTC),
}


def brain(messages) -> str:
    system, user = messages[0]["content"], messages[-1]["content"]
    if system in EXTRACTOR_SYSTEMS:
        out = []
        for n, body in re.findall(
            r"^\[(\d+)\] \([^)]*\)\n(.*?)(?=^\[\d+\] |\Z)", user, re.M | re.S
        ):
            for phrase, statement in FACTS.items():
                if f"User: {phrase}" in body:
                    out.append({"statement": statement, "keywords": [], "sources": [int(n)]})
        return json.dumps({"facts": out})
    groups = re.findall(r"Group (\d+)\nNEW: (.*?)\nEXISTING:\n", user)
    if system == DEDUP_SYSTEM:
        return json.dumps(
            {"decisions": [{"group": int(g), "duplicate_of": None} for g, _ in groups]}
        )
    if system == CONTRADICTION_SYSTEM:
        return json.dumps(
            {"decisions": [{"group": int(g), "conflicts_with": []} for g, _ in groups]}
        )
    if system == LINKER_SYSTEM:
        # Link every new fact to its closest neighbour. The Giant fact is "elaborated_by",
        # which the Linker stores reversed (neighbour -> new).
        return json.dumps(
            {
                "decisions": [
                    {
                        "group": int(g),
                        "links": [
                            {"to": "A", "type": "elaborated_by" if "Giant" in new else "related"}
                        ],
                    }
                    for g, new in groups
                ]
            }
        )
    assert system == ABSTRACTOR_SYSTEM
    clusters = re.findall(r"Cluster (\d+)\n((?:- .*\n?)+)", user)
    return json.dumps(
        {
            "themes": [
                {
                    "cluster": int(c),
                    "title": "User's bikes",
                    "summary": f"The user owns {len(body.strip().splitlines())} bikes.",
                    "keywords": ["bikes", "cycling"],
                }
                for c, body in clusters
            ]
        }
    )


def themes(url: str):
    return db(
        url,
        "select id, content, keywords, status::text, version, valid_from, recorded_at"
        " from memory_notes where tier = 'theme' order by recorded_at",
    )


def members(url: str, theme_id) -> list[str]:
    rows = db(
        url,
        "select n.content from memory_links l join memory_notes n on n.id = l.dst_id"
        " where l.src_id = %s and l.link_type = 'summarizes' order by n.content",
        theme_id,
    )
    return [r["content"] for r in rows]


def fact_links(url: str):
    return db(
        url,
        "select s.content as src, d.content as dst, l.link_type::text as type"
        " from memory_links l join memory_notes s on s.id = l.src_id"
        " join memory_notes d on d.id = l.dst_id"
        " where l.link_type not in ('summarizes', 'supersedes') order by s.content, d.content",
    )


def test_links_then_a_theme_then_a_theme_update(make_client, fresh_db):
    sleep_llm = FakeLLM(brain)
    client = make_client(sleep_llm=sleep_llm)

    chat(client, "My road bike is a Specialized", T["jan"])
    chat(client, "My commuter bike is a Trek", T["jan"] + timedelta(minutes=1))
    chat(client, "My exam is on Friday", T["jan"] + timedelta(minutes=2))
    jan = sleep(client, T["jan_sleep"])["cycle"]
    assert (jan["facts_created"], jan["links_created"], jan["themes_created"]) == (3, 1, 0)
    assert fact_links(fresh_db) == [
        {
            "src": "The user's commuter bike is a Trek.",
            "dst": "The user's road bike is a Specialized.",
            "type": "related",
        }
    ]
    assert themes(fresh_db) == []  # two bike facts are not a cluster yet

    chat(client, "My mountain bike is a Giant", T["mar"])
    mar = sleep(client, T["mar_sleep"])["cycle"]
    assert (mar["links_created"], mar["themes_created"], mar["themes_updated"]) == (1, 1, 0)
    giant_link = [lk for lk in fact_links(fresh_db) if "Giant" in lk["dst"]]
    assert giant_link and giant_link[0]["type"] == "elaborates"  # reversed: neighbour -> new
    [theme] = themes(fresh_db)
    assert theme["content"] == "User's bikes: The user owns 3 bikes."
    assert theme["keywords"] == ["bikes", "cycling"]
    assert theme["valid_from"] == T["jan"]  # earliest member
    assert theme["recorded_at"] == T["mar_sleep"]
    assert members(fresh_db, theme["id"]) == [
        "The user's commuter bike is a Trek.",
        "The user's mountain bike is a Giant.",
        "The user's road bike is a Specialized.",
    ]

    calls_before = len(sleep_llm.trace_names)
    chat(client, "My gravel bike is a Canyon", T["apr"])
    apr = sleep(client, T["apr_sleep"])["cycle"]
    assert (apr["themes_created"], apr["themes_updated"]) == (0, 1)
    [theme2] = themes(fresh_db)
    assert theme2["id"] == theme["id"] and theme2["version"] == 2
    assert theme2["content"] == "User's bikes: The user owns 4 bikes."
    assert len(members(fresh_db, theme2["id"])) == 4
    [version] = db(fresh_db, "select * from memory_versions where note_id = %s", theme["id"])
    assert version["content"] == theme["content"] and "re-summarised" in version["reason"]
    assert sleep_llm.trace_names[calls_before:].count("sleep.abstractor") == 1


def test_unchanged_cluster_costs_no_abstractor_call(make_client, fresh_db):
    sleep_llm = FakeLLM(brain)
    client = make_client(sleep_llm=sleep_llm)
    for i, phrase in enumerate(list(FACTS)[:3]):
        chat(client, phrase, T["jan"] + timedelta(minutes=i))
    sleep(client, T["jan_sleep"])
    assert sleep_llm.trace_names.count("sleep.abstractor") == 1

    chat(client, "My exam is on Friday", T["mar"])  # joins no cluster
    cycle = sleep(client, T["mar_sleep"])["cycle"]
    assert (cycle["themes_created"], cycle["themes_updated"]) == (0, 0)
    assert sleep_llm.trace_names.count("sleep.abstractor") == 1


def test_linker_and_abstractor_can_be_switched_off(make_client, fresh_db):
    sleep_llm = FakeLLM(brain)
    client = make_client(sleep_llm=sleep_llm, enable_linker=False, enable_abstractor=False)
    for i, phrase in enumerate(list(FACTS)[:3]):
        chat(client, phrase, T["jan"] + timedelta(minutes=i))
    cycle = sleep(client, T["jan_sleep"])["cycle"]
    assert cycle["facts_created"] == 3
    assert (cycle["links_created"], cycle["themes_created"]) == (0, 0)
    assert "sleep.linker" not in sleep_llm.trace_names
    assert "sleep.abstractor" not in sleep_llm.trace_names


def test_failed_abstractor_call_resumes_without_redoing_earlier_steps(make_client, fresh_db):
    sleep_llm = FakeLLM(brain)
    sleep_llm.fail_on = ["sleep.abstractor"]
    client = make_client(sleep_llm=sleep_llm)
    for i, phrase in enumerate(list(FACTS)[:3]):
        chat(client, phrase, T["jan"] + timedelta(minutes=i))
    first = sleep(client, T["jan_sleep"])
    assert first["status"] == "failed"
    assert (
        db(fresh_db, "select count(*) as n from memory_notes where tier <> 'episode'")[0]["n"] == 0
    )

    second = sleep(client, T["jan_sleep"])
    assert second["status"] == "succeeded" and second["resumed"]
    assert second["cycle"]["themes_created"] == 1 and second["cycle"]["links_created"] == 2
    assert sleep_llm.trace_names.count("sleep.extract") == 1
    assert sleep_llm.trace_names.count("sleep.linker") == 1
    assert sleep_llm.trace_names.count("sleep.abstractor") == 2  # failed once, then ok


def test_cycle_history_reports_the_new_counters(make_client, fresh_db):
    client = make_client(sleep_llm=FakeLLM(brain))
    for i, phrase in enumerate(list(FACTS)[:3]):
        chat(client, phrase, T["jan"] + timedelta(minutes=i))
    sleep(client, T["jan_sleep"])
    [cycle] = client.get("/sleep/cycles").json()
    assert cycle["links_created"] == 2 and cycle["themes_created"] == 1
