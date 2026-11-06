"""Phase 3 step 3 against real Postgres: the hierarchical retriever (spec §5).

Memory is written directly (themes, links, superseded and archived notes) so each test
controls exactly what exists. Hash embeddings score by shared words, which makes "shares
no words with the question" a precise way to say "direct search can't find this".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.memory.models import LinkType, NewNote, Status, Tier
from tests.integration.conftest import FakeLLM

NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)
JAN = datetime(2026, 1, 1, tzinfo=UTC)
APR = datetime(2026, 4, 1, tzinfo=UTC)


@pytest.fixture
def mem(make_client):
    """A tiny DSL over the app's own store and retriever, run on the app's event loop."""

    class Mem:
        def __init__(self, **overrides):
            self.client = make_client(llm=FakeLLM(), **overrides)
            self.c = self.client.app.state.container

        def run(self, fn):
            return self.client.portal.call(fn)

        def add(self, content, tier=Tier.FACT, *, at=JAN, status=Status.ACTIVE, valid_to=None,
                recorded_at=None):  # fmt: skip
            [vec] = self.c.embedder.embed_documents([content])
            note = NewNote(tier=tier, content=content, embedding=vec, valid_from=at,
                           recorded_at=recorded_at or at, status=status,
                           valid_to=valid_to)  # fmt: skip
            [out] = self.run(lambda: self.c.store.insert_notes([note]))
            return out.id

        def link(self, src, dst, link_type):
            async def go():
                async with self.c.store.transaction() as conn:
                    await self.c.store.add_link(conn, src, dst, link_type, cycle_id=None)

            self.run(go)

        def retrieve(self, query, **kw):
            kw.setdefault("now", NOW)
            return self.run(lambda: self.c.retriever.retrieve(query, **kw))

    return Mem


def by_content(results):
    return {s.note.content: s for s in results}


def test_theme_members_are_reached_even_when_search_misses_them(mem):
    m = mem(retrieval_candidate_pool=1, retrieval_top_k=3)
    theme = m.add("Cycling gear: the user rides bikes on weekends.", Tier.THEME)
    helmet = m.add("The user bought a new helmet in May.")  # no words shared with the query
    m.link(theme, helmet, LinkType.SUMMARIZES)
    for i in range(6):  # direct matches that fill the search pools
        m.add(f"The user packed camping gear number {i}.")

    out = m.retrieve("What cycling gear do I have?")
    assert out[0].note.id == theme and out[0].note.tier is Tier.THEME  # themes first
    got = by_content(out)
    assert got["The user bought a new helmet in May."].via == "theme"
    assert got["The user bought a new helmet in May."].score == pytest.approx(out[0].score * 0.85)


def test_one_link_hop_in_either_direction(mem):
    m = mem()
    exam = m.add("The user's exam is on Friday.")
    room = m.add("The user booked a quiet library room.")
    tutor = m.add("Maria tutors the user every Tuesday.")
    m.link(room, exam, LinkType.CAUSED_BY)  # room -> exam: found from the exam side
    m.link(exam, tutor, LinkType.RELATED)
    m.add("The user's favourite colour is green.")

    got = by_content(m.retrieve("When is my exam?"))
    assert got["The user booked a quiet library room."].via == "link"
    assert got["Maria tutors the user every Tuesday."].via == "link"
    assert got["The user's exam is on Friday."].via == "search"


def test_expansion_respects_as_of(mem):
    m = mem()
    exam = m.add("The user's exam is on Friday.", at=JAN)
    later = m.add("The user booked a quiet library room.", at=APR)
    m.link(later, exam, LinkType.CAUSED_BY)
    got = by_content(m.retrieve("When is my exam?", as_of=JAN + timedelta(days=1)))
    assert "The user booked a quiet library room." not in got


def test_time_phrase_brings_back_what_was_true_then(mem):
    m = mem()
    m.add("The user lives in Dallas.", at=APR)
    m.add("The user lives in Austin.", at=JAN, status=Status.SUPERSEDED, valid_to=APR)

    now_q = m.retrieve("Where do I live?")
    assert [s.note.content for s in now_q] == ["The user lives in Dallas."]

    then = m.retrieve("Where did I live in February?")
    assert [s.note.content for s in then][0] == "The user lives in Austin."  # boosted
    assert m.c.retriever.last_window.phrase == "in February"
    assert {s.note.content for s in then} == {
        "The user lives in Austin.",
        "The user lives in Dallas.",
    }


def test_archived_turns_are_a_fallback_only(mem):
    m = mem()
    m.add("The user's favourite colour is green.")
    m.add(
        "User: Can you recommend a book?\nAssistant: Try Project Hail Mary.",
        Tier.EPISODE,
        status=Status.ARCHIVED,
    )
    strong = m.retrieve("What is my favourite colour?")
    assert all(s.via != "fallback" for s in strong)  # good fact match: no fallback

    convo = by_content(m.retrieve("Which book did you recommend?"))
    hit = convo["User: Can you recommend a book?\nAssistant: Try Project Hail Mary."]
    assert hit.via == "fallback"


def test_without_themes_or_links_it_is_plain_search(mem):
    m = mem()
    m.add("The user's exam is on Friday.")
    m.add("User: my exam is on Friday", Tier.EPISODE)
    out = m.retrieve("When is my exam?")
    assert {s.via for s in out} == {"search"}
    assert {s.note.tier for s in out} == {Tier.FACT, Tier.EPISODE}


def test_theme_and_tier_options(mem):
    m = mem(retrieval_theme_k=0)
    m.add("Cycling gear: the user rides bikes on weekends.", Tier.THEME)
    m.add("The user rides a Trek bike.")
    assert all(s.note.tier is not Tier.THEME for s in m.retrieve("cycling bikes"))
    out = m.retrieve("cycling bikes", tiers=(Tier.FACT,))
    assert {s.note.tier for s in out} == {Tier.FACT}


def test_chat_shows_themes_and_superseded_periods_to_the_model(mem):
    seen = []

    def reply(messages):
        seen.append(messages[-1]["content"])
        return "ok"

    m = mem()
    m.client.app.state.container.responder._llm = FakeLLM(reply)
    m.add("Home: the user moved from Austin to Dallas.", Tier.THEME, at=JAN)
    m.add("The user lives in Austin.", at=JAN, status=Status.SUPERSEDED, valid_to=APR)
    m.client.post(
        "/chat", json={"message": "Where did I live in February?", "occurred_at": NOW.isoformat()}
    ).raise_for_status()
    [prompt] = seen
    assert "THEME (since 2026-01-01) Home: the user moved from Austin to Dallas." in prompt
    assert "FACT (true 2026-01-01 to 2026-04-01) The user lives in Austin." in prompt
