"""Extractor validation, structured-output parsing, the idle-trigger policy, and the
responder's memory context with supersession history."""

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.agents.responder import build_memory_context
from app.agents.sleep.extractor import ExtractionOutput, batch_episodes, to_candidates
from app.agents.sleep.state import EpisodeIn
from app.agents.sleep.trigger import ActivityClock, TriggerPolicy, should_sleep
from app.llm.client import LLMError, LLMResponse
from app.llm.structured import chat_json, extract_json
from app.memory.models import MemoryNote, ScoredNote, Status, Tier

T0 = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)


def ep(content: str = "User: hi", when: datetime = T0) -> EpisodeIn:
    return EpisodeIn(id=uuid4(), content=content, valid_from=when)


# --- extractor ------------------------------------------------------------------------------


def test_batches_respect_char_and_count_limits():
    eps = [ep("x" * 400) for _ in range(5)]
    assert [len(b) for b in batch_episodes(eps, max_chars=1000, max_items=10)] == [2, 2, 1]
    assert [len(b) for b in batch_episodes(eps, max_chars=10**6, max_items=2)] == [2, 2, 1]
    assert batch_episodes([], max_chars=10, max_items=1) == []
    # an oversized episode still gets a batch of its own
    assert [len(b) for b in batch_episodes([ep("y" * 9000)], max_chars=100, max_items=5)] == [1]


def test_to_candidates_validates_provenance_dates_and_duplicates():
    e1, e2 = ep(when=T0), ep(when=T0 + timedelta(days=1))
    out = ExtractionOutput.model_validate(
        {
            "facts": [
                {"statement": "The user lives in Dallas.", "keywords": ["Dallas", " home "],
                 "sources": [1, 2, 2], "valid_from": "2026-03-01"},
                {"statement": "No provenance.", "sources": [9]},
                {"statement": "the user lives in dallas.", "sources": [1]},  # dup, any case
                {"statement": "Future-dated.", "sources": 1, "valid_from": "2027-01-01"},
                {"statement": "Bad date.", "sources": [2], "valid_from": "next week"},
                {"statement": "   ", "sources": [1]},
            ]
        }
    )  # fmt: skip
    c = to_candidates(out, [e1, e2], batch_index=3)
    assert [x.content for x in c] == ["The user lives in Dallas.", "Future-dated.", "Bad date."]
    assert c[0].key == "3.0" and c[0].keywords == ["dallas", "home"]
    assert c[0].source_episode_ids == [e1.id, e2.id]
    assert c[0].valid_from == datetime(2026, 3, 1, tzinfo=UTC)  # explicit, in the past: kept
    assert c[1].valid_from == e1.valid_from  # explicit but after the source: ignored
    assert c[2].valid_from == e2.valid_from  # unparseable: falls back to the source


# --- structured output ----------------------------------------------------------------------


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"facts": []}\n```') == {"facts": []}
    assert extract_json('Sure! Here you go: {"facts": [1]} Hope that helps.') == {"facts": [1]}
    with pytest.raises(ValueError):
        extract_json("no json here")


class ScriptedLLM:
    name, model = "s", "s"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.seen: list[list[dict]] = []

    async def chat(self, messages, **_):
        self.seen.append(list(messages))
        return LLMResponse(self.replies.pop(0), "s", 3, 2, 1.0)

    async def aclose(self):
        pass


async def test_chat_json_repairs_once_with_the_error():
    llm = ScriptedLLM(['{"facts": "oops"}', '{"facts": []}'])
    out, usage = await chat_json(
        llm, [{"role": "user", "content": "go"}], ExtractionOutput, trace_name="t", max_tokens=9
    )
    assert out.facts == [] and (usage.tokens, usage.calls) == (10, 2)
    assert "valid JSON" in llm.seen[1][-1]["content"]


async def test_chat_json_gives_up_after_repair():
    llm = ScriptedLLM(["nope", "still nope"])
    with pytest.raises(LLMError, match="invalid JSON"):
        await chat_json(llm, [], ExtractionOutput, trace_name="t", max_tokens=9)


# --- trigger policy -------------------------------------------------------------------------

P = TriggerPolicy(idle_seconds=600, idle_min_episodes=3, backlog_episodes=30)


@pytest.mark.parametrize(
    ("pending", "idle", "expected"),
    [
        (0, 10_000, False),  # nothing to do
        (2, 10_000, False),  # idle, but too little to be worth the LLM calls
        (3, 599, False),  # enough, not idle yet
        (3, 600, True),  # (a) idle
        (30, 0, True),  # (b) backlog, even mid-conversation
    ],
)
def test_should_sleep(pending, idle, expected):
    assert should_sleep(pending, idle, P) is expected


def test_activity_clock():
    now = [100.0]
    clock = ActivityClock(lambda: now[0])
    now[0] = 160.0
    assert clock.idle_seconds() == 60.0
    clock.touch()
    assert clock.idle_seconds() == 0.0


# --- responder context ----------------------------------------------------------------------


def note(tier: Tier, content: str, start: datetime, end: datetime | None = None) -> MemoryNote:
    return MemoryNote(
        id=uuid4(), tier=tier, content=content, strength=1.0, access_count=0,
        last_accessed=None, valid_from=start, valid_to=end, recorded_at=start,
        status=Status.SUPERSEDED if end else Status.ACTIVE, version=1,
    )  # fmt: skip


def test_memory_context_shows_fact_validity_and_superseded_history():
    jan = datetime(2026, 1, 10, tzinfo=UTC)
    dallas = note(Tier.FACT, "The user lives in Dallas.", T0)
    austin = note(Tier.FACT, "The user lives in Austin.", jan, T0)
    turn = note(Tier.EPISODE, "User: I moved to Dallas", T0)
    scored = [ScoredNote(note=n, score=0.5, semantic_sim=0.5, keyword_score=0, recency=1)
              for n in (dallas, turn)]  # fmt: skip
    text, refmap = build_memory_context(scored, [], {dallas.id: [austin]})
    assert refmap == {1: dallas.id, 2: austin.id, 3: turn.id}
    lines = text.splitlines()
    assert lines[1] == "[M1] FACT (true since 2026-03-02) The user lives in Dallas."
    assert lines[2] == "    [M2] previously (2026-01-10 to 2026-03-02) The user lives in Austin."
    assert lines[3].startswith("[M3] (2026-03-02 14:00 UTC) User:")


def test_note_timestamps_are_normalised_to_utc():
    chicago = timezone(timedelta(hours=-6))
    n = note(Tier.FACT, "x", datetime(2026, 3, 1, 21, 0, tzinfo=chicago))
    assert n.valid_from == datetime(2026, 3, 2, 3, 0, tzinfo=UTC)
    assert n.valid_from.utcoffset() == timedelta(0)
    text, _ = build_memory_context(
        [ScoredNote(note=n, score=1, semantic_sim=1, keyword_score=0, recency=1)], []
    )
    assert "true since 2026-03-02" in text  # not the session-local 2026-03-01
