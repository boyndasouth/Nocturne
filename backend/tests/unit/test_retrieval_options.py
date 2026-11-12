"""Phase 4 retrieval options: splitting comparison questions, keeping each side's best
notes, and extra facts for counting questions."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.memory.models import MemoryNote, ScoredNote, Status, Tier
from app.memory.retriever import add_extra_facts, keep_per_side, split_query

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("question", "sides"),
    [
        (
            "Which device did I set up first, the smart thermostat or the mesh network system?",
            ["the smart thermostat", "the mesh network system"],
        ),
        ("Did I adopt the cat or the dog earlier?", ["Did I adopt the cat", "the dog earlier"]),
        (
            "What's the difference between my old laptop and the new tablet?",
            ["my old laptop", "the new tablet"],
        ),
        ("Python vs Rust: which did I learn first?", ["Python", "Rust"]),
        (
            "How many days did I spend in total in Lisbon and in Porto?",
            ["How many days did I spend in total in Lisbon", "in Porto"],
        ),
    ],
)
def test_comparisons_are_split(question, sides):
    assert split_query(question) == sides


@pytest.mark.parametrize(
    "question",
    [
        "How many doctors did I visit?",
        "Is my mom using the same grocery list method as me?",
        "What did I eat and drink at the party?",  # "and" without a comparison word
        "Do you remember whether I renewed it or not?",
        "Can you suggest some activities for my commute?",
        "Should I pick this one or that one?",  # sides too vague to search
        "",
    ],
)
def test_other_questions_are_not_split(question):
    assert split_query(question) == []


def _s(tier: Tier, score: float, name: str = "") -> ScoredNote:
    note = MemoryNote(
        id=uuid4(), tier=tier, content=name or f"{tier.value} {score}", keywords=[],
        strength=1.0, access_count=0, last_accessed=None,
        valid_from=T0 + timedelta(minutes=score * 100), valid_to=None, recorded_at=T0,
        status=Status.ACTIVE, source_episode_ids=[], version=1,
    )  # fmt: skip
    return ScoredNote(note=note, score=score, semantic_sim=score, keyword_score=0, recency=1)


def test_each_side_keeps_its_best_notes_replacing_the_weakest_turns():
    top = [_s(Tier.FACT, 0.9, "mesh 1"), _s(Tier.FACT, 0.85, "mesh 2"),
           _s(Tier.EPISODE, 0.8, "mesh turn"), _s(Tier.EPISODE, 0.7, "other turn")]  # fmt: skip
    thermostat = [_s(Tier.FACT, 0.6, "thermostat"), _s(Tier.EPISODE, 0.5, "thermostat turn")]
    mesh = [top[0], top[1]]
    out = keep_per_side(top, [thermostat, mesh], slots=2)
    assert [s.note.content for s in out] == [
        "mesh 1", "mesh 2", "thermostat", "thermostat turn"
    ]  # fmt: skip
    assert [s.via for s in out if "thermostat" in s.note.content] == ["split", "split"]
    assert len(out) == len(top)


def test_sides_already_in_the_top_change_nothing():
    top = [_s(Tier.FACT, 0.9), _s(Tier.FACT, 0.8), _s(Tier.EPISODE, 0.7)]
    assert keep_per_side(top, [[top[0]], [top[1]]], slots=2) == top


def test_a_side_never_evicts_another_sides_note():
    a, b = _s(Tier.FACT, 0.9, "a"), _s(Tier.FACT, 0.5, "b")
    out = keep_per_side([a], [[a], [b]], slots=1)  # no room: b is added, a stays
    assert [s.note.content for s in out] == ["a", "b"]


def test_extra_facts_follow_the_top_and_respect_the_ratio():
    top = [_s(Tier.EPISODE, 0.9), _s(Tier.FACT, 0.8)]
    ranked = [*top, _s(Tier.FACT, 0.7), _s(Tier.EPISODE, 0.65), _s(Tier.FACT, 0.6),
              _s(Tier.FACT, 0.5), _s(Tier.FACT, 0.4)]  # fmt: skip
    out = add_extra_facts(top, ranked, n=5, ratio=0.6)  # floor 0.54
    assert [s.score for s in out] == [0.9, 0.8, 0.7, 0.6]
    assert add_extra_facts(top, ranked, n=1, ratio=0.6)[2:] == [ranked[2]]
    assert add_extra_facts(top, ranked, n=0, ratio=0.6) == top


@pytest.mark.parametrize(
    ("question", "searched"),
    [
        ("How many plants did I acquire in the last month?", "How many plants did I acquire?"),
        ("How many MCU films did I watch in the last 3 months?", "How many MCU films did I watch?"),
        (
            "Which platform did I gain the most followers on over the past month?",
            "Which platform did I gain the most followers on?",
        ),
        ("What certification did I complete last month?", "What certification did I complete?"),
        (
            "I mentioned a sports event two weeks ago. What was the event?",
            "I mentioned a sports event. What was the event?",
        ),
        # ranges keep their phrase: the window only covers one end
        (
            "Which rides did I take from July to October?",
            "Which rides did I take from July to October?",
        ),
        (
            "Which airline did I fly in March and April?",
            "Which airline did I fly in March and April?",
        ),
        # nothing left to search for: keep the question
        ("Anything last month?", "Anything last month?"),
    ],
)
def test_the_time_phrase_is_left_out_of_the_search(question, searched):
    from app.memory.retriever import without_time_phrase
    from app.memory.temporal import parse_time_window

    window = parse_time_window(question, datetime(2023, 6, 15, tzinfo=UTC))
    assert window is not None
    assert without_time_phrase(question, window) == searched


def test_no_window_no_change():
    from app.memory.retriever import without_time_phrase

    assert without_time_phrase("How many plants do I have?", None) == "How many plants do I have?"
