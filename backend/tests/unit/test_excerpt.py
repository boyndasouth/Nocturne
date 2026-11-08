from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.memory.excerpt import excerpt, fair_caps, fit, query_terms
from app.memory.models import MemoryNote, ScoredNote, Status, Tier
from app.memory.retriever import reserve_fact_slots

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_query_terms_drop_stopwords_keep_short_tokens_with_digits_and_fold_plurals():
    assert query_terms("What move did you make after 27. Kg2 Bd5+?") == {
        "move", "make", "27", "kg2", "bd5"
    }  # fmt: skip
    assert query_terms("How many doctors did I visit?") == {"many", "doctor", "visit"}


def test_fair_caps_give_short_items_everything_and_split_the_rest():
    assert fair_caps([100, 5000, 300, 8000], 4000, item_max=3000) == [100, 1800, 300, 1800]
    assert fair_caps([100, 200], 4000, item_max=3000) == [100, 200]  # all fit
    assert fair_caps([9000], 99999, item_max=3000) == [3000]  # never above item_max
    assert fair_caps([5000, 5000], 100, item_max=3000) == [200, 200]  # floor of 200
    assert fair_caps([], 100, item_max=3000) == []


def _turn(n_filler: int, needle: str) -> str:
    return (
        "User: Can you help me plan?\nAssistant: "
        + " ".join(f"Point {i} is general advice." for i in range(n_filler))
        + f" {needle} "
        + " ".join(f"Closing note {i}." for i in range(n_filler))
    )


def test_excerpt_finds_the_passage_deep_inside_a_long_reply():
    text = _turn(80, "Dilute tea tree oil one part to ten parts carrier oil.")
    out = excerpt(text, query_terms("What ratio for tea tree oil did you suggest?"), 400)
    assert len(out) <= 420
    assert out.startswith("User: Can you help me plan?\nAssistant: … ")
    assert "one part to ten parts" in out and out.endswith("…")


def test_excerpt_without_a_match_keeps_the_beginning():
    text = _turn(80, "Nothing relevant.")
    out = excerpt(text, query_terms("What is my cat's name?"), 300)
    assert out.startswith("User: Can you help me plan?\nAssistant: Point 0 is general advice.")


def test_excerpt_of_one_huge_sentence_centres_on_the_match():
    text = "Assistant: " + "word " * 400 + "the code is ZX81 " + "word " * 400
    out = excerpt(text, query_terms("What was the code ZX81?"), 200)
    assert "ZX81" in out and len(out) <= 220


def test_fit_leaves_short_texts_alone():
    texts = ["short", "x" * 50]
    assert fit(texts, "anything", budget=1000, item_max=3000) == texts


def _s(tier: Tier, score: float) -> ScoredNote:
    note = MemoryNote(
        id=uuid4(), tier=tier, content=f"{tier.value} {score}", keywords=[], strength=1.0,
        access_count=0, last_accessed=None, valid_from=T0 + timedelta(minutes=score * 100),
        valid_to=None, recorded_at=T0, status=Status.ACTIVE, source_episode_ids=[], version=1,
    )  # fmt: skip
    return ScoredNote(note=note, score=score, semantic_sim=score, keyword_score=0, recency=1)


def test_fact_slots_replace_the_weakest_turns_with_good_enough_facts():
    ranked = [_s(Tier.EPISODE, x) for x in (0.9, 0.8, 0.7, 0.6)]
    ranked += [_s(Tier.FACT, 0.58), _s(Tier.FACT, 0.55), _s(Tier.FACT, 0.2)]
    out = reserve_fact_slots(ranked, k=4, slots=3, ratio=0.6)
    assert [(s.note.tier, s.score) for s in out] == [
        (Tier.EPISODE, 0.9),
        (Tier.EPISODE, 0.8),
        (Tier.FACT, 0.58),
        (Tier.FACT, 0.55),
    ]  # the 0.2 fact is below 0.6 x 0.9, so it stays out


def test_fact_slots_change_nothing_when_facts_are_already_there():
    ranked = [_s(Tier.FACT, 0.9), _s(Tier.FACT, 0.8), _s(Tier.FACT, 0.7), _s(Tier.EPISODE, 0.6)]
    assert reserve_fact_slots(ranked, k=3, slots=2, ratio=0.6) == ranked[:3]
    assert reserve_fact_slots([], k=3, slots=2, ratio=0.6) == []
