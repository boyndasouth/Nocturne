from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.memory.models import Candidate, MemoryNote, Status, Tier
from app.memory.scoring import ScoreWeights, combined_score, rank_candidates, recency_score

NOW = datetime(2026, 3, 1, tzinfo=UTC)


def note(*, strength: float = 1.0, age_days: float = 0.0) -> MemoryNote:
    t = NOW - timedelta(days=age_days)
    return MemoryNote(
        id=uuid4(), tier=Tier.EPISODE, content="x", strength=strength, access_count=0,
        last_accessed=None, valid_from=t, valid_to=None, recorded_at=t, status=Status.ACTIVE,
        version=1,
    )  # fmt: skip


def test_recency_is_one_now_and_decays():
    assert recency_score(NOW, NOW, 30) == pytest.approx(1.0)
    assert recency_score(NOW - timedelta(days=30), NOW, 30) == pytest.approx(0.3679, abs=1e-4)


def test_recency_of_future_event_is_capped_at_one():
    assert recency_score(NOW + timedelta(days=5), NOW, 30) == 1.0


def test_recency_rejects_bad_tau():
    with pytest.raises(ValueError):
        recency_score(NOW, NOW, 0)


def test_combined_score_matches_formula_with_default_weights():
    w = ScoreWeights()
    s = combined_score(semantic_sim=0.8, keyword_score=0.5, strength=0.4, recency=0.2, weights=w)
    assert s == pytest.approx(0.6 * 0.8 + 0.2 * 0.5 + 0.1 * 0.4 + 0.1 * 0.2)


def test_combined_score_clamps_inputs():
    w = ScoreWeights()
    hi = combined_score(semantic_sim=5, keyword_score=5, strength=5, recency=5, weights=w)
    lo = combined_score(semantic_sim=-1, keyword_score=-1, strength=-1, recency=-1, weights=w)
    assert hi == pytest.approx(1.0)
    assert lo == 0.0


def test_combined_score_treats_nan_as_zero():
    w = ScoreWeights(alpha=1, beta=0, gamma=0, delta=0)
    nan = float("nan")
    assert combined_score(semantic_sim=nan, keyword_score=0, strength=0, recency=0, weights=w) == 0


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        ScoreWeights(alpha=-0.1)


def test_rank_orders_by_score_and_truncates():
    a = Candidate(note=note(), semantic_sim=0.9, keyword_score=0.0)
    b = Candidate(note=note(), semantic_sim=0.2, keyword_score=0.0)
    c = Candidate(note=note(), semantic_sim=0.5, keyword_score=0.9)
    ranked = rank_candidates(
        [b, c, a], now=NOW, weights=ScoreWeights(), recency_tau_days=30, top_k=2
    )
    assert [r.note.id for r in ranked] == [a.note.id, c.note.id]


def test_rank_breaks_ties_by_recency():
    old = Candidate(note=note(age_days=10), semantic_sim=0.5, keyword_score=0.0)
    new = Candidate(note=note(age_days=0), semantic_sim=0.5, keyword_score=0.0)
    w = ScoreWeights(alpha=1, beta=0, gamma=0, delta=0)  # recency carries no score weight
    ranked = rank_candidates([old, new], now=NOW, weights=w, recency_tau_days=30, top_k=2)
    assert ranked[0].note.id == new.note.id


def test_strength_and_recency_can_reorder_equal_similarity():
    weak_old = Candidate(note=note(strength=0.1, age_days=90), semantic_sim=0.7, keyword_score=0)
    strong_new = Candidate(note=note(strength=1.0, age_days=1), semantic_sim=0.7, keyword_score=0)
    ranked = rank_candidates(
        [weak_old, strong_new], now=NOW, weights=ScoreWeights(), recency_tau_days=30, top_k=2
    )
    assert ranked[0].note.id == strong_new.note.id
    assert ranked[0].score > ranked[1].score
