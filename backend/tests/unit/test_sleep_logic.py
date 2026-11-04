"""Core sleep-cycle decisions: dedup thresholds, neighbour selection, supersession ordering,
and the apply plan (duplicate chains, backdated facts, supersession chains)."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest

from app.agents.sleep import contradiction, dedup
from app.agents.sleep.apply import plan_apply, resolve_ref
from app.agents.sleep.contradiction import Timed, resolve
from app.agents.sleep.dedup import Band, DedupThresholds, classify, plan_dedup
from app.agents.sleep.judge import format_groups, letter_to_ref, neighbor_lists
from app.agents.sleep.state import Conflict, FactCandidate, Group, NeighborRef
from app.memory.models import MemoryNote, Neighbor, Status, Tier

JAN = datetime(2026, 1, 10, tzinfo=UTC)
MAR = datetime(2026, 3, 2, tzinfo=UTC)
MAY = datetime(2026, 5, 20, tzinfo=UTC)
T = DedupThresholds(auto=0.97, judge=0.82)


def cand(key: str, content: str = "x", when: datetime = MAR) -> FactCandidate:
    return FactCandidate(key=key, content=content, valid_from=when, source_episode_ids=[uuid4()])


def nref(ref: str, sim: float, when: datetime = JAN) -> NeighborRef:
    return NeighborRef(ref=ref, content=ref, valid_from=when, sim=sim)


def fact(content: str = "f", when: datetime = JAN) -> MemoryNote:
    return MemoryNote(
        id=uuid4(), tier=Tier.FACT, content=content, strength=1.0, access_count=0,
        last_accessed=None, valid_from=when, valid_to=None, recorded_at=when,
        status=Status.ACTIVE, version=1,
    )  # fmt: skip


# --- dedup thresholds -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sim", "band"),
    [
        (1.0, Band.AUTO),
        (0.97, Band.AUTO),  # inclusive lower edge
        (0.9699, Band.JUDGE),
        (0.82, Band.JUDGE),
        (0.8199, Band.DISTINCT),
        (-1.0, Band.DISTINCT),
    ],
)
def test_similarity_bands(sim, band):
    assert classify(sim, T) is band


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        DedupThresholds(auto=0.8, judge=0.9)


def test_plan_dedup_auto_merges_judges_and_ignores_distinct():
    a, b, c = cand("0.0"), cand("0.1"), cand("0.2")
    auto, groups = plan_dedup(
        [a, b, c],
        [
            [nref("n:x", 0.99), nref("n:y", 0.9)],  # auto -> best neighbour
            [nref("n:y", 0.9), nref("n:z", 0.5)],  # judge, distinct neighbour dropped
            [nref("n:z", 0.5)],  # distinct
        ],
        T,
    )
    assert auto == {"0.0": "n:x"}
    assert [g.key for g in groups] == ["0.1"]
    assert [n.ref for n in groups[0].neighbors] == ["n:y"]


def test_contradiction_scale_similarity_is_not_auto_merged():
    """'lives in Austin' vs 'lives in Dallas' can score ~0.9 with bge-small. That must reach
    a judge, never a silent auto-merge."""
    auto, groups = plan_dedup([cand("0.0")], [[nref("n:austin", 0.91)]], T)
    assert auto == {} and len(groups) == 1


# --- neighbours -----------------------------------------------------------------------------


def test_neighbors_only_look_at_earlier_candidates_and_respect_threshold_and_k():
    cands = [cand("0.0"), cand("0.1"), cand("0.2")]
    sims = np.array([[1, 0.9, 0.9], [0.9, 1, 0.3], [0.9, 0.3, 1]], dtype=np.float32)
    existing = [[], [], [Neighbor(note=fact(), sim=0.95), Neighbor(note=fact(), sim=0.1)]]
    out = neighbor_lists(cands, sims, existing, min_sim=0.8, k=5)
    assert out[0] == []  # nothing earlier
    assert [n.ref for n in out[1]] == ["c:0.0"]
    assert [n.ref[:2] for n in out[2]] == ["n:", "c:"]  # by sim, below-threshold dropped
    assert len(neighbor_lists(cands, sims, existing, min_sim=0.0, k=1)[2]) == 1


def test_neighbors_skip_ineligible_candidates_on_both_sides():
    cands = [cand("0.0"), cand("0.1")]
    sims = np.ones((2, 2), dtype=np.float32)
    out = neighbor_lists(cands, sims, [[], []], min_sim=0.5, k=3, eligible=[False, True])
    assert out == [[], []]


def test_letters_map_to_refs_and_bad_letters_are_ignored():
    g = Group(key="0.0", content="new", valid_from=MAR, neighbors=[nref("n:a", 0.9)])
    assert letter_to_ref(g, "A") == "n:a"
    assert letter_to_ref(g, " [a] ") == "n:a"
    assert letter_to_ref(g, "B") is None
    assert letter_to_ref(g, "AB") is None
    assert letter_to_ref(g, None) is None
    assert "[A] (true since 2026-01-10) n:a" in format_groups([g], with_dates=True)


def test_dedup_decisions_ignore_out_of_range_groups():
    g = Group(key="0.3", content="new", valid_from=MAR, neighbors=[nref("n:a", 0.9)])
    out = dedup.DedupOutput.model_validate(
        {"decisions": [{"group": 1, "duplicate_of": "A"}, {"group": 7, "duplicate_of": "A"}]}
    )
    assert dedup.apply_decisions([g], out) == {"0.3": "n:a"}


def test_conflict_decisions_carry_both_valid_times():
    g = Group(key="0.0", content="Dallas", valid_from=MAR, neighbors=[nref("n:a", 0.8, JAN)])
    out = contradiction.ContradictionOutput.model_validate(
        {"decisions": [{"group": 1, "conflicts_with": ["A", "A", "Z"], "reason": "moved"}]}
    )
    [c] = contradiction.apply_decisions([g], out)
    assert (c.new, c.old, c.new_valid_from, c.old_valid_from) == ("c:0.0", "n:a", MAR, JAN)


# --- supersession ---------------------------------------------------------------------------


def test_later_valid_time_wins_regardless_of_argument_order():
    old, new = Timed("n:austin", JAN, 0), Timed("c:dallas", MAR, 1)
    for a, b in [(old, new), (new, old)]:
        s = resolve(a, b)
        assert (s.winner, s.loser, s.valid_to) == ("c:dallas", "n:austin", MAR)


def test_backdated_new_fact_loses_to_newer_existing_fact():
    """Notes ingested today about January don't override what's been true since March."""
    s = resolve(Timed("c:austin", JAN, 1), Timed("n:dallas", MAR, 0))
    assert (s.winner, s.loser, s.valid_to) == ("n:dallas", "c:austin", MAR)


def test_same_valid_time_prefers_the_fact_learned_later():
    s = resolve(Timed("n:friday", MAR, 0), Timed("c:monday", MAR, 3))
    assert s.winner == "c:monday"


# --- apply plan -----------------------------------------------------------------------------


def test_duplicate_chain_resolves_to_existing_note():
    dups = {"0.1": "c:0.0", "0.0": "n:x"}
    assert resolve_ref("c:0.1", dups) == "n:x"
    assert resolve_ref("c:0.2", dups) == "c:0.2"
    with pytest.raises(ValueError):
        resolve_ref("c:a", {"a": "c:b", "b": "c:a"})


def test_plan_folds_in_cycle_duplicates_and_merges_into_existing():
    x = uuid4()
    a, b, c = cand("0.0", when=MAR), cand("0.1", when=JAN), cand("0.2")
    plan = plan_apply([a, b, c], {"0.1": "c:0.0", "0.2": "n:" + str(x)}, [], {x: JAN})
    [ins] = plan.inserts
    assert ins.key == "0.0" and ins.folded_keys == ["0.1"]
    assert ins.valid_from == JAN  # earliest evidence
    assert ins.source_episode_ids == [*a.source_episode_ids, *b.source_episode_ids]
    [m] = plan.merges
    assert m.note_id == x and m.add_episode_ids == c.source_episode_ids
    assert plan.supersessions == []


def test_plan_supersession_chain_uses_earliest_winner():
    """A (Jan) -> B (Mar) -> C (May), all pairwise conflicting: A ends when B starts, and the
    links form a chain C -> B -> A rather than C -> A."""
    a = uuid4()
    b, c = cand("0.0", "B", MAR), cand("0.1", "C", MAY)

    def conflict(new: str, old: str, nv: datetime, ov: datetime) -> Conflict:
        return Conflict(new=new, old=old, new_valid_from=nv, old_valid_from=ov)

    plan = plan_apply(
        [b, c],
        {},
        [
            conflict("c:0.0", f"n:{a}", MAR, JAN),
            conflict("c:0.1", f"n:{a}", MAY, JAN),
            conflict("c:0.1", "c:0.0", MAY, MAR),
        ],
        {a: JAN},
    )
    got = [(s.winner, s.loser, s.valid_to) for s in plan.supersessions]
    assert got == [("c:0.0", f"n:{a}", MAR), ("c:0.1", "c:0.0", MAY)]


def test_plan_skips_conflict_between_facts_merged_into_one():
    x = uuid4()
    plan = plan_apply(
        [cand("0.0")],
        {"0.0": f"n:{x}"},
        [Conflict(new="c:0.0", old=f"n:{x}", new_valid_from=MAR, old_valid_from=JAN)],
        {x: JAN},
    )
    assert plan.supersessions == []


def test_plan_conflict_on_merged_candidate_uses_target_note_time():
    """New 'Dallas' (Mar) duplicates existing Dallas fact D (Feb) and conflicts with Austin A
    (Jan). The supersession is D over A, dated by D's own valid time."""
    a, d = uuid4(), uuid4()
    feb = MAR - timedelta(days=30)
    plan = plan_apply(
        [cand("0.0", "Dallas", MAR)],
        {"0.0": f"n:{d}"},
        [Conflict(new="c:0.0", old=f"n:{a}", new_valid_from=MAR, old_valid_from=JAN)],
        {a: JAN, d: feb},
    )
    [s] = plan.supersessions
    assert (s.winner, s.loser, s.valid_to) == (f"n:{d}", f"n:{a}", feb)
    assert plan.inserts == []
