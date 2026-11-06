"""Linker, Abstractor and clustering: pure logic, no database or LLM."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import numpy as np
import pytest

from app.agents.sleep.abstractor import (
    AbstractConfig,
    AbstractorOutput,
    ExistingTheme,
    apply_output,
    format_clusters,
    overlap,
    plan_themes,
    theme_content,
)
from app.agents.sleep.cluster import average_linkage, normalize
from app.agents.sleep.linker import LinkerOutput, apply_decisions
from app.agents.sleep.state import ClusterFact, Group, NeighborRef, ThemeText, note_ref

T0 = datetime(2026, 1, 1, tzinfo=UTC)


# --- clustering -----------------------------------------------------------------------------


def _brute(vectors, threshold):
    m = normalize(vectors)
    sim = m @ m.T
    clusters = [[i] for i in range(len(vectors))]
    while True:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = sim[np.ix_(clusters[a], clusters[b])].mean()
                if s >= threshold and (best is None or s > best[0] + 1e-12):
                    best = (s, a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


@pytest.mark.parametrize("seed", range(40))
def test_average_linkage_matches_greedy_reference(seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(rng.integers(1, 5), 6))
    n = int(rng.integers(1, 30))
    v = centers[rng.integers(0, len(centers), n)] + rng.normal(scale=0.5, size=(n, 6))
    t = float(rng.uniform(0.3, 0.9))
    assert average_linkage(v, threshold=t) == _brute(v, t)


def test_average_linkage_avoids_single_linkage_chaining():
    # a~b and b~c are 0.71, a and c are 0: single linkage would chain all three.
    a, c = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    b = (a + c) / np.linalg.norm(a + c)
    assert average_linkage([a, b, c], threshold=0.7) == [[0, 1], [2]]


def test_average_linkage_min_size_and_empty():
    v = [[1, 0], [1, 0.01], [0, 1]]
    assert average_linkage(v, threshold=0.9, min_size=2) == [[0, 1]]
    assert average_linkage([], threshold=0.5) == []


# --- Linker ---------------------------------------------------------------------------------


def _group(key="0.0", n=2) -> Group:
    return Group(
        key=key,
        content="new fact",
        valid_from=T0,
        neighbors=[
            NeighborRef(ref=f"n:{UUID(int=i + 1)}", content=f"old {i}", valid_from=T0, sim=0.8)
            for i in range(n)
        ],
    )


def test_linker_maps_labels_and_reverses_the_passive_ones():
    g = _group()
    a, b = g.neighbors[0].ref, g.neighbors[1].ref
    out = apply_decisions(
        [g],
        LinkerOutput.model_validate(
            {
                "decisions": [
                    {"group": 1, "links": [{"to": "A", "type": "Elaborated-By"}]},
                    {"group": 1, "links": [{"to": "[B]", "type": "caused_by"}]},
                ]
            }
        ),
    )
    assert [(lk.src, lk.dst, lk.type) for lk in out] == [
        (a, "c:0.0", "elaborates"),
        ("c:0.0", b, "caused_by"),
    ]


def test_linker_drops_bad_groups_letters_labels_and_repeat_pairs():
    g = _group(n=1)
    out = apply_decisions(
        [g],
        LinkerOutput.model_validate(
            {
                "decisions": [
                    {"group": 2, "links": [{"to": "A", "type": "related"}]},  # no group 2
                    {"group": 1, "links": [{"to": "C", "type": "related"}]},  # no neighbour C
                    {"group": 1, "links": [{"to": "A", "type": "supersedes"}]},  # not a label
                    {"group": 1, "links": [{"to": "A", "type": "related"}]},
                    {"group": 1, "links": [{"to": "A", "type": "causes"}]},  # pair taken
                ]
            }
        ),
    )
    assert [(lk.dst, lk.type) for lk in out] == [(g.neighbors[0].ref, "related")]


# --- Abstractor -----------------------------------------------------------------------------

CFG = AbstractConfig(threshold=0.9, min_size=3, match_overlap=0.5, max_per_cycle=12)


def _facts(groups: dict[str, int]):
    """Facts in tight clusters: one axis per group name, tiny jitter."""
    facts, vecs = [], []
    axes = {name: i for i, name in enumerate(groups)}
    rng = np.random.default_rng(0)
    for name, count in groups.items():
        for j in range(count):
            v = np.zeros(len(groups) + 1)
            v[axes[name]] = 1.0
            v += rng.normal(scale=0.01, size=v.shape)
            facts.append(
                ClusterFact(
                    ref=note_ref(uuid4()), content=f"{name} {j}", valid_from=T0 + timedelta(days=j)
                )
            )
            vecs.append(v.tolist())
    return facts, vecs


def _ids(facts, prefix):
    return tuple(UUID(f.ref[2:]) for f in facts if f.content.startswith(prefix))


def test_new_clusters_become_plans_and_small_ones_are_ignored():
    facts, vecs = _facts({"bikes": 4, "exams": 3, "pets": 2})
    plans, retired = plan_themes(facts, vecs, [], CFG)
    assert [(p.theme_id, len(p.members)) for p in plans] == [(None, 4), (None, 3)]
    assert plans[0].valid_from == T0  # earliest member
    assert retired == []


def test_unchanged_theme_needs_no_plan():
    facts, vecs = _facts({"bikes": 4})
    theme = ExistingTheme(uuid4(), "Bikes: ...", _ids(facts, "bikes"))
    assert plan_themes(facts, vecs, [theme], CFG) == ([], [])


def test_grown_theme_is_updated_not_duplicated():
    facts, vecs = _facts({"bikes": 6})
    theme = ExistingTheme(uuid4(), "Bikes: ...", _ids(facts, "bikes")[:3])
    [plan], retired = plan_themes(facts, vecs, [theme], CFG)
    assert plan.theme_id == theme.id and len(plan.members) == 6
    assert plan.previous == "Bikes: ..." and retired == []


def test_superseded_member_triggers_a_resummary():
    facts, vecs = _facts({"bikes": 4})
    gone = uuid4()  # a member that's no longer active
    theme = ExistingTheme(uuid4(), "Bikes", (*_ids(facts, "bikes"), gone))
    [plan], _ = plan_themes(facts, vecs, [theme], CFG)
    assert plan.theme_id == theme.id
    assert note_ref(gone) not in plan.members


def test_dissolved_theme_is_retired_but_a_scattered_one_is_kept():
    facts, vecs = _facts({"bikes": 4})
    dissolved = ExistingTheme(uuid4(), "Old", (uuid4(), uuid4(), uuid4()))  # all inactive
    kept_ids = _ids(facts, "bikes")
    matched = ExistingTheme(uuid4(), "Bikes", kept_ids)
    # A second theme over the same, still active, facts: unmatched (one-to-one), but its
    # members are active, so it stays.
    twin = ExistingTheme(uuid4(), "Bikes again", kept_ids)
    plans, retired = plan_themes(facts, vecs, [dissolved, matched, twin], CFG)
    assert plans == [] and retired == [dissolved.id]


def test_cap_keeps_the_biggest_clusters():
    facts, vecs = _facts({"a": 3, "b": 5, "c": 4})
    plans, _ = plan_themes(facts, vecs, [], AbstractConfig(threshold=0.9, max_per_cycle=2))
    assert [len(p.members) for p in plans] == [5, 4]


def test_llm_sees_the_newest_members_with_dates():
    facts, vecs = _facts({"bikes": 5})
    [plan], _ = plan_themes(facts, vecs, [], AbstractConfig(threshold=0.9, facts_per_theme=2))
    assert [f.content for f in plan.facts] == ["bikes 3", "bikes 4"]
    assert "- (since 2026-01-05) bikes 4" in format_clusters([plan])


def test_overlap_coefficient():
    assert overlap({"a", "b", "c"}, {"a", "b", "c", "d", "e"}) == 1.0
    assert overlap({"a"}, set()) == 0.0


def test_abstractor_output_parsing_and_content():
    facts, vecs = _facts({"bikes": 3, "exams": 3})
    plans, _ = plan_themes(facts, vecs, [], CFG)
    texts = apply_output(
        plans,
        AbstractorOutput.model_validate(
            {
                "themes": [
                    {"cluster": 1, "title": "User's bikes.", "summary": "Owns  three\nbikes.",
                     "keywords": [" bikes ", "", "cycling"]},
                    {"cluster": 2, "title": None, "summary": "unrelated"},
                    {"cluster": 9, "title": "nope", "summary": "x"},
                ]
            }
        ),
    )  # fmt: skip
    assert list(texts) == [plans[0].key]
    t = texts[plans[0].key]
    assert t.keywords == ["bikes", "cycling"]
    assert theme_content(t) == "User's bikes: Owns three bikes."
    assert len(theme_content(ThemeText(title="x" * 200, summary="y" * 900))) == 600
