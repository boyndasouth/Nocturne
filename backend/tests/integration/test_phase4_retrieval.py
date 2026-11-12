"""Phase 4 retrieval options against real Postgres (hash embeddings score by shared words)."""

from __future__ import annotations

from tests.integration.test_phase3_retrieval import mem  # noqa: F401  (fixture)

COMPARE = "Which device did I set up first, the smart thermostat or the mesh network system?"


def contents(results):
    return [s.note.content for s in results]


def fill(m):
    for i in range(6):  # share most of the question's words
        m.add(f"The user set up the mesh network system device {i}.")
    return m.add("The user installed a smart thermostat in March.")


def test_without_split_one_side_crowds_out_the_other(mem):  # noqa: F811
    m = mem(retrieval_top_k=3, retrieval_fact_slots=0, retrieval_extra_facts=0,
            retrieval_split_queries=False)  # fmt: skip
    fill(m)
    assert "The user installed a smart thermostat in March." not in contents(m.retrieve(COMPARE))


def test_split_keeps_both_sides(mem):  # noqa: F811
    m = mem(retrieval_top_k=3, retrieval_fact_slots=0, retrieval_extra_facts=0,
            retrieval_split_queries=True)  # fmt: skip
    thermostat = fill(m)
    out = m.retrieve(COMPARE)
    assert len(out) == 3
    got = {s.note.id: s for s in out}
    assert thermostat in got and got[thermostat].via == "split"
    assert sum("mesh network" in c for c in contents(out)) == 2


def test_extra_facts_widen_the_result_for_counting(mem):  # noqa: F811
    m = mem(retrieval_top_k=2, retrieval_extra_facts=3)
    for doctor in ("dermatologist", "ENT specialist", "primary care doctor", "dentist"):
        m.add(f"The user visited a {doctor} doctor.")
    m.add("The user likes hiking.")
    out = m.retrieve("How many doctors did I visit?")
    assert sum("doctor" in c for c in contents(out)) == 4
    assert "The user likes hiking." not in contents(out)


def test_time_words_no_longer_outrank_the_topic(mem):  # noqa: F811
    """'in the last month' becomes a window; searching its words too ranked a parking ticket
    'last month' above the plants (a dev miss: 'How many plants did I acquire...')."""
    from datetime import UTC, datetime

    may = datetime(2026, 5, 10, tzinfo=UTC)
    question = "How many plants did I acquire in the last month?"
    kw = dict(retrieval_top_k=1, retrieval_fact_slots=0, retrieval_extra_facts=0)

    def run(strip: bool) -> list[str]:
        m = mem(retrieval_strip_time_phrase=strip, **kw)
        m.add("The user got a parking ticket last month.", at=may)
        m.add("The user started a new job last month.", at=may)
        m.add("The user bought plants: a peace lily and a succulent.", at=may)
        return contents(m.retrieve(question))

    assert run(False) != ["The user bought plants: a peace lily and a succulent."]
    assert run(True) == ["The user bought plants: a peace lily and a succulent."]
