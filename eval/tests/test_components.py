"""Judge, LLM cache/meter, baselines and reports: no database, no network."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.llm.client import LLMResponse
from app.llm.embeddings import HashEmbedder
from eval.baselines.full_context import FullContextSystem, fit_recent
from eval.baselines.naive_rag import NaiveRagSystem, top_k
from eval.datasets.schema import Conversation, Question, Session, Turn
from eval.fake import FakeLLM
from eval.judge import RUBRICS, Judge, build_prompt, parse_verdict, rubric_for
from eval.llm import EvalLLM, ResponseCache
from eval.report import format_table, latest, percentile, summarize


def q(**kw) -> Question:
    base = {
        "id": "q1",
        "conversation_id": "c1",
        "question": "Where do I live?",
        "answer": "Dallas",
        "category": "knowledge-update",
        "question_type": "knowledge-update",
        "asked_at": datetime(2023, 6, 1, tzinfo=UTC),
    }
    return Question(**{**base, **kw})


# --- Judge ----------------------------------------------------------------------------------


def test_rubric_selection():
    assert rubric_for(q()) == "knowledge-update"
    assert rubric_for(q(category="temporal-reasoning", question_type="temporal-reasoning")) == (
        "temporal"
    )
    assert rubric_for(q(category="temporal", question_type="2")) == "temporal"  # LoCoMo
    assert rubric_for(q(question_type="single-session-preference", category="x")) == "preference"
    assert rubric_for(q(abstention=True, category="abstention")) == "abstention"
    assert rubric_for(q(category="multi-hop", question_type="1")) == "default"


def test_prompt_contains_question_gold_and_response():
    p = build_prompt(q(), "You moved to Dallas.")
    assert "Question: Where do I live?" in p
    assert "Correct Answer: Dallas" in p
    assert "Model Response: You moved to Dallas." in p
    assert all("{response}" in r for r in RUBRICS.values())


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("yes", True),
        ("No.", False),
        ("**Yes** - it names Dallas", True),
        ("", None),
        ("The answer is correct", None),
        ("no, not yes", False),
    ],
)
def test_parse_verdict(text, want):
    assert parse_verdict(text) is want


async def test_judge_with_fake_llm():
    judge = Judge(FakeLLM())
    assert (await judge.grade(q(), "You live in Dallas now.")).correct is True
    assert (await judge.grade(q(), "You live in Austin.")).correct is False
    abst = q(abstention=True, category="abstention", answer="No dog was mentioned.")
    assert (await judge.grade(abst, "I don't have that in memory.")).correct is True


# --- Cache and meter ------------------------------------------------------------------------


class CountingLLM:
    name, model = "p", "m"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **_):
        self.calls += 1
        return LLMResponse(
            text=f"r{self.calls}", model="m", input_tokens=10, output_tokens=2, latency_ms=5.0
        )

    async def aclose(self) -> None:
        pass


async def test_cache_replays_and_meter_counts_hits_at_full_cost(tmp_path):
    inner = CountingLLM()
    cache = ResponseCache(tmp_path / "c.sqlite")
    llm = EvalLLM(inner, cache=cache)
    msgs = [{"role": "user", "content": "hi"}]
    first = await llm.chat(msgs, temperature=0.0)
    again = await llm.chat(msgs, temperature=0.0)
    other = await llm.chat(msgs, temperature=0.5)  # different request, different key
    assert (first.text, again.text, other.text) == ("r1", "r1", "r2")
    assert inner.calls == 2
    assert (llm.meter.calls, llm.meter.cache_hits, llm.meter.tokens) == (3, 1, 36)

    reopened = EvalLLM(CountingLLM(), cache=ResponseCache(tmp_path / "c.sqlite"))
    assert (await reopened.chat(msgs, temperature=0.0)).text == "r1"  # survives restarts


async def test_cache_key_includes_provider_extras(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    msgs = [{"role": "user", "content": "hi"}]
    low = EvalLLM(CountingLLM(), cache=cache, extra_key={"reasoning_effort": "low"})
    high = EvalLLM(CountingLLM(), cache=cache, extra_key={"reasoning_effort": "high"})
    await low.chat(msgs)
    await high.chat(msgs)
    assert high.meter.cache_hits == 0


# --- Baselines ------------------------------------------------------------------------------


def test_fit_recent_keeps_newest_lines_within_budget():
    lines = ["a" * 40, "b" * 40, "c" * 40]  # 41 chars each with newline
    kept, dropped = fit_recent(lines, budget_tokens=21)  # 84 chars
    assert kept == ["b" * 40, "c" * 40] and dropped == 1
    assert fit_recent(lines, budget_tokens=1000) == (lines, 0)
    assert fit_recent(lines, budget_tokens=1) == ([], 3)


def test_top_k_orders_by_cosine_and_breaks_ties_to_newer():
    m = np.array([[1, 0], [0, 1], [1, 0], [0.9, 0.1]], dtype=np.float32)
    assert top_k(m, np.array([1, 0], dtype=np.float32), 3) == [2, 0, 3]
    assert top_k(np.zeros((0, 2), dtype=np.float32), np.array([1, 0]), 3) == []


def _conv() -> Conversation:
    t = datetime(2023, 1, 1, 9, tzinfo=UTC)
    return Conversation(
        id="c1",
        benchmark="t",
        style="chat",
        sessions=(
            Session("s1", t, (Turn("user", "I live in Austin"), Turn("assistant", "Nice."))),
            Session(
                "s2",
                t.replace(month=3),
                (Turn("user", "I moved to Dallas"), Turn("assistant", "Congrats.")),
            ),
        ),
        questions=(q(asked_at=t.replace(month=4)),),
    )


async def test_naive_rag_retrieves_and_hides_future_turns():
    rag = NaiveRagSystem(answer_llm=FakeLLM(), embedder=HashEmbedder(), top_k=1, max_tokens=100)
    stats = await rag.prepare(_conv())
    assert (stats.episodes, stats.memory_items) == (2, 2)
    ans = await rag.answer(q(question="Dallas move?", asked_at=datetime(2023, 4, 1, tzinfo=UTC)))
    assert "Dallas" in ans.text and ans.retrieved == 1 and ans.retrieval_ms is not None
    early = await rag.answer(q(question="Dallas move?", asked_at=datetime(2023, 2, 1, tzinfo=UTC)))
    assert "Dallas" not in early.text  # the March turn didn't exist yet


async def test_full_context_truncates_oldest_and_reports_it():
    fc = FullContextSystem(answer_llm=FakeLLM(), budget_tokens=25, max_tokens=100)
    stats = await fc.prepare(_conv())
    assert fc.truncated_episodes == 1 and stats.memory_items == 1
    ans = await fc.answer(q())
    assert "Dallas" in ans.text and ans.retrieval_ms is None


# --- Reports --------------------------------------------------------------------------------


def _row(system, cat, correct, status="ok", ms=None, qid=None):
    return {
        "system": system,
        "category": cat,
        "correct": correct,
        "status": status,
        "retrieval_ms": ms,
        "question_id": qid or f"{cat}-{correct}-{ms}",
        "answer_tokens": 10,
        "judge_tokens": 1,
    }


def test_latest_keeps_the_last_attempt():
    rows = [
        {"system": "a", "question_id": "1", "status": "error"},
        {"system": "a", "question_id": "1", "status": "ok"},
    ]
    assert latest(rows, ("system", "question_id")) == [rows[1]]


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([float(i) for i in range(1, 101)], 95) == 95.0


def test_summarize_category_accuracy_task_average_and_errors():
    rows = [
        _row("n", "ku", True, ms=10.0, qid="1"),
        _row("n", "ku", False, ms=20.0, qid="2"),
        _row("n", "tr", True, ms=30.0, qid="3"),
        _row("n", "tr", None, ms=40.0, qid="4"),  # unparseable verdict: counts as wrong
        _row("n", "tr", None, status="error", qid="5"),
    ]
    convs = [
        {"system": "n", "status": "ok", "episodes": 10, "memory_items": 4, "sleep_tokens": 100}
    ]
    summary, [tot] = summarize(rows, convs)
    by = {s["category"]: s for s in summary}
    assert (by["ku"]["accuracy"], by["tr"]["accuracy"], by["tr"]["unjudged"]) == (0.5, 0.5, 1)
    assert tot["questions"] == 4 and tot["errors"] == 1
    assert tot["accuracy"] == 0.5 and tot["task_avg_accuracy"] == 0.5
    assert (tot["retrieval_p50_ms"], tot["retrieval_p95_ms"]) == (20.0, 40.0)
    assert (tot["memory_ratio"], tot["sleep_tokens"]) == (0.4, 100)
    table = format_table(summary, [tot])
    assert "ku" in table and "50% (1/2)" in table and "task-averaged" in table


# --- Sleep cadence and quota stop -----------------------------------------------------------


def test_sleep_points_every_n_sessions_always_ends_with_the_last():
    from eval.systems import sleep_points

    ends = list(range(1, 8))
    assert sleep_points(ends, 1) == ends
    assert sleep_points(ends, 3) == [3, 6, 7]
    assert sleep_points(ends, 7) == [7]
    assert sleep_points(ends, 50) == [7]
    assert sleep_points([], 5) == []


def test_quota_error_is_found_through_wrapped_exceptions():
    from app.llm.client import QuotaExhaustedError
    from eval.runner import quota_error

    quota = QuotaExhaustedError("gemini: daily quota used up")
    try:
        try:
            raise quota
        except QuotaExhaustedError as exc:
            raise RuntimeError("sleep cycle failed") from exc
    except RuntimeError as wrapped:
        assert quota_error(wrapped) is quota
    assert quota_error(RuntimeError("other")) is None
