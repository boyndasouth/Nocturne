"""Forgetter maths and Auditor sampling/scoring: no database or LLM."""

from __future__ import annotations

import math

import pytest

from app.agents.sleep.auditor import (
    AuditItem,
    AuditorOutput,
    format_items,
    sample_items,
    score_output,
)
from app.agents.sleep.forgetter import ForgetConfig, days_until_archived
from app.llm.client import _reset_hint


def test_default_thresholds_mean_about_a_month_and_five_months():
    cfg = ForgetConfig()
    assert days_until_archived(1.0, 30, cfg.episode_theta) == pytest.approx(29.8, abs=0.1)
    assert days_until_archived(1.0, 30, cfg.fact_theta) == pytest.approx(150.1, abs=0.2)
    # Reinforced twice (τ 30 -> 45 -> 67.5): a used fact lasts much longer.
    assert days_until_archived(1.0, 67.5, cfg.fact_theta) > 330
    assert days_until_archived(0.001, 30, cfg.fact_theta) == 0.0
    assert math.isclose(math.exp(-1), cfg.episode_theta, abs_tol=0.01)


def _items(kinds: dict[str, int]) -> list[AuditItem]:
    return [AuditItem(k, f"{k} {i}") for k, n in kinds.items() for i in range(n)]


def test_sampling_spreads_across_kinds_and_is_reproducible():
    items = _items({"link": 40, "supersede": 2, "theme": 3, "merge": 1})
    picked = sample_items(items, n=8, seed="cycle-1")
    kinds = [it.kind for it in picked]
    assert kinds.count("supersede") == 2 and kinds.count("merge") == 1
    assert kinds.count("theme") == 2 and kinds.count("link") == 3  # round-robin
    assert picked == sample_items(items, n=8, seed="cycle-1")
    assert sample_items(items, n=8, seed="cycle-2") != picked  # other links drawn
    assert sample_items([], n=8, seed="x") == []
    assert len(sample_items(_items({"link": 3}), n=8, seed="x")) == 3


def test_scoring_ignores_bad_items_and_labels():
    items = _items({"link": 2, "theme": 1})
    out = AuditorOutput.model_validate(
        {
            "verdicts": [
                {"item": 1, "verdict": "Good", "note": "ok"},
                {"item": 2, "verdict": "wrong", "note": "backwards"},
                {"item": 2, "verdict": "good"},  # repeat: first one counts
                {"item": 3, "verdict": "questionable"},
                {"item": 4, "verdict": "good"},  # no item 4
                {"item": 3, "verdict": "meh"},
            ]
        }
    )
    result = score_output(items, out)
    assert result.score == 0.5  # (1 + 0 + 0.5) / 3
    assert result.by_kind() == {"link": 0.5, "theme": 0.5}
    assert [v["verdict"] for v in result.verdicts] == ["good", "wrong", "questionable"]
    assert score_output(items, AuditorOutput()).score is None
    assert format_items(items[:2]).startswith("Item 1\nlink 0\n\nItem 2\nlink 1")


@pytest.mark.parametrize(
    ("provider", "words"),
    [("gemini", "midnight Pacific"), ("judge-groq", "rolling 24 hours"), ("openrouter", "reset")],
)
def test_quota_reset_hints(provider, words):
    assert words in _reset_hint(provider)
