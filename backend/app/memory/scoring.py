"""Retrieval scoring (spec §5).

    score = α·semantic_sim + β·keyword_score + γ·strength + δ·recency

Every input is clamped to [0, 1], so the score is bounded by α+β+γ+δ.
This module is pure; it doesn't touch the database.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.memory.models import Candidate, ScoredNote


@dataclass(frozen=True)
class ScoreWeights:
    alpha: float = 0.6  # semantic similarity
    beta: float = 0.2  # keyword (full-text) score
    gamma: float = 0.1  # memory strength
    delta: float = 0.1  # recency

    def __post_init__(self) -> None:
        if min(self.alpha, self.beta, self.gamma, self.delta) < 0:
            raise ValueError("score weights must be non-negative")


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def recency_score(event_time: datetime, now: datetime, tau_days: float) -> float:
    """exp(-age/τ). 1.0 for events at or after `now`, decaying towards 0."""
    if tau_days <= 0:
        raise ValueError("tau_days must be positive")
    age_days = max(0.0, (now - event_time).total_seconds() / 86_400)
    return math.exp(-age_days / tau_days)


def combined_score(
    *,
    semantic_sim: float,
    keyword_score: float,
    strength: float,
    recency: float,
    weights: ScoreWeights,
) -> float:
    return (
        weights.alpha * _clamp01(semantic_sim)
        + weights.beta * _clamp01(keyword_score)
        + weights.gamma * _clamp01(strength)
        + weights.delta * _clamp01(recency)
    )


def rank_candidates(
    candidates: Sequence[Candidate],
    *,
    now: datetime,
    weights: ScoreWeights,
    recency_tau_days: float,
    top_k: int,
) -> list[ScoredNote]:
    """Score, sort (score desc, then newer first, then id for determinism) and truncate."""
    from app.memory.models import ScoredNote

    scored: list[ScoredNote] = []
    for c in candidates:
        rec = recency_score(c.note.valid_from, now, recency_tau_days)
        scored.append(
            ScoredNote(
                note=c.note,
                score=combined_score(
                    semantic_sim=c.semantic_sim,
                    keyword_score=c.keyword_score,
                    strength=c.note.strength,
                    recency=rec,
                    weights=weights,
                ),
                semantic_sim=c.semantic_sim,
                keyword_score=c.keyword_score,
                recency=rec,
            )
        )
    scored.sort(key=lambda s: (-s.score, -s.note.valid_from.timestamp(), str(s.note.id)))
    return scored[: max(0, top_k)]
