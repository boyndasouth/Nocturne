"""Retriever (spec §5).

Phase 1: hybrid search over the episode tier only.
Phase 3 adds themes → facts → one-hop graph expansion → episode fallback, plus temporal
filtering on valid_from/valid_to.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.llm.embeddings import Embedder, aembed_query
from app.llm.tracing import Tracer
from app.memory.models import ScoredNote, Status, Tier
from app.memory.scoring import ScoreWeights, rank_candidates
from app.memory.store import MemoryStore


class HierarchicalRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        *,
        weights: ScoreWeights,
        recency_tau_days: float,
        top_k: int = 8,
        candidate_pool: int = 40,
        tracer: Tracer | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._weights = weights
        self._tau = recency_tau_days
        self._top_k = top_k
        self._pool = candidate_pool
        self._tracer = tracer or Tracer()

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        now: datetime | None = None,
        as_of: datetime | None = None,
    ) -> list[ScoredNote]:
        """`now` drives recency scoring. `as_of`, used only for simulated time in benchmark
        replay, hides notes recorded later. Live mode leaves it None so app/DB clock skew
        can't hide a note that was just written."""
        k = top_k or self._top_k
        now = now or datetime.now(UTC)
        with self._tracer.observe("retrieve", as_type="retriever", input={"query": query}) as obs:
            qvec = await aembed_query(self._embedder, query)
            candidates = await self._store.hybrid_candidates(
                qvec,
                query,
                tiers=[Tier.EPISODE],
                statuses=[Status.ACTIVE],
                pool_size=max(self._pool, k),
                as_of=as_of,
            )
            ranked = rank_candidates(
                candidates, now=now, weights=self._weights, recency_tau_days=self._tau, top_k=k
            )
            obs.update(
                output=[
                    {"id": str(s.note.id), "score": round(s.score, 4), "tier": s.note.tier}
                    for s in ranked
                ],
                metadata={"candidates": len(candidates)},
            )
            return ranked
