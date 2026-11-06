"""Hierarchical retriever (spec §5).

    1. Hybrid search (vector ∪ full-text) per tier: themes, facts, episodes. Separate pools,
       so hundreds of raw turns can't crowd the few themes and facts out of the candidates.
    2. Theme expansion: the facts summarised by the top themes.
    3. Graph expansion: one hop along related / elaborates / example_of / caused_by links
       from the top facts, in either direction.
    4. Episode fallback: archived (already consolidated) episodes are searched too when the
       best match is weak, or the question is about a past conversation ("you told me...").
    5. Temporal filter: a time phrase ("in March", "last week") admits superseded facts that
       were true then, and boosts notes whose valid time overlaps the window.

Every note is scored with the §5 formula. A note reached by expansion also inherits a
share of its parent's score (spreading activation): parent_score · decay. Its final score
is the higher of the two, so expansion can surface a fact that shares no words with the
question (multi-hop) but never pushes down a direct match.

The result is up to `theme_k` themes (overview, first) followed by the top `top_k` facts
and episodes. With no themes or links in memory (before any sleep), this reduces to
Phase 2's ranking over facts and episodes.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from app.llm.embeddings import Embedder, aembed_query
from app.llm.tracing import Tracer
from app.memory.models import Candidate, LinkType, ScoredNote, Status, Tier
from app.memory.scoring import ScoreWeights, rank_candidates
from app.memory.store import MemoryStore
from app.memory.temporal import TimeWindow, parse_time_window

EXPANSION_LINKS = (
    LinkType.RELATED,
    LinkType.ELABORATES,
    LinkType.EXAMPLE_OF,
    LinkType.CAUSED_BY,
)

# Questions about the conversation itself: the answer may only be in the raw turns (facts
# hold what the *user* said; "what did you recommend" needs the assistant's words).
_CONVERSATIONAL_RE = re.compile(
    r"\b(?:you\s+(?:said|told|mentioned|recommended|suggested|gave|wrote|explained|listed|"
    r"shared|provided|helped)|(?:we|i)\s+(?:talk|discuss|chat|spoke|speak)\w*|"
    r"our\s+(?:previous\s+|last\s+|earlier\s+)?(?:conversation|chat|discussion)|"
    r"remind\s+me|previous\s+conversation|last\s+time\s+we)\b",
    re.IGNORECASE,
)


def is_conversational(query: str) -> bool:
    return bool(_CONVERSATIONAL_RE.search(query))


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 8  # facts + episodes returned
    candidate_pool: int = 40  # per tier, per channel (vector / keyword)
    theme_k: int = 2  # themes returned, ahead of the rest
    theme_pool: int = 10
    theme_decay: float = 0.85  # a member fact inherits theme_score · this
    link_decay: float = 0.7  # a linked note inherits fact_score · this
    expand_from: int = 8  # how many top facts are expanded along links
    temporal: bool = True
    temporal_weight: float = 0.15  # added when a note's valid time overlaps the window
    fallback_min_score: float = 0.45  # below this best score, search archived episodes too


@dataclass
class _Hit:
    scored: ScoredNote
    inherited: float = 0.0
    via: str = "search"

    @property
    def base(self) -> float:
        return max(self.scored.score, self.inherited)


def overlaps_window(note_tier: Tier, valid_from: datetime, valid_to: datetime | None,
                    window: TimeWindow) -> bool:  # fmt: skip
    if note_tier is Tier.EPISODE:  # an episode is a moment: when it was said
        return window.start <= valid_from < window.end
    return window.overlaps(valid_from, valid_to)


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
        config: RetrievalConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._weights = weights
        self._tau = recency_tau_days
        base = config or RetrievalConfig()
        self._cfg = RetrievalConfig(**{**base.__dict__, "top_k": top_k,
                                       "candidate_pool": candidate_pool})  # fmt: skip
        self._tracer = tracer or Tracer()
        self.last_window: TimeWindow | None = None

    def _rank(self, candidates: Sequence[Candidate], now: datetime) -> list[ScoredNote]:
        return rank_candidates(
            candidates,
            now=now,
            weights=self._weights,
            recency_tau_days=self._tau,
            top_k=len(candidates),
        )

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        now: datetime | None = None,
        as_of: datetime | None = None,
        tiers: tuple[Tier, ...] = (Tier.THEME, Tier.FACT, Tier.EPISODE),
    ) -> list[ScoredNote]:
        """`now` drives recency scoring and resolves relative time phrases. `as_of`, used
        only for simulated time in benchmark replay, hides notes recorded later. Live mode
        leaves it None so app/DB clock skew can't hide a note that was just written."""
        cfg = self._cfg
        k = top_k or cfg.top_k
        now = now or datetime.now(UTC)
        with self._tracer.observe("retrieve", as_type="retriever", input={"query": query}) as obs:
            window = parse_time_window(query, now) if cfg.temporal else None
            self.last_window = window
            during = (window.start, window.end) if window else None
            qvec = await aembed_query(self._embedder, query)

            # 1. Per-tier hybrid search
            searches = [
                self._store.hybrid_candidates(
                    qvec,
                    query,
                    tiers=[tier],
                    statuses=[Status.ACTIVE],
                    pool_size=cfg.theme_pool if tier is Tier.THEME else max(cfg.candidate_pool, k),
                    as_of=as_of,
                    true_during=during if tier is Tier.FACT else None,
                )
                for tier in tiers
            ]
            hits: dict[UUID, _Hit] = {}
            for found in await asyncio.gather(*searches):
                for s in self._rank(found, now):
                    hits[s.note.id] = _Hit(s)

            # 2. Themes -> member facts
            top_themes = self._top(hits, Tier.THEME, cfg.theme_k)
            if top_themes and Tier.FACT in tiers:
                links = await self._store.links_of(
                    [h.scored.note.id for h in top_themes], types=[LinkType.SUMMARIZES]
                )
                parents = {h.scored.note.id: h.base for h in top_themes}
                await self._expand(
                    hits,
                    ((r["dst_id"], parents[r["src_id"]]) for r in links if r["src_id"] in parents),
                    decay=cfg.theme_decay,
                    via="theme",
                    qvec=qvec, query=query, now=now, as_of=as_of, during=during,
                )  # fmt: skip

            # 3. One hop along fact links, from the best facts so far
            seeds = self._top(hits, Tier.FACT, cfg.expand_from)
            if seeds:
                seed_score = {h.scored.note.id: h.base for h in seeds}
                links = await self._store.links_of(list(seed_score), types=list(EXPANSION_LINKS))
                pairs = []
                for r in links:
                    for here, there in ((r["src_id"], r["dst_id"]), (r["dst_id"], r["src_id"])):
                        if here in seed_score:
                            pairs.append((there, seed_score[here]))
                await self._expand(hits, pairs, decay=cfg.link_decay, via="link",
                                   qvec=qvec, query=query, now=now, as_of=as_of,
                                   during=during)  # fmt: skip

            # 4. Fallback to archived episodes
            fallback = False
            if Tier.EPISODE in tiers:
                best = max(
                    (h.base for h in hits.values() if h.scored.note.tier is not Tier.THEME),
                    default=0.0,
                )
                fallback = best < cfg.fallback_min_score or is_conversational(query)
                if fallback:
                    found = await self._store.hybrid_candidates(
                        qvec, query, tiers=[Tier.EPISODE], statuses=[Status.ARCHIVED],
                        pool_size=max(cfg.candidate_pool, k), as_of=as_of,
                    )  # fmt: skip
                    for s in self._rank(found, now):
                        hits.setdefault(s.note.id, _Hit(s, via="fallback"))

            # 5. Final score: max(own, inherited) + temporal boost
            result: list[ScoredNote] = []
            for h in hits.values():
                n = h.scored.note
                boost = (
                    cfg.temporal_weight
                    if window and overlaps_window(n.tier, n.valid_from, n.valid_to, window)
                    else 0.0
                )
                via = h.via if h.inherited > h.scored.score else "search"
                if h.via == "fallback":
                    via = "fallback"
                result.append(h.scored.model_copy(update={"score": h.base + boost, "via": via}))
            result.sort(key=lambda s: (-s.score, -s.note.valid_from.timestamp(), str(s.note.id)))
            themes = [s for s in result if s.note.tier is Tier.THEME][: cfg.theme_k]
            rest = [s for s in result if s.note.tier is not Tier.THEME][:k]
            ranked = themes + rest

            obs.update(
                output=[
                    {
                        "id": str(s.note.id),
                        "score": round(s.score, 4),
                        "tier": s.note.tier,
                        "via": s.via,
                    }
                    for s in ranked
                ],  # fmt: skip
                metadata={
                    "candidates": len(hits),
                    "window": [window.start.isoformat(), window.end.isoformat(), window.phrase]
                    if window
                    else None,
                    "fallback": fallback,
                },
            )
            return ranked

    @staticmethod
    def _top(hits: dict[UUID, _Hit], tier: Tier, n: int) -> list[_Hit]:
        same = [h for h in hits.values() if h.scored.note.tier is tier]
        same.sort(key=lambda h: (-h.base, str(h.scored.note.id)))
        return same[:n]

    async def _expand(
        self,
        hits: dict[UUID, _Hit],
        pairs: Iterable[tuple[UUID, float]],
        *,
        decay: float,
        via: str,
        qvec: Sequence[float],
        query: str,
        now: datetime,
        as_of: datetime | None,
        during: tuple[datetime, datetime] | None,
    ) -> None:
        """Give each target the best parent_score · decay among its parents, scoring
        targets not seen yet (only those that pass the status / as-of / window rules)."""
        inherit: dict[UUID, float] = {}
        for target, parent in pairs:
            inherit[target] = max(inherit.get(target, 0.0), parent * decay)
        missing = [t for t in inherit if t not in hits]
        if missing:
            found = await self._store.score_notes(
                missing, qvec, query, statuses=[Status.ACTIVE], as_of=as_of, true_during=during
            )
            for s in self._rank(found, now):
                hits[s.note.id] = _Hit(s)
        for target, score in inherit.items():
            h = hits.get(target)
            if h is not None and h.scored.note.tier is not Tier.THEME and score > h.inherited:
                h.inherited = score
                h.via = via
