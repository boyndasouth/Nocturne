"""Hierarchical retriever (spec §5).

    1. Hybrid search (vector ∪ full-text) per tier: themes, facts, episodes. Separate pools,
       so hundreds of raw turns can't crowd the few themes and facts out of the candidates.
    2. Theme expansion: the facts summarised by the top themes.
    3. Graph expansion: one hop along related / elaborates / example_of / caused_by links
       from the top facts, in either direction.
    4. Fallback: what the Forgetter archived (consolidated episodes, unused facts) is
       searched too when the best match is weak, or the question is about a past
       conversation ("you told me...").
    5. Temporal filter: a time phrase ("in March", "last week") admits superseded facts that
       were true then, and boosts notes whose valid time overlaps the window.

Every note is scored with the §5 formula. A note reached by expansion also inherits a
share of its parent's score (spreading activation): parent_score · decay. Its final score
is the higher of the two, so expansion can surface a fact that shares no words with the
question (multi-hop) but never pushes down a direct match.

The result is up to `theme_k` themes (overview, first) followed by the top `top_k` facts
and episodes. With no themes or links in memory (before any sleep), this reduces to
Phase 2's ranking over facts and episodes.

Two options (Phase 4, off by default) for questions one ranked list serves badly:
- extra_facts: up to N more good-enough facts after the top k. Counting questions ("how
  many doctors did I visit?") need every matching fact, and facts are short.
- split_queries: a comparison ("the thermostat or the mesh network?", "between A and B")
  is also searched one side at a time, and each side's best notes are kept. Otherwise the
  side that matches the question's wording better fills every slot.
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
    fact_slots: int = 3  # top_k slots reserved for facts that score >= ratio x the best
    fact_slot_ratio: float = 0.6
    extra_facts: int = 0  # more facts after the top k (score >= fact_slot_ratio x the best)
    split_queries: bool = False  # search each side of "A or B" / "between A and B" too
    split_slots: int = 2  # best notes kept per side


# "A or B", "A vs B": the left side starts after the last comma / colon / "between".
_OR_RE = re.compile(r"(?:^|[,;:]\s*)(?P<a>[^,;:]+?)\s+(?:or|vs\.?|versus)\s+(?P<b>[^,;:?!.]+)", re.I)
_BETWEEN_RE = re.compile(r"\bbetween\s+(?P<a>.+?)\s+and\s+(?P<b>[^?!.]+)", re.I)
# "and" joins far more than comparisons, so only these question shapes are split on it.
_AND_TRIGGER_RE = re.compile(
    r"\b(?:both|total|combined|altogether|compare|comparison|difference|first|earlier|later)\b",
    re.I,
)
_AND_RE = re.compile(r"(?P<a>[^,;:]+?)\s+and\s+(?P<b>[^?!.]+)", re.I)
_EMPTY_SIDES = {
    "not", "no", "so", "else", "something", "anything", "nothing", "other", "others",
    "this", "that", "these", "those", "one", "ones", "them", "the",
}  # fmt: skip


def split_query(query: str) -> list[str]:
    """The two sides of a comparison question, or [] if it isn't one.

    "Which did I set up first, the smart thermostat or the mesh network system?"
    -> ["the smart thermostat", "the mesh network system"]"""
    for pattern in (_BETWEEN_RE, _OR_RE):
        m = pattern.search(query)
        if m:
            break
    else:
        m = _AND_RE.search(query) if _AND_TRIGGER_RE.search(query) else None
    if m is None:
        return []
    sides = [m.group("a").strip(" ,"), m.group("b").strip(" ,")]
    for side in sides:
        words = re.findall(r"[a-z0-9']+", side.lower())
        if not any(len(w) >= 3 and w not in _EMPTY_SIDES for w in words):
            return []
    if sides[0].lower() == sides[1].lower():
        return []
    return sides


def keep_per_side(
    top: Sequence[ScoredNote], sides: Sequence[Sequence[ScoredNote]], *, slots: int
) -> list[ScoredNote]:
    """`top` with each side's best `slots` notes guaranteed a place. A missing note
    replaces the lowest-ranked episode, else the lowest-ranked note no side needs."""
    out = list(top)
    keep: set[UUID] = set()
    for ranked in sides:
        for s in ranked[:slots]:
            keep.add(s.note.id)
            if any(o.note.id == s.note.id for o in out):
                continue
            victims = [i for i, o in enumerate(out) if o.note.id not in keep]
            episodes = [i for i in victims if out[i].note.tier is Tier.EPISODE]
            if episodes or victims:
                out[(episodes or victims)[-1]] = s.model_copy(update={"via": "split"})
            else:
                out.append(s.model_copy(update={"via": "split"}))
    out.sort(key=lambda s: (-s.score, -s.note.valid_from.timestamp(), str(s.note.id)))
    return out


def add_extra_facts(
    top: Sequence[ScoredNote], ranked: Sequence[ScoredNote], *, n: int, ratio: float
) -> list[ScoredNote]:
    """`top` plus up to n more facts from `ranked` that score >= ratio x the best, appended
    after the top (they're less relevant, and short, so they cost little context)."""
    out = list(top)
    if n <= 0 or not out:
        return out
    have = {s.note.id for s in out}
    floor = ratio * max(s.score for s in out)
    extra = [s for s in ranked if s.note.tier is Tier.FACT and s.note.id not in have
             and s.score >= floor]  # fmt: skip
    return out + extra[:n]


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


def reserve_fact_slots(
    ranked: Sequence[ScoredNote], *, k: int, slots: int, ratio: float
) -> list[ScoredNote]:
    """The top k of `ranked` (best first), except that up to `slots` of them go to facts
    when the top k has fewer: good-enough facts (score >= ratio x the best) from further
    down replace the lowest-ranked turns. Long raw turns can otherwise fill every slot,
    leaving out the facts sleep distilled from turns that weren't retrieved."""
    top = list(ranked[:k])
    have = sum(1 for s in top if s.note.tier is Tier.FACT)
    if have >= slots or not top:
        return top
    floor = ratio * top[0].score
    extra = [s for s in ranked[k:] if s.note.tier is Tier.FACT and s.score >= floor]
    extra = extra[: slots - have]
    for fact in extra:
        for i in range(len(top) - 1, -1, -1):  # drop the lowest-ranked turn
            if top[i].note.tier is Tier.EPISODE:
                top[i] = fact
                break
    top.sort(key=lambda s: (-s.score, -s.note.valid_from.timestamp(), str(s.note.id)))
    return top


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
            sides = split_query(query) if cfg.split_queries else []
            searches = [self._search(q, k=k, now=now, as_of=as_of, tiers=tiers, window=window)
                        for q in [query, *sides]]  # fmt: skip
            (result, candidates, fallback), *side_results = await asyncio.gather(*searches)

            themes = [s for s in result if s.note.tier is Tier.THEME][: cfg.theme_k]
            notes = [s for s in result if s.note.tier is not Tier.THEME]
            rest = reserve_fact_slots(notes, k=k, slots=cfg.fact_slots, ratio=cfg.fact_slot_ratio)
            if side_results:
                rest = keep_per_side(
                    rest,
                    [[s for s in r if s.note.tier is not Tier.THEME] for r, _, _ in side_results],
                    slots=cfg.split_slots,
                )
            rest = add_extra_facts(rest, notes, n=cfg.extra_facts, ratio=cfg.fact_slot_ratio)
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
                    "candidates": candidates,
                    "window": [window.start.isoformat(), window.end.isoformat(), window.phrase]
                    if window
                    else None,
                    "fallback": fallback,
                    "split": sides or None,
                },
            )
            return ranked

    async def _search(
        self,
        query: str,
        *,
        k: int,
        now: datetime,
        as_of: datetime | None,
        tiers: tuple[Tier, ...],
        window: TimeWindow | None,
    ) -> tuple[list[ScoredNote], int, bool]:
        """Steps 1-5 for one query: every candidate with its final score, best first; the
        candidate count; and whether the archive fallback ran."""
        cfg = self._cfg
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

        # 4. Fallback to what the Forgetter archived (turns and facts)
        fallback = False
        archived_tiers = [t for t in (Tier.FACT, Tier.EPISODE) if t in tiers]
        if archived_tiers:
            best = max(
                (h.base for h in hits.values() if h.scored.note.tier is not Tier.THEME),
                default=0.0,
            )
            fallback = best < cfg.fallback_min_score or is_conversational(query)
            if fallback:
                searches = [
                    self._store.hybrid_candidates(
                        qvec, query, tiers=[t], statuses=[Status.ARCHIVED],
                        pool_size=max(cfg.candidate_pool, k), as_of=as_of,
                    )
                    for t in archived_tiers
                ]  # fmt: skip
                for found in await asyncio.gather(*searches):
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
        return result, len(hits), fallback

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
