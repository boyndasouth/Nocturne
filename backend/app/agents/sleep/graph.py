"""The sleep graph (spec §3.2), built with LangGraph.

    extract ⟲ ─► dedup_prepare ─► dedup_judge ⟲ ─► conflict_prepare ─► conflict_judge ⟲
      ─► link_prepare ─► link_judge ⟲ ─► theme_prepare ─► theme_judge ⟲ ─► apply ─► audit

⟲ = one LLM batch per step. LangGraph checkpoints after every step, so a cycle that fails
on a rate limit resumes at the batch that failed instead of re-spending quota on the ones that
succeeded. Batches are large (25 episodes per extraction call) to fit
free-tier request quotas; an extraction batch whose answer comes back cut off or unparseable
is split in two in the frozen plan and retried, so a big batch costs at most a few extra
calls.

Every step before `apply` only reads memory; `apply` writes it in one transaction.

The Linker and Abstractor plan against the memory as it *will* be once this cycle applies
(new facts in, superseded ones out), which `plan_apply` can compute before anything is
written. That's what lets their output join the same transaction.

The Forgetter has no LLM step: it runs inside `apply`, last, so it sees this cycle's facts
and theme memberships. The Auditor runs after the commit: it only reads, and its failures
are recorded, never raised, so it can't undo or fail a cycle that already succeeded.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.sleep.abstractor import (
    AbstractConfig,
    ExistingTheme,
    plan_themes,
    summarise_clusters,
    theme_content,
)
from app.agents.sleep.apply import ApplyPlan, StructurePlan, apply_plan, plan_apply
from app.agents.sleep.auditor import AuditItem, audit, sample_items
from app.agents.sleep.contradiction import judge_conflicts
from app.agents.sleep.dedup import DedupThresholds, judge_duplicates, plan_dedup
from app.agents.sleep.extractor import batch_episodes, extract_batch
from app.agents.sleep.forgetter import ForgetConfig
from app.agents.sleep.judge import chunks, cosine_matrix, neighbor_lists
from app.agents.sleep.linker import judge_links
from app.agents.sleep.state import (
    ClusterFact,
    Conflict,
    EpisodeIn,
    FactCandidate,
    Group,
    Link,
    SleepState,
    ThemePlan,
    ThemeText,
    cand_ref,
    note_ref,
    parse_ref,
)
from app.config import Settings
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder, aembed_documents
from app.llm.structured import InvalidJSONError, Usage
from app.llm.tracing import Tracer
from app.memory.cycles import CycleStore
from app.memory.store import MemoryStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SleepConfig:
    extract_batch_chars: int = 40_000
    extract_batch_episodes: int = 25
    extract_assistant_chars: int = 400
    extract_prompt: str = "v2"
    judge_batch_groups: int = 40
    neighbors_k: int = 4
    dedup: DedupThresholds = DedupThresholds()
    conflict_sim: float = 0.65
    enable_dedup: bool = True
    enable_contradiction: bool = True
    enable_linker: bool = True
    link_sim: float = 0.70
    enable_abstractor: bool = True
    abstract: AbstractConfig = AbstractConfig()
    theme_batch_clusters: int = 12
    theme_max_facts: int = 2000
    forget: ForgetConfig = ForgetConfig()
    enable_auditor: bool = True
    audit_sample: int = 8
    max_tokens: int = 8192
    reinforce_eta: float = 0.2

    @classmethod
    def from_settings(cls, s: Settings) -> SleepConfig:
        return cls(
            extract_batch_chars=s.extract_batch_chars,
            extract_batch_episodes=s.extract_batch_episodes,
            extract_assistant_chars=s.extract_assistant_chars,
            extract_prompt=s.extract_prompt,
            judge_batch_groups=s.judge_batch_groups,
            neighbors_k=s.neighbors_k,
            dedup=DedupThresholds(auto=s.dedup_auto_sim, judge=s.dedup_judge_sim),
            conflict_sim=s.conflict_sim,
            enable_dedup=s.enable_dedup,
            enable_contradiction=s.enable_contradiction,
            enable_linker=s.enable_linker,
            link_sim=s.link_sim,
            enable_abstractor=s.enable_abstractor,
            abstract=AbstractConfig(
                threshold=s.theme_sim,
                min_size=s.theme_min_facts,
                match_overlap=s.theme_match_overlap,
                max_per_cycle=s.theme_max_per_cycle,
                facts_per_theme=s.theme_facts_shown,
            ),
            theme_batch_clusters=s.theme_batch_clusters,
            theme_max_facts=s.theme_max_facts,
            forget=ForgetConfig(
                enabled=s.enable_forgetter,
                episode_theta=s.forget_episode_theta,
                fact_theta=s.forget_fact_theta,
                max_access=s.forget_max_access,
            ),
            enable_auditor=s.enable_auditor,
            audit_sample=s.audit_sample,
            max_tokens=s.sleep_max_tokens,
            reinforce_eta=s.reinforce_eta,
        )


@dataclass
class SleepDeps:
    store: MemoryStore
    cycles: CycleStore
    embedder: Embedder
    llm: LLMClient
    tracer: Tracer
    config: SleepConfig


def initial_state(cycle_id: UUID, clock: datetime, episodes: list[EpisodeIn]) -> SleepState:
    return {
        "cycle_id": str(cycle_id),
        "clock": clock.isoformat(),
        "episodes": [e.model_dump(mode="json") for e in episodes],
        "extract_cursor": 0,
        "candidates": [],
        "duplicates": {},
        "conflicts": [],
        "tokens": 0,
        "llm_calls": 0,
    }


def _usage(state: SleepState, usage: Usage) -> dict[str, int]:
    return {
        "tokens": state.get("tokens", 0) + usage.tokens,
        "llm_calls": state.get("llm_calls", 0) + usage.calls,
    }


def _candidates(state: SleepState) -> list[FactCandidate]:
    return [FactCandidate.model_validate(c) for c in state.get("candidates", [])]


Node = Callable[[SleepState], Awaitable[dict[str, Any]]]


def build_sleep_graph(
    deps: SleepDeps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    cfg = deps.config

    def traced(name: str, fn: Node) -> Node:
        async def run(state: SleepState) -> dict[str, Any]:
            await deps.cycles.heartbeat(UUID(state["cycle_id"]))
            with deps.tracer.observe(f"sleep.{name}", as_type="span") as obs:
                out = await fn(state)
                obs.update(output={k: v for k, v in out.items() if isinstance(v, int | str)})
                return out

        return run

    # --- 1. Extractor -------------------------------------------------------------------------

    async def extract(state: SleepState) -> dict[str, Any]:
        episodes = [EpisodeIn.model_validate(e) for e in state["episodes"]]
        # Freeze batch boundaries in the state on the first step, so a resume after a config
        # change still lines up with the candidates already extracted.
        plan: list[list[int]] = state.get("extract_plan") or [  # type: ignore[assignment]
            [episodes.index(e) for e in batch]
            for batch in batch_episodes(
                episodes,
                max_chars=cfg.extract_batch_chars,
                max_items=cfg.extract_batch_episodes,
                assistant_chars=cfg.extract_assistant_chars,
            )
        ]
        cursor = state.get("extract_cursor", 0)
        if cursor >= len(plan):
            return {"extract_plan": plan, "extract_cursor": cursor}
        batch = [episodes[i] for i in plan[cursor]]
        try:
            new, usage = await extract_batch(
                deps.llm,
                batch,
                cursor,
                max_tokens=cfg.max_tokens,
                assistant_chars=cfg.extract_assistant_chars,
                prompt=cfg.extract_prompt,
            )
        except InvalidJSONError as exc:
            if len(plan[cursor]) < 2:
                raise
            # Too much for one answer: halve this batch and retry. Later batches shift right,
            # but none of them has produced candidates yet, so candidate keys stay unique.
            log.warning("extract batch %d (%d episodes) split: %s", cursor, len(batch), exc)
            half = (len(plan[cursor]) + 1) // 2
            split = [plan[cursor][:half], plan[cursor][half:]]
            return {
                "extract_plan": [*plan[:cursor], *split, *plan[cursor + 1 :]],
                "extract_cursor": cursor,
                **_usage(state, exc.usage),
            }
        return {
            "extract_plan": plan,
            "extract_cursor": cursor + 1,
            "candidates": [*state.get("candidates", []), *(c.model_dump(mode="json") for c in new)],
            **_usage(state, usage),
        }

    def after_extract(state: SleepState) -> str:
        return (
            "extract" if state["extract_cursor"] < len(state["extract_plan"]) else "dedup_prepare"
        )

    # --- 2. Deduplicator ----------------------------------------------------------------------

    async def dedup_prepare(state: SleepState) -> dict[str, Any]:
        cands = _candidates(state)
        if not cfg.enable_dedup or not cands:
            return {"duplicates": {}, "dedup_groups": [], "dedup_cursor": 0}
        vecs = await aembed_documents(deps.embedder, [c.content for c in cands])
        existing = await deps.store.nearest_active_facts(
            vecs, k=cfg.neighbors_k, min_sim=cfg.dedup.judge
        )
        neighbors = neighbor_lists(
            cands, cosine_matrix(vecs), existing, min_sim=cfg.dedup.judge, k=cfg.neighbors_k
        )
        auto, groups = plan_dedup(cands, neighbors, cfg.dedup)
        return {
            "duplicates": auto,
            "dedup_groups": [g.model_dump(mode="json") for g in groups],
            "dedup_cursor": 0,
        }

    async def dedup_judge(state: SleepState) -> dict[str, Any]:
        groups = [Group.model_validate(g) for g in state["dedup_groups"]]
        batches = chunks(groups, cfg.judge_batch_groups)
        cursor = state["dedup_cursor"]
        found, usage = await judge_duplicates(deps.llm, batches[cursor], max_tokens=cfg.max_tokens)
        return {
            "duplicates": {**state["duplicates"], **found},
            "dedup_cursor": cursor + 1,
            **_usage(state, usage),
        }

    def has_more(
        groups_key: str, cursor_key: str, then: str, loop: str, size: int | None = None
    ) -> Callable[..., str]:
        def route(state: SleepState) -> str:
            n = len(chunks(state.get(groups_key, []), size or cfg.judge_batch_groups))  # type: ignore[arg-type]
            return loop if state.get(cursor_key, 0) < n else then

        return route

    async def view(state: SleepState) -> tuple[ApplyPlan, dict[UUID, datetime], set[str]]:
        """The apply plan for the decisions so far, the valid_from of every existing note it
        references, and the refs it supersedes: memory as it will be after this cycle."""
        cands = _candidates(state)
        conflicts = [Conflict.model_validate(c) for c in state.get("conflicts", [])]
        duplicates: dict[str, str] = state.get("duplicates", {})
        refs = [*duplicates.values(), *(c.new for c in conflicts), *(c.old for c in conflicts)]
        note_ids = list(dict.fromkeys(UUID(v) for k, v in map(parse_ref, refs) if k == "n"))
        notes = await deps.store.notes_by_ids(note_ids)
        if len(notes) != len(note_ids):
            raise RuntimeError("a fact referenced by this cycle no longer exists")
        valid_from = {n.id: n.valid_from for n in notes}
        plan = plan_apply(cands, duplicates, conflicts, valid_from)
        return plan, valid_from, {sp.loser for sp in plan.supersessions}

    # --- 3. Contradiction Resolver ------------------------------------------------------------

    async def conflict_prepare(state: SleepState) -> dict[str, Any]:
        cands = _candidates(state)
        survivors = [c.key not in state["duplicates"] for c in cands]
        if not cfg.enable_contradiction or not any(survivors):
            return {"conflict_groups": [], "conflict_cursor": 0}
        vecs = await aembed_documents(deps.embedder, [c.content for c in cands])
        existing = await deps.store.nearest_active_facts(
            [v for v, keep in zip(vecs, survivors, strict=True) if keep],
            k=cfg.neighbors_k,
            min_sim=cfg.conflict_sim,
        )
        it = iter(existing)
        existing_all = [next(it) if keep else [] for keep in survivors]
        neighbors = neighbor_lists(
            cands,
            cosine_matrix(vecs),
            existing_all,
            min_sim=cfg.conflict_sim,
            k=cfg.neighbors_k,
            eligible=survivors,
        )
        groups = [
            Group(key=c.key, content=c.content, valid_from=c.valid_from, neighbors=n)
            for c, n in zip(cands, neighbors, strict=True)
            if n
        ]
        return {
            "conflict_groups": [g.model_dump(mode="json") for g in groups],
            "conflict_cursor": 0,
        }

    async def conflict_judge(state: SleepState) -> dict[str, Any]:
        groups = [Group.model_validate(g) for g in state["conflict_groups"]]
        batches = chunks(groups, cfg.judge_batch_groups)
        cursor = state["conflict_cursor"]
        found, usage = await judge_conflicts(deps.llm, batches[cursor], max_tokens=cfg.max_tokens)
        return {
            "conflicts": [*state.get("conflicts", []), *(c.model_dump(mode="json") for c in found)],
            "conflict_cursor": cursor + 1,
            **_usage(state, usage),
        }

    # --- 4. Linker ----------------------------------------------------------------------------

    async def link_prepare(state: SleepState) -> dict[str, Any]:
        empty = {"link_groups": [], "link_cursor": 0, "links": []}
        if not cfg.enable_linker:
            return empty
        cands = _candidates(state)
        plan, _, losers = await view(state)
        keep = {p.key for p in plan.inserts if cand_ref(p.key) not in losers}
        eligible = [c.key in keep for c in cands]
        if not any(eligible):
            return empty
        vecs = await aembed_documents(deps.embedder, [c.content for c in cands])
        probes = [v for v, e in zip(vecs, eligible, strict=True) if e]
        found = await deps.store.nearest_active_facts(
            probes, k=cfg.neighbors_k + len(losers), min_sim=cfg.link_sim
        )
        it = iter(found)
        existing = [
            [n for n in next(it) if note_ref(n.note.id) not in losers] if e else []
            for e in eligible
        ]
        neighbors = neighbor_lists(
            cands,
            cosine_matrix(vecs),
            existing,
            min_sim=cfg.link_sim,
            k=cfg.neighbors_k,
            eligible=eligible,
        )
        groups = [
            Group(key=c.key, content=c.content, valid_from=c.valid_from, neighbors=n)
            for c, n in zip(cands, neighbors, strict=True)
            if n
        ]
        return {**empty, "link_groups": [g.model_dump(mode="json") for g in groups]}

    async def link_judge(state: SleepState) -> dict[str, Any]:
        groups = [Group.model_validate(g) for g in state["link_groups"]]
        batches = chunks(groups, cfg.judge_batch_groups)
        cursor = state["link_cursor"]
        found, usage = await judge_links(deps.llm, batches[cursor], max_tokens=cfg.max_tokens)
        return {
            "links": [*state.get("links", []), *(lk.model_dump(mode="json") for lk in found)],
            "link_cursor": cursor + 1,
            **_usage(state, usage),
        }

    # --- 5. Abstractor ------------------------------------------------------------------------

    async def theme_prepare(state: SleepState) -> dict[str, Any]:
        empty = {"theme_plans": [], "theme_cursor": 0, "theme_texts": {}, "themes_retired": []}
        if not cfg.enable_abstractor:
            return empty
        plan, _, losers = await view(state)
        existing = await deps.store.active_fact_vectors(limit=cfg.theme_max_facts)
        facts: list[ClusterFact] = []
        vectors: list[list[float]] = []
        for note, vec in existing:
            if note_ref(note.id) not in losers:
                facts.append(
                    ClusterFact(
                        ref=note_ref(note.id), content=note.content, valid_from=note.valid_from
                    )
                )
                vectors.append(vec)
        new = [p for p in plan.inserts if cand_ref(p.key) not in losers]
        vectors.extend(await aembed_documents(deps.embedder, [p.content for p in new]))
        facts.extend(
            ClusterFact(ref=cand_ref(p.key), content=p.content, valid_from=p.valid_from)
            for p in new
        )
        themes = [
            ExistingTheme(id=t.id, content=t.content, members=tuple(members))
            for t, members in await deps.store.active_themes()
        ]
        plans, retired = plan_themes(facts, vectors, themes, cfg.abstract)
        return {
            **empty,
            "theme_plans": [p.model_dump(mode="json") for p in plans],
            "themes_retired": [str(t) for t in retired],
        }

    async def theme_judge(state: SleepState) -> dict[str, Any]:
        plans = [ThemePlan.model_validate(p) for p in state["theme_plans"]]
        batches = chunks(plans, cfg.theme_batch_clusters)
        cursor = state["theme_cursor"]
        texts, usage = await summarise_clusters(
            deps.llm, batches[cursor], max_tokens=cfg.max_tokens
        )
        return {
            "theme_texts": {
                **state.get("theme_texts", {}),
                **{k: v.model_dump(mode="json") for k, v in texts.items()},
            },
            "theme_cursor": cursor + 1,
            **_usage(state, usage),
        }

    # --- Apply (single transaction) -----------------------------------------------------------

    async def apply(state: SleepState) -> dict[str, Any]:
        plan, valid_from, _ = await view(state)
        texts = {k: ThemeText.model_validate(v) for k, v in state.get("theme_texts", {}).items()}
        structure = StructurePlan(
            links=[Link.model_validate(lk) for lk in state.get("links", [])],
            themes=[
                (tp, texts[tp.key])
                for tp in (ThemePlan.model_validate(p) for p in state.get("theme_plans", []))
                if tp.key in texts
            ],
            retired_themes=[UUID(t) for t in state.get("themes_retired", [])],
        )
        result = await apply_plan(
            plan,
            store=deps.store,
            cycles=deps.cycles,
            embedder=deps.embedder,
            cycle_id=UUID(state["cycle_id"]),
            clock=datetime.fromisoformat(state["clock"]),
            episode_ids=[UUID(e["id"]) for e in state["episodes"]],
            note_valid_from=valid_from,
            reinforce_eta=cfg.reinforce_eta,
            tokens_used=state.get("tokens", 0),
            llm_calls=state.get("llm_calls", 0),
            structure=structure,
            forget=cfg.forget,
        )
        return {
            "result": {
                "facts_created": result.facts_created,
                "merges": result.merges,
                "supersessions": result.supersessions,
                "links_created": result.links_created,
                "themes_created": result.themes_created,
                "themes_updated": result.themes_updated,
                "themes_archived": result.themes_archived,
                "episodes_archived": result.episodes_archived,
                "facts_archived": result.facts_archived,
            }
        }

    # --- 7. Auditor (after the commit; observe only) ------------------------------------------

    async def audit_node(state: SleepState) -> dict[str, Any]:
        if not cfg.enable_auditor or cfg.audit_sample <= 0:
            return {}
        cycle_id = UUID(state["cycle_id"])
        try:
            items = await _audit_items(state)
            # Seeded by the decisions, not the cycle id, so an eval rerun draws the same
            # sample and replays from the LLM cache.
            seed = "\n".join(sorted(f"{it.kind}:{it.text}" for it in items))
            picked = sample_items(items, n=cfg.audit_sample, seed=seed)
            if not picked:
                return {}
            result, usage = await audit(deps.llm, picked, max_tokens=cfg.max_tokens)
            await deps.cycles.record_audit(
                cycle_id,
                score=result.score,
                items=len(result.verdicts),
                detail={"by_kind": result.by_kind(), "verdicts": result.verdicts},
                tokens=usage.tokens,
                llm_calls=usage.calls,
            )
            return {"audit": {"score": result.score, "by_kind": result.by_kind()}}
        except Exception as exc:  # the cycle already committed; never fail it from here
            log.warning("audit of cycle %s skipped: %s", cycle_id, exc)
            try:
                await deps.cycles.record_audit(
                    cycle_id,
                    score=None,
                    items=0,
                    detail={"error": f"{type(exc).__name__}: {exc}"[:500]},
                )
            except Exception:
                log.exception("could not record the audit error for cycle %s", cycle_id)
            return {}

    async def _audit_items(state: SleepState) -> list[AuditItem]:
        plan, _, _ = await view(state)
        cands = {c.key: c for c in _candidates(state)}
        links = [Link.model_validate(lk) for lk in state.get("links", [])]
        texts = {k: ThemeText.model_validate(v) for k, v in state.get("theme_texts", {}).items()}
        themes = [
            (tp, texts[tp.key])
            for tp in (ThemePlan.model_validate(p) for p in state.get("theme_plans", []))
            if tp.key in texts
        ]
        refs = [lk.src for lk in links] + [lk.dst for lk in links]
        refs += [r for sp in plan.supersessions for r in (sp.winner, sp.loser)]
        ids = {UUID(v) for k, v in map(parse_ref, refs) if k == "n"}
        ids |= {m.note_id for m in plan.merges}
        notes = {n.id: n for n in await deps.store.notes_by_ids(list(ids))}

        def text(ref: str) -> str:
            kind, value = parse_ref(ref)
            if kind == "c":
                return cands[value].content
            note = notes.get(UUID(value))
            return note.content if note else "(missing)"

        def since(ref: str) -> str:
            kind, value = parse_ref(ref)
            vf = cands[value].valid_from if kind == "c" else notes[UUID(value)].valid_from
            return f"{vf:%Y-%m-%d}"

        items: list[AuditItem] = []
        for m in plan.merges:
            for key in m.candidate_keys:
                items.append(
                    AuditItem(
                        "merge", f"MERGE\nA: {cands[key].content}\nB: {notes[m.note_id].content}"
                    )
                )
        for ins in plan.inserts:
            for key in ins.folded_keys:
                items.append(
                    AuditItem("merge", f"MERGE\nA: {cands[key].content}\nB: {ins.content}")
                )
        for sp in plan.supersessions:
            items.append(
                AuditItem(
                    "supersede",
                    f"SUPERSEDE\nNEWER (since {since(sp.winner)}): {text(sp.winner)}\n"
                    f"OLDER (since {since(sp.loser)}): {text(sp.loser)}\nReason given: {sp.reason}",
                )
            )
        for lk in links:
            items.append(
                AuditItem("link", f"LINK\nA: {text(lk.src)}\nType: {lk.type}\nB: {text(lk.dst)}")
            )
        for tp, tt in themes:
            facts = "\n".join(f"- {f.content}" for f in tp.facts[:10])
            items.append(AuditItem("theme", f"THEME\n{theme_content(tt)}\nFacts:\n{facts}"))
        return items

    g = StateGraph(SleepState)
    for name, fn in [
        ("extract", extract),
        ("dedup_prepare", dedup_prepare),
        ("dedup_judge", dedup_judge),
        ("conflict_prepare", conflict_prepare),
        ("conflict_judge", conflict_judge),
        ("link_prepare", link_prepare),
        ("link_judge", link_judge),
        ("theme_prepare", theme_prepare),
        ("theme_judge", theme_judge),
        ("apply", apply),
        ("audit", audit_node),
    ]:
        g.add_node(name, traced(name, fn))
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", after_extract, ["extract", "dedup_prepare"])
    dedup_route = has_more("dedup_groups", "dedup_cursor", "conflict_prepare", "dedup_judge")
    g.add_conditional_edges("dedup_prepare", dedup_route, ["dedup_judge", "conflict_prepare"])
    g.add_conditional_edges("dedup_judge", dedup_route, ["dedup_judge", "conflict_prepare"])
    conflict_route = has_more(
        "conflict_groups", "conflict_cursor", "link_prepare", "conflict_judge"
    )
    g.add_conditional_edges("conflict_prepare", conflict_route, ["conflict_judge", "link_prepare"])
    g.add_conditional_edges("conflict_judge", conflict_route, ["conflict_judge", "link_prepare"])
    link_route = has_more("link_groups", "link_cursor", "theme_prepare", "link_judge")
    g.add_conditional_edges("link_prepare", link_route, ["link_judge", "theme_prepare"])
    g.add_conditional_edges("link_judge", link_route, ["link_judge", "theme_prepare"])
    theme_route = has_more(
        "theme_plans", "theme_cursor", "apply", "theme_judge", size=cfg.theme_batch_clusters
    )
    g.add_conditional_edges("theme_prepare", theme_route, ["theme_judge", "apply"])
    g.add_conditional_edges("theme_judge", theme_route, ["theme_judge", "apply"])
    g.add_edge("apply", "audit")
    g.add_edge("audit", END)
    return g.compile(checkpointer=checkpointer)
