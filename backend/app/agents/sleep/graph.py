"""The sleep graph (spec §3.2), built with LangGraph.

    extract ⟲ ─► dedup_prepare ─► dedup_judge ⟲ ─► conflict_prepare ─► conflict_judge ⟲ ─► apply

⟲ = one LLM batch per step. LangGraph checkpoints after every step, so a cycle that fails
on a rate limit resumes at the batch that failed instead of re-spending quota on the ones that
succeeded. Every step before `apply` only reads memory; `apply` writes it in one transaction.

Phase 3 adds Linker, Abstractor, Forgetter, and Auditor between conflict_judge and the end.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.sleep.apply import apply_plan, plan_apply
from app.agents.sleep.contradiction import judge_conflicts
from app.agents.sleep.dedup import DedupThresholds, judge_duplicates, plan_dedup
from app.agents.sleep.extractor import batch_episodes, extract_batch
from app.agents.sleep.judge import chunks, cosine_matrix, neighbor_lists
from app.agents.sleep.state import (
    Conflict,
    EpisodeIn,
    FactCandidate,
    Group,
    SleepState,
    parse_ref,
)
from app.config import Settings
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder, aembed_documents
from app.llm.structured import Usage
from app.llm.tracing import Tracer
from app.memory.cycles import CycleStore
from app.memory.store import MemoryStore


@dataclass(frozen=True)
class SleepConfig:
    extract_batch_chars: int = 12_000
    extract_batch_episodes: int = 25
    judge_batch_groups: int = 12
    neighbors_k: int = 4
    dedup: DedupThresholds = DedupThresholds()
    conflict_sim: float = 0.65
    enable_dedup: bool = True
    enable_contradiction: bool = True
    max_tokens: int = 8192
    reinforce_eta: float = 0.2

    @classmethod
    def from_settings(cls, s: Settings) -> SleepConfig:
        return cls(
            extract_batch_chars=s.extract_batch_chars,
            extract_batch_episodes=s.extract_batch_episodes,
            judge_batch_groups=s.judge_batch_groups,
            neighbors_k=s.neighbors_k,
            dedup=DedupThresholds(auto=s.dedup_auto_sim, judge=s.dedup_judge_sim),
            conflict_sim=s.conflict_sim,
            enable_dedup=s.enable_dedup,
            enable_contradiction=s.enable_contradiction,
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
                episodes, max_chars=cfg.extract_batch_chars, max_items=cfg.extract_batch_episodes
            )
        ]
        cursor = state.get("extract_cursor", 0)
        if cursor >= len(plan):
            return {"extract_plan": plan, "extract_cursor": cursor}
        batch = [episodes[i] for i in plan[cursor]]
        new, usage = await extract_batch(deps.llm, batch, cursor, max_tokens=cfg.max_tokens)
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

    def has_more(groups_key: str, cursor_key: str, then: str, loop: str) -> Callable[..., str]:
        def route(state: SleepState) -> str:
            n = len(chunks(state.get(groups_key, []), cfg.judge_batch_groups))  # type: ignore[arg-type]
            return loop if state.get(cursor_key, 0) < n else then

        return route

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

    # --- Apply (single transaction) -----------------------------------------------------------

    async def apply(state: SleepState) -> dict[str, Any]:
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
        )
        return {
            "result": {
                "facts_created": result.facts_created,
                "merges": result.merges,
                "supersessions": result.supersessions,
            }
        }

    g = StateGraph(SleepState)
    for name, fn in [
        ("extract", extract),
        ("dedup_prepare", dedup_prepare),
        ("dedup_judge", dedup_judge),
        ("conflict_prepare", conflict_prepare),
        ("conflict_judge", conflict_judge),
        ("apply", apply),
    ]:
        g.add_node(name, traced(name, fn))
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", after_extract, ["extract", "dedup_prepare"])
    dedup_route = has_more("dedup_groups", "dedup_cursor", "conflict_prepare", "dedup_judge")
    g.add_conditional_edges("dedup_prepare", dedup_route, ["dedup_judge", "conflict_prepare"])
    g.add_conditional_edges("dedup_judge", dedup_route, ["dedup_judge", "conflict_prepare"])
    conflict_route = has_more("conflict_groups", "conflict_cursor", "apply", "conflict_judge")
    g.add_conditional_edges("conflict_prepare", conflict_route, ["conflict_judge", "apply"])
    g.add_conditional_edges("conflict_judge", conflict_route, ["conflict_judge", "apply"])
    g.add_edge("apply", END)
    return g.compile(checkpointer=checkpointer)
