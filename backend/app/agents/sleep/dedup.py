"""Deduplicator (spec §3.2, step 2): merge candidates that restate a known fact.

Three similarity bands (cosine, bge-small):
    sim ≥ auto        -> merge without asking (near-identical wording)
    judge ≤ sim < auto -> ask the LLM; it says "duplicate" only for same information
    sim < judge        -> distinct
The auto band is deliberately narrow: "lives in Austin" and "lives in Dallas" can score
~0.9, so a looser auto-merge would silently swallow contradictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from app.agents.sleep.judge import format_groups, letter_to_ref
from app.agents.sleep.state import FactCandidate, Group, NeighborRef
from app.llm.client import LLMClient
from app.llm.prompts import DEDUP_SYSTEM
from app.llm.structured import Usage, chat_json


class Band(StrEnum):
    AUTO = "auto"
    JUDGE = "judge"
    DISTINCT = "distinct"


@dataclass(frozen=True)
class DedupThresholds:
    auto: float = 0.97
    judge: float = 0.82

    def __post_init__(self) -> None:
        if not 0.0 <= self.judge <= self.auto <= 1.0:
            raise ValueError("need 0 ≤ judge ≤ auto ≤ 1")


def classify(sim: float, t: DedupThresholds) -> Band:
    if sim >= t.auto:
        return Band.AUTO
    if sim >= t.judge:
        return Band.JUDGE
    return Band.DISTINCT


def plan_dedup(
    candidates: Sequence[FactCandidate],
    neighbors: Sequence[Sequence[NeighborRef]],
    t: DedupThresholds,
) -> tuple[dict[str, str], list[Group]]:
    """Split candidates into auto-merges {key: ref} and groups for the LLM judge.
    `neighbors[i]` must be sorted by similarity, highest first."""
    auto: dict[str, str] = {}
    groups: list[Group] = []
    for cand, neigh in zip(candidates, neighbors, strict=True):
        if not neigh:
            continue
        band = classify(neigh[0].sim, t)
        if band is Band.AUTO:
            auto[cand.key] = neigh[0].ref
        elif band is Band.JUDGE:
            groups.append(
                Group(
                    key=cand.key,
                    content=cand.content,
                    valid_from=cand.valid_from,
                    neighbors=[n for n in neigh if classify(n.sim, t) is not Band.DISTINCT],
                )
            )
    return auto, groups


class _Decision(BaseModel):
    group: int
    duplicate_of: str | None = None


class DedupOutput(BaseModel):
    decisions: list[_Decision] = Field(default_factory=list)


def apply_decisions(groups: Sequence[Group], output: DedupOutput) -> dict[str, str]:
    """Missing or invalid answers mean "not a duplicate": keeping an extra fact is cheaper
    than losing one."""
    out: dict[str, str] = {}
    for d in output.decisions:
        if 1 <= d.group <= len(groups):
            g = groups[d.group - 1]
            ref = letter_to_ref(g, d.duplicate_of)
            if ref is not None:
                out[g.key] = ref
    return out


async def judge_duplicates(
    llm: LLMClient, groups: Sequence[Group], *, max_tokens: int
) -> tuple[dict[str, str], Usage]:
    if not groups:
        return {}, Usage()
    output, usage = await chat_json(
        llm,
        [
            {"role": "system", "content": DEDUP_SYSTEM},
            {"role": "user", "content": format_groups(groups, with_dates=False)},
        ],
        DedupOutput,
        trace_name="sleep.dedup",
        max_tokens=max_tokens,
    )
    return apply_decisions(groups, output), usage
