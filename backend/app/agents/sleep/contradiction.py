"""Contradiction Resolver (spec §3.2, step 3).

The LLM only answers "do these conflict?". Which fact wins is decided by valid time, not by
the model and not by processing order: the fact that became true later supersedes the other.
That keeps history correct when material arrives out of order (notes ingested from last
month, benchmark replays) and makes supersession cycles impossible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from app.agents.sleep.judge import format_groups, letter_to_ref
from app.agents.sleep.state import Conflict, Group
from app.llm.client import LLMClient
from app.llm.prompts import CONTRADICTION_SYSTEM
from app.llm.structured import Usage, chat_json


@dataclass(frozen=True)
class Timed:
    ref: str
    valid_from: datetime
    order: int  # tie-break: higher = learned later (existing facts < this cycle's candidates)


@dataclass(frozen=True)
class Supersession:
    winner: str
    loser: str
    valid_to: datetime  # when the loser stopped being true


def resolve(a: Timed, b: Timed) -> Supersession:
    """The later-valid fact wins. On equal valid time, the one learned later wins (a same-day
    correction), then the ref, so the result never depends on argument order."""

    def rank(t: Timed) -> tuple[datetime, int, str]:
        return (t.valid_from, t.order, t.ref)

    winner, loser = (a, b) if rank(a) > rank(b) else (b, a)
    return Supersession(winner=winner.ref, loser=loser.ref, valid_to=winner.valid_from)


class _Decision(BaseModel):
    group: int
    conflicts_with: list[str] = Field(default_factory=list)
    reason: str = ""


class ContradictionOutput(BaseModel):
    decisions: list[_Decision] = Field(default_factory=list)


def apply_decisions(groups: Sequence[Group], output: ContradictionOutput) -> list[Conflict]:
    out: list[Conflict] = []
    for d in output.decisions:
        if not 1 <= d.group <= len(groups):
            continue
        g = groups[d.group - 1]
        for letter in dict.fromkeys(d.conflicts_with):
            ref = letter_to_ref(g, letter)
            if ref is None:
                continue
            old = next(n for n in g.neighbors if n.ref == ref)
            out.append(
                Conflict(
                    new=f"c:{g.key}",
                    old=ref,
                    reason=d.reason[:300],
                    new_valid_from=g.valid_from,
                    old_valid_from=old.valid_from,
                )
            )
    return out


async def judge_conflicts(
    llm: LLMClient, groups: Sequence[Group], *, max_tokens: int
) -> tuple[list[Conflict], Usage]:
    if not groups:
        return [], Usage()
    output, usage = await chat_json(
        llm,
        [
            {"role": "system", "content": CONTRADICTION_SYSTEM},
            {"role": "user", "content": format_groups(groups, with_dates=True)},
        ],
        ContradictionOutput,
        trace_name="sleep.contradiction",
        max_tokens=max_tokens,
    )
    return apply_decisions(groups, output), usage
