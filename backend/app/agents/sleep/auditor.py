"""Auditor (spec §3.2 step 7): LLM-as-judge on a sample of each cycle's decisions.

After a cycle applies, up to `sample` of its merges, supersessions, links and themes are
drawn (round-robin across kinds, so a cycle with 40 links and 2 supersessions still gets its
supersessions checked; seeded by the cycle id, so a rerun audits the same items) and judged
in one LLM call. Verdicts score good = 1, questionable = 0.5, wrong = 0. The cycle row keeps
the mean and every verdict; the trace keeps them too (Langfuse, when configured).

The audit observes; it never changes memory, and it can't fail a cycle: by the time it runs
the cycle has committed, so an LLM error (a quota, bad JSON) is recorded on the cycle row
and swallowed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.client import LLMClient
from app.llm.prompts import AUDITOR_SYSTEM
from app.llm.structured import Usage, chat_json

KINDS = ("merge", "supersede", "link", "theme")
SCORES = {"good": 1.0, "questionable": 0.5, "wrong": 0.0}


@dataclass(frozen=True)
class AuditItem:
    kind: str  # one of KINDS
    text: str  # what the judge sees


def sample_items(items: Sequence[AuditItem], *, n: int, seed: str) -> list[AuditItem]:
    by_kind: dict[str, list[AuditItem]] = {k: [] for k in KINDS}
    for it in items:
        by_kind[it.kind].append(it)
    rng = random.Random(seed)
    for group in by_kind.values():
        rng.shuffle(group)
    out: list[AuditItem] = []
    while len(out) < n and any(by_kind.values()):
        for k in KINDS:
            if by_kind[k] and len(out) < n:
                out.append(by_kind[k].pop())
    return out


def format_items(items: Sequence[AuditItem]) -> str:
    return "\n\n".join(f"Item {i}\n{it.text}" for i, it in enumerate(items, start=1))


class _Verdict(BaseModel):
    item: int
    verdict: str
    note: str = ""


class AuditorOutput(BaseModel):
    verdicts: list[_Verdict] = Field(default_factory=list)


@dataclass(frozen=True)
class AuditResult:
    score: float | None  # mean over judged items; None if nothing could be judged
    verdicts: list[dict]  # {"kind", "verdict", "note", "item"}

    def by_kind(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for v in self.verdicts:
            out.setdefault(v["kind"], []).append(SCORES[v["verdict"]])
        return {k: round(sum(s) / len(s), 3) for k, s in out.items()}


def score_output(items: Sequence[AuditItem], output: AuditorOutput) -> AuditResult:
    seen: set[int] = set()
    verdicts = []
    for v in output.verdicts:
        label = v.verdict.strip().lower()
        if not 1 <= v.item <= len(items) or v.item in seen or label not in SCORES:
            continue
        seen.add(v.item)
        it = items[v.item - 1]
        verdicts.append({"item": v.item, "kind": it.kind, "verdict": label, "note": v.note[:200]})
    verdicts.sort(key=lambda v: v["item"])
    if not verdicts:
        return AuditResult(score=None, verdicts=[])
    mean = sum(SCORES[v["verdict"]] for v in verdicts) / len(verdicts)
    return AuditResult(score=round(mean, 4), verdicts=verdicts)


async def audit(
    llm: LLMClient, items: Sequence[AuditItem], *, max_tokens: int
) -> tuple[AuditResult, Usage]:
    if not items:
        return AuditResult(score=None, verdicts=[]), Usage()
    output, usage = await chat_json(
        llm,
        [
            {"role": "system", "content": AUDITOR_SYSTEM},
            {"role": "user", "content": format_items(items)},
        ],
        AuditorOutput,
        trace_name="sleep.auditor",
        max_tokens=max_tokens,
    )
    return score_output(items, output), usage
