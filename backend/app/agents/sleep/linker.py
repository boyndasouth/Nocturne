"""Linker (spec §3.2, step 4): typed links between new facts and the facts near them.

Candidates are this cycle's surviving facts (not duplicates, not superseded). Each one is
grouped with its nearest active facts, existing and earlier in this cycle, and an LLM labels
each pair, in batches. The LLM reads types as "NEW <type> EXISTING"; reversed labels
("elaborated_by", "causes", ...) are stored by swapping ends, so every link reads
src <type> dst:

    elaborates   src adds detail to dst
    example_of   src is a specific case of dst
    caused_by    src happened because of dst
    related      same specific subject (stored once, src = the newer fact)

`supersedes` stays with the Contradiction Resolver; pairs it already linked are skipped.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.agents.sleep.judge import format_groups, letter_to_ref
from app.agents.sleep.state import Group, Link
from app.llm.client import LLMClient
from app.llm.prompts import LINKER_SYSTEM
from app.llm.structured import Usage, chat_json
from app.memory.models import LinkType

# label -> (stored type, swap src/dst)
LABELS: dict[str, tuple[LinkType, bool]] = {
    "related": (LinkType.RELATED, False),
    "elaborates": (LinkType.ELABORATES, False),
    "elaborated_by": (LinkType.ELABORATES, True),
    "example_of": (LinkType.EXAMPLE_OF, False),
    "has_example": (LinkType.EXAMPLE_OF, True),
    "caused_by": (LinkType.CAUSED_BY, False),
    "causes": (LinkType.CAUSED_BY, True),
}


class _LinkOut(BaseModel):
    to: str
    type: str


class _Decision(BaseModel):
    group: int
    links: list[_LinkOut] = Field(default_factory=list)


class LinkerOutput(BaseModel):
    decisions: list[_Decision] = Field(default_factory=list)


def normalise_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def apply_decisions(groups: Sequence[Group], output: LinkerOutput) -> list[Link]:
    """Keep only well-formed links: a real group, a letter that names one of its neighbours,
    a known label. At most one link per pair; the first label wins."""
    out: list[Link] = []
    seen: set[frozenset[str]] = set()
    for d in output.decisions:
        if not 1 <= d.group <= len(groups):
            continue
        g = groups[d.group - 1]
        new = f"c:{g.key}"
        for item in d.links:
            ref = letter_to_ref(g, item.to)
            label = LABELS.get(normalise_label(item.type))
            if ref is None or label is None:
                continue
            pair = frozenset((new, ref))
            if pair in seen:
                continue
            seen.add(pair)
            link_type, swap = label
            src, dst = (ref, new) if swap else (new, ref)
            out.append(Link(src=src, dst=dst, type=link_type.value))
    return out


async def judge_links(
    llm: LLMClient, groups: Sequence[Group], *, max_tokens: int
) -> tuple[list[Link], Usage]:
    if not groups:
        return [], Usage()
    output, usage = await chat_json(
        llm,
        [
            {"role": "system", "content": LINKER_SYSTEM},
            {"role": "user", "content": format_groups(groups, with_dates=False)},
        ],
        LinkerOutput,
        trace_name="sleep.linker",
        max_tokens=max_tokens,
    )
    return apply_decisions(groups, output), usage
