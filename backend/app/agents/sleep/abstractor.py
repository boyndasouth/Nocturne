"""Abstractor (spec §3.2, step 5): cluster active facts and keep a `theme` note per cluster.

Each cycle re-clusters the facts that will be active once the cycle applies (existing
active facts, minus the ones it supersedes, plus its new facts), then reconciles the
clusters with the themes that already exist:

    cluster matches a theme, same members   -> nothing to do (no LLM call)
                                               ("same" includes: none of them superseded)
    cluster matches a theme, members differ -> re-summarise, edit the theme (versioned)
    cluster matches no theme                -> summarise into a new theme
    theme matches no cluster                -> kept, unless fewer than `min_size` of its
                                               members are still active: then archived

A cluster matches a theme when they share at least `match_overlap` of the smaller one's
members (overlap coefficient, not Jaccard, so a theme that grew from 3 facts to 8 is
still the same theme). Matching is one-to-one, best overlap first.

Clustering is deterministic and needs no LLM; the LLM only writes summaries, a few
clusters per call, and at most `max_per_cycle` per cycle (the rest wait for later cycles).
Themes link to their members with `summarizes` (theme -> fact).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, Field

from app.agents.sleep.cluster import average_linkage
from app.agents.sleep.state import ClusterFact, ThemePlan, ThemeText, note_ref
from app.llm.client import LLMClient
from app.llm.prompts import ABSTRACTOR_SYSTEM, CLUSTER_TEMPLATE
from app.llm.structured import Usage, chat_json

MAX_TITLE_CHARS = 80
MAX_CONTENT_CHARS = 600


@dataclass(frozen=True)
class AbstractConfig:
    threshold: float = 0.72  # average cosine similarity within a cluster
    min_size: int = 3
    match_overlap: float = 0.5
    max_per_cycle: int = 12
    facts_per_theme: int = 25  # how many members the LLM sees per cluster


@dataclass(frozen=True)
class ExistingTheme:
    id: UUID
    content: str
    members: tuple[UUID, ...]  # every fact it summarises, active or not


def overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def plan_themes(
    facts: Sequence[ClusterFact],
    vectors: Sequence[Sequence[float]],
    themes: Sequence[ExistingTheme],
    cfg: AbstractConfig,
) -> tuple[list[ThemePlan], list[UUID]]:
    """Pure: returns (clusters to summarise, theme ids to archive). `facts` are the facts
    active after this cycle; `vectors` their embeddings, in the same order."""
    clusters = [
        [facts[i] for i in idx]
        for idx in average_linkage(vectors, threshold=cfg.threshold, min_size=cfg.min_size)
    ]
    active = {f.ref for f in facts}
    live: dict[UUID, set[str]] = {t.id: {note_ref(m) for m in t.members} & active for t in themes}

    # One-to-one matching, best overlap first; ties to the bigger intersection, then ids.
    pairs = sorted(
        (
            (overlap(cset, live[t.id]), len(cset & live[t.id]), ci, str(t.id))
            for ci, c in enumerate(clusters)
            for cset in [{f.ref for f in c}]
            for t in themes
        ),
        key=lambda p: (-p[0], -p[1], p[2], p[3]),
    )
    by_id = {str(t.id): t for t in themes}
    matched_cluster: dict[int, ExistingTheme] = {}
    matched_theme: set[str] = set()
    for score, _, ci, tid in pairs:
        if score < cfg.match_overlap:
            break
        if ci in matched_cluster or tid in matched_theme:
            continue
        matched_cluster[ci] = by_id[tid]
        matched_theme.add(tid)

    plans: list[ThemePlan] = []
    for ci, c in enumerate(clusters):
        refs = [f.ref for f in c]
        theme = matched_cluster.get(ci)
        # Unchanged only if every member is still active and none joined or left; a member
        # that was superseded means the summary may state something no longer true.
        if theme is not None and set(refs) == {note_ref(m) for m in theme.members}:
            continue
        shown = sorted(c, key=lambda f: (f.valid_from, f.ref))[-cfg.facts_per_theme :]
        plans.append(
            ThemePlan(
                key=f"t{ci}",
                theme_id=theme.id if theme else None,
                members=refs,
                facts=shown,
                valid_from=min(f.valid_from for f in c),
                previous=theme.content if theme else "",
            )
        )
    plans.sort(key=lambda p: (-len(p.members), p.key))
    plans = plans[: cfg.max_per_cycle]

    retired = [
        t.id for t in themes if str(t.id) not in matched_theme and len(live[t.id]) < cfg.min_size
    ]
    return plans, retired


def theme_content(text: ThemeText) -> str:
    title = text.title.strip().rstrip(".:")[:MAX_TITLE_CHARS]
    summary = " ".join(text.summary.split())
    return f"{title}: {summary}"[:MAX_CONTENT_CHARS]


def format_clusters(plans: Sequence[ThemePlan]) -> str:
    blocks = []
    for n, p in enumerate(plans, start=1):
        facts = "\n".join(f"- (since {f.valid_from:%Y-%m-%d}) {f.content}" for f in p.facts)
        blocks.append(CLUSTER_TEMPLATE.format(n=n, facts=facts))
    return "\n\n".join(blocks)


class _ThemeOut(BaseModel):
    cluster: int
    title: str | None = None
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)


class AbstractorOutput(BaseModel):
    themes: list[_ThemeOut] = Field(default_factory=list)


def apply_output(plans: Sequence[ThemePlan], output: AbstractorOutput) -> dict[str, ThemeText]:
    """Plan key -> text. Clusters the LLM skipped, or titled null ("no real subject"), get
    no entry: no new theme, and an existing theme keeps its current text."""
    out: dict[str, ThemeText] = {}
    for t in output.themes:
        if not 1 <= t.cluster <= len(plans) or not t.title or not t.title.strip():
            continue
        if not t.summary.strip():
            continue
        keywords = [k.strip() for k in t.keywords if k and k.strip()][:8]
        out[plans[t.cluster - 1].key] = ThemeText(
            title=t.title, summary=t.summary, keywords=keywords
        )
    return out


async def summarise_clusters(
    llm: LLMClient, plans: Sequence[ThemePlan], *, max_tokens: int
) -> tuple[dict[str, ThemeText], Usage]:
    if not plans:
        return {}, Usage()
    output, usage = await chat_json(
        llm,
        [
            {"role": "system", "content": ABSTRACTOR_SYSTEM},
            {"role": "user", "content": format_clusters(plans)},
        ],
        AbstractorOutput,
        trace_name="sleep.abstractor",
        max_tokens=max_tokens,
    )
    return apply_output(plans, output), usage
