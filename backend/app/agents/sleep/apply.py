"""Apply step: turn the judges' decisions into memory changes.

`plan_apply` is pure: it resolves duplicate chains, folds evidence into canonical facts, and
orders supersessions by valid time. `apply_plan` then writes everything in ONE transaction:
facts, merges, supersessions, the Linker's links, the Abstractor's themes, the episodes
marked consolidated and the cycle marked succeeded. A cycle either fully happens or not at
all, and a resumed cycle can never double-apply.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar
from uuid import UUID

from app.agents.sleep.abstractor import theme_content
from app.agents.sleep.contradiction import Supersession, Timed, resolve
from app.agents.sleep.state import (
    Conflict,
    FactCandidate,
    Link,
    ThemePlan,
    ThemeText,
    cand_ref,
    parse_ref,
)
from app.llm.embeddings import Embedder, aembed_documents
from app.memory.cycles import CycleStore
from app.memory.models import LinkType, NewNote, Status, Tier
from app.memory.store import MemoryStore


@dataclass
class InsertPlan:
    key: str  # canonical candidate
    content: str
    keywords: list[str]
    valid_from: datetime
    source_episode_ids: list[UUID]
    folded_keys: list[str] = field(default_factory=list)  # in-cycle duplicates merged in


@dataclass
class MergePlan:
    note_id: UUID
    add_episode_ids: list[UUID]
    candidate_keys: list[str]


@dataclass
class SupersessionPlan:
    winner: str  # ref
    loser: str  # ref
    valid_to: datetime
    reason: str


@dataclass
class ApplyPlan:
    inserts: list[InsertPlan]
    merges: list[MergePlan]
    supersessions: list[SupersessionPlan]


T = TypeVar("T")


def _unique(items: Sequence[T]) -> list[T]:
    return list(dict.fromkeys(items))


def resolve_ref(ref: str, duplicates: Mapping[str, str]) -> str:
    """Follow candidate -> duplicate links to the canonical ref. Duplicates only point at
    earlier candidates or existing notes, so chains end; the guard is belt and braces."""
    seen: set[str] = set()
    while ref.startswith("c:") and ref[2:] in duplicates:
        if ref in seen:
            raise ValueError(f"duplicate cycle at {ref}")
        seen.add(ref)
        ref = duplicates[ref[2:]]
    return ref


def plan_apply(
    candidates: Sequence[FactCandidate],
    duplicates: Mapping[str, str],
    conflicts: Sequence[Conflict],
    note_valid_from: Mapping[UUID, datetime],
) -> ApplyPlan:
    """`note_valid_from` must hold valid_from for every existing note referenced in
    `duplicates` or `conflicts`."""
    by_key = {c.key: c for c in candidates}
    order = {c.key: i + 1 for i, c in enumerate(candidates)}  # existing notes are order 0

    inserts: dict[str, InsertPlan] = {}
    merges: dict[UUID, MergePlan] = {}
    for cand in candidates:
        target = resolve_ref(cand_ref(cand.key), duplicates)
        kind, value = parse_ref(target)
        if kind == "c":
            canon = by_key[value]
            plan = inserts.setdefault(
                value,
                InsertPlan(
                    key=value,
                    content=canon.content,
                    keywords=list(canon.keywords),
                    valid_from=canon.valid_from,
                    source_episode_ids=[],
                ),
            )
            plan.source_episode_ids = _unique([*plan.source_episode_ids, *cand.source_episode_ids])
            plan.keywords = _unique([*plan.keywords, *cand.keywords])[:8]
            plan.valid_from = min(plan.valid_from, cand.valid_from)
            if cand.key != value:
                plan.folded_keys.append(cand.key)
        else:
            note_id = UUID(value)
            m = merges.setdefault(note_id, MergePlan(note_id, [], []))
            m.add_episode_ids = _unique([*m.add_episode_ids, *cand.source_episode_ids])
            m.candidate_keys.append(cand.key)

    def timed(ref: str) -> Timed:
        kind, value = parse_ref(ref)
        if kind == "c":
            return Timed(ref, inserts[value].valid_from, order[value])
        return Timed(ref, note_valid_from[UUID(value)], 0)

    # Earliest winner per loser: that's when the loser stopped being true. Later winners
    # supersede the earliest winner too, so the history stays a chain (C -> B -> A).
    best: dict[str, tuple[Supersession, str]] = {}
    for c in conflicts:
        a, b = resolve_ref(c.new, duplicates), resolve_ref(c.old, duplicates)
        if a == b:
            continue
        s = resolve(timed(a), timed(b))
        current = best.get(s.loser)
        if current is None or (s.valid_to, s.winner) < (current[0].valid_to, current[0].winner):
            best[s.loser] = (s, c.reason)
    supersessions = [
        SupersessionPlan(winner=s.winner, loser=s.loser, valid_to=s.valid_to, reason=reason)
        for s, reason in sorted(best.values(), key=lambda x: (x[0].valid_to, x[0].loser))
    ]
    return ApplyPlan(
        inserts=list(inserts.values()), merges=list(merges.values()), supersessions=supersessions
    )


@dataclass
class StructurePlan:
    """Linker and Abstractor output, in refs. Every ref is a canonical candidate or an
    existing note, so it resolves to a note id inside the apply transaction."""

    links: list[Link] = field(default_factory=list)
    themes: list[tuple[ThemePlan, ThemeText]] = field(default_factory=list)  # new + updated
    retired_themes: list[UUID] = field(default_factory=list)


@dataclass(frozen=True)
class ApplyResult:
    facts_created: int
    merges: int
    supersessions: int
    inserted_ids: dict[str, UUID]
    links_created: int = 0
    themes_created: int = 0
    themes_updated: int = 0
    themes_archived: int = 0


async def apply_plan(
    plan: ApplyPlan,
    *,
    store: MemoryStore,
    cycles: CycleStore,
    embedder: Embedder,
    cycle_id: UUID,
    clock: datetime,
    episode_ids: Sequence[UUID],
    note_valid_from: Mapping[UUID, datetime],
    reinforce_eta: float,
    tokens_used: int,
    llm_calls: int,
    structure: StructurePlan | None = None,
) -> ApplyResult:
    structure = structure or StructurePlan()
    vectors = await aembed_documents(embedder, [p.content for p in plan.inserts])
    theme_texts = [theme_content(text) for _, text in structure.themes]
    theme_vectors = await aembed_documents(embedder, theme_texts)
    links_created = themes_created = themes_updated = 0
    async with store.transaction() as conn:
        inserted = await store.insert_notes_tx(
            conn,
            [
                NewNote(
                    tier=Tier.FACT,
                    content=p.content,
                    embedding=v,
                    keywords=p.keywords,
                    valid_from=p.valid_from,
                    recorded_at=clock,
                    source_episode_ids=p.source_episode_ids,
                )
                for p, v in zip(plan.inserts, vectors, strict=True)
            ],
        )
        ids = {p.key: n.id for p, n in zip(plan.inserts, inserted, strict=True)}
        valid_from = {n.id: n.valid_from for n in inserted} | dict(note_valid_from)

        def note_id(ref: str) -> UUID:
            kind, value = parse_ref(ref)
            return ids[value] if kind == "c" else UUID(value)

        for m in plan.merges:
            await store.merge_evidence_tx(
                conn,
                m.note_id,
                m.add_episode_ids,
                eta=reinforce_eta,
                cycle_id=cycle_id,
                reason=f"merged duplicate evidence from {len(m.add_episode_ids)} episode(s)",
            )
        for s in plan.supersessions:
            winner, loser = note_id(s.winner), note_id(s.loser)
            await store.edit_note(
                conn,
                loser,
                status=Status.SUPERSEDED,
                valid_to=max(s.valid_to, valid_from[loser]),
                cycle_id=cycle_id,
                reason=f"superseded by {winner}" + (f": {s.reason}" if s.reason else ""),
            )
            await store.add_link(conn, winner, loser, LinkType.SUPERSEDES, cycle_id=cycle_id)

        # Linker
        for link in structure.links:
            src, dst = note_id(link.src), note_id(link.dst)
            if src != dst and await store.add_link(
                conn, src, dst, LinkType(link.type), cycle_id=cycle_id
            ):
                links_created += 1

        # Abstractor
        for (tp, text), content, vec in zip(
            structure.themes, theme_texts, theme_vectors, strict=True
        ):
            members = [note_id(r) for r in tp.members]
            if tp.theme_id is None:
                [theme] = await store.insert_notes_tx(
                    conn,
                    [
                        NewNote(
                            tier=Tier.THEME,
                            content=content,
                            embedding=vec,
                            keywords=text.keywords,
                            valid_from=tp.valid_from,
                            recorded_at=clock,
                        )
                    ],
                )
                theme_id = theme.id
                themes_created += 1
            else:
                theme_id = tp.theme_id
                await store.edit_note(
                    conn,
                    theme_id,
                    content=content,
                    keywords=text.keywords,
                    embedding=vec,
                    cycle_id=cycle_id,
                    reason=f"theme re-summarised over {len(members)} facts",
                )
                themes_updated += 1
            await store.replace_theme_members_tx(conn, theme_id, members, cycle_id=cycle_id)
        for theme_id in structure.retired_themes:
            await store.edit_note(
                conn,
                theme_id,
                status=Status.ARCHIVED,
                cycle_id=cycle_id,
                reason="theme retired: too few of its facts are still active",
            )

        await store.mark_consolidated(conn, episode_ids, cycle_id)
        await cycles.finish_tx(
            conn,
            cycle_id,
            episodes_processed=len(episode_ids),
            facts_created=len(inserted),
            merges=len(plan.merges) + sum(len(p.folded_keys) for p in plan.inserts),
            supersessions=len(plan.supersessions),
            tokens_used=tokens_used,
            llm_calls=llm_calls,
            links_created=links_created,
            themes_created=themes_created,
            themes_updated=themes_updated,
            archived=len(structure.retired_themes),
        )
    return ApplyResult(
        facts_created=len(inserted),
        merges=len(plan.merges),
        supersessions=len(plan.supersessions),
        inserted_ids=ids,
        links_created=links_created,
        themes_created=themes_created,
        themes_updated=themes_updated,
        themes_archived=len(structure.retired_themes),
    )
