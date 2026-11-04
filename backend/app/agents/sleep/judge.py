"""Shared pieces for the dedup and contradiction judges: neighbour search over existing facts
and earlier candidates, and the lettered group prompt format."""

from __future__ import annotations

import string
from collections.abc import Sequence
from datetime import datetime
from typing import TypeVar

import numpy as np

from app.agents.sleep.state import FactCandidate, Group, NeighborRef, cand_ref, note_ref
from app.llm.prompts import GROUP_TEMPLATE
from app.memory.models import Neighbor

LETTERS = string.ascii_uppercase
T = TypeVar("T")


def cosine_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.where(norms == 0, 1.0, norms)
    return m @ m.T


def neighbor_lists(
    candidates: Sequence[FactCandidate],
    sims: np.ndarray,
    existing: Sequence[Sequence[Neighbor]],
    *,
    min_sim: float,
    k: int,
    eligible: Sequence[bool] | None = None,
) -> list[list[NeighborRef]]:
    """For each candidate: its top-k neighbours with sim ≥ min_sim among existing facts and
    *earlier* eligible candidates (earlier only, so each pair is judged once and merges can't
    form cycles). Ties break towards existing facts, then by ref for determinism."""
    eligible = eligible if eligible is not None else [True] * len(candidates)
    out: list[list[NeighborRef]] = []
    for i in range(len(candidates)):
        pool: list[NeighborRef] = [
            NeighborRef(
                ref=note_ref(n.note.id),
                content=n.note.content,
                valid_from=n.note.valid_from,
                sim=n.sim,
            )
            for n in existing[i]
            if n.sim >= min_sim
        ]
        for j in range(i):
            if eligible[j] and float(sims[i, j]) >= min_sim:
                other = candidates[j]
                pool.append(
                    NeighborRef(
                        ref=cand_ref(other.key),
                        content=other.content,
                        valid_from=other.valid_from,
                        sim=float(sims[i, j]),
                    )
                )
        pool.sort(key=lambda n: (-round(n.sim, 6), n.ref[0] != "n", n.ref))
        out.append(pool[:k] if eligible[i] else [])
    return out


def _date(d: datetime) -> str:
    return f"{d:%Y-%m-%d}"


def format_groups(groups: Sequence[Group], *, with_dates: bool) -> str:
    blocks = []
    for n, g in enumerate(groups, start=1):
        new = f"(true since {_date(g.valid_from)}) {g.content}" if with_dates else g.content
        existing = "\n".join(
            f"[{LETTERS[i]}] "
            + (f"(true since {_date(nb.valid_from)}) " if with_dates else "")
            + nb.content
            for i, nb in enumerate(g.neighbors[: len(LETTERS)])
        )
        blocks.append(GROUP_TEMPLATE.format(n=n, new=new, existing=existing))
    return "\n\n".join(blocks)


def letter_to_ref(group: Group, letter: str | None) -> str | None:
    if not letter:
        return None
    letter = letter.strip().strip("[]").upper()
    if len(letter) != 1 or letter not in LETTERS:
        return None
    idx = LETTERS.index(letter)
    return group.neighbors[idx].ref if idx < len(group.neighbors) else None


def chunks(items: Sequence[T], size: int) -> list[list[T]]:
    size = max(1, size)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
