"""Average-linkage agglomerative clustering on cosine similarity, numpy only (spec §3.2:
the Abstractor clusters fact embeddings; HDBSCAN or agglomerative).

Two clusters merge while their *average* pairwise similarity is at least `threshold`.
Average linkage avoids single linkage's chaining (A~B, B~C, so A, B and C all merge even
when A and C are unrelated), which matters with small embedding models whose unrelated
sentences already sit around 0.4-0.5 cosine.

Implementation: the nearest-neighbour chain algorithm. Average linkage is reducible, so
merging reciprocal nearest neighbours in any order gives the same clusters as greedy
best-pair merging, in O(n^2) time and memory instead of O(n^3). A cluster whose best
partner is below the threshold can never merge again (merges never raise average
similarity above the larger of the two parts), so it's finished on the spot.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float64)
    if m.size == 0:
        return m.reshape(0, 0)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def average_linkage(
    vectors: Sequence[Sequence[float]], *, threshold: float, min_size: int = 1
) -> list[list[int]]:
    """Clusters (lists of input indexes) with at least `min_size` members, each sorted, the
    list ordered by size descending then by first index. Deterministic: ties between
    equally similar partners go to the lower index."""
    n = len(vectors)
    if n == 0:
        return []
    m = normalize(vectors)
    sim = m @ m.T
    np.fill_diagonal(sim, -np.inf)

    size = np.ones(n)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = np.ones(n, dtype=bool)
    done: list[list[int]] = []
    chain: list[int] = []

    def finish(i: int) -> None:
        done.append(sorted(members.pop(i)))
        active[i] = False
        sim[i, :] = -np.inf
        sim[:, i] = -np.inf

    while active.any():
        if not chain:
            chain.append(int(np.flatnonzero(active)[0]))
        top = chain[-1]
        row = sim[top]
        best = int(np.argmax(row))  # first index among ties
        if not active[best] or row[best] < threshold:
            chain.pop()
            finish(top)
            continue
        # Prefer the previous chain element on ties, so reciprocal pairs are recognised.
        if len(chain) > 1 and row[chain[-2]] >= row[best]:
            best = chain[-2]
        if len(chain) > 1 and best == chain[-2]:
            a, b = chain.pop(), chain.pop()
            keep, drop = min(a, b), max(a, b)
            merged = (size[keep] * sim[keep] + size[drop] * sim[drop]) / (size[keep] + size[drop])
            size[keep] += size[drop]
            members[keep].extend(members.pop(drop))
            active[drop] = False
            sim[drop, :] = -np.inf
            sim[:, drop] = -np.inf
            merged[keep] = -np.inf
            merged[~active] = -np.inf
            sim[keep, :] = merged
            sim[:, keep] = merged
        else:
            chain.append(best)

    out = [c for c in done if len(c) >= min_size]
    out.sort(key=lambda c: (-len(c), c[0]))
    return out
