"""Forgetting curve and access reinforcement (spec §5).

    strength(t) = strength_0 · exp(-(t - last_accessed) / τ)
    on access:    strength_0 ← min(1, strength_0 + η·(1 - strength_0)),  τ ← τ·κ

Phase 1 uses `reinforce` on the online path. The per-note τ update needs a column that §4.1
doesn't have yet; that's proposed for Phase 3, when the Forgetter is built.
"""

from __future__ import annotations

import math
from datetime import datetime


def reinforce(strength0: float, eta: float) -> float:
    """Move strength a fraction η of the way towards 1."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    s = max(0.0, min(1.0, strength0))
    return min(1.0, s + eta * (1.0 - s))


def current_strength(
    strength0: float, last_accessed: datetime | None, now: datetime, tau_days: float
) -> float:
    """Decayed strength at `now`. Notes never accessed haven't started decaying."""
    if tau_days <= 0:
        raise ValueError("tau_days must be positive")
    if last_accessed is None:
        return strength0
    elapsed_days = max(0.0, (now - last_accessed).total_seconds() / 86_400)
    return strength0 * math.exp(-elapsed_days / tau_days)
