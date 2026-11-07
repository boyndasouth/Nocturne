"""Forgetter (spec §3.2 step 6, §5): archive notes whose strength has decayed away.

    strength(t) = strength_0 · exp(-(t - last_accessed) / τ)     (never accessed: recorded_at)

Each note has its own τ, which grows on every access (τ ← τ·κ, the spacing effect), so a
memory that keeps being used both resets its clock and fades more slowly afterwards.

What gets archived, and why the rules are conservative:

    episodes  consolidated into facts by an earlier cycle, and decayed below
              `episode_theta` (~30 days untouched at τ = 30). The facts carry their
              content; the raw turn stays reachable through retrieval's archived-episode
              fallback (weak matches, "what did you tell me...").
    facts     decayed below `fact_theta` (~5 months untouched at τ = 30) AND accessed at most
              `max_access` times AND not summarised by an active theme. Archived facts are
              still searched by the same fallback, so forgetting lowers clutter in the main
              search without making anything unrecoverable.

Exempt (spec §5): themes, superseded facts (they're history), episodes not consolidated yet.

No LLM. It runs inside the apply transaction, after this cycle's facts and themes are
written, so a fact that just joined a theme is already protected. Archiving is a status
change with a version snapshot, like every other sleep edit: reversible, never a delete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import psycopg


@dataclass(frozen=True)
class ForgetConfig:
    enabled: bool = True
    episode_theta: float = 0.37  # ≈ e^-1: one τ since last use
    fact_theta: float = 0.0067  # ≈ e^-5: five τ since last use
    max_access: int = 1  # facts used more often than this are kept
    batch_limit: int = 5000  # per cycle, oldest first


def days_until_archived(strength0: float, tau_days: float, theta: float) -> float:
    """How long an untouched note lasts: strength0 · e^(-d/τ) = θ."""
    if strength0 <= theta:
        return 0.0
    return tau_days * math.log(strength0 / theta)


_SELECT = """
with decayed as (
    select n.id, n.tier, n.recorded_at,
           n.strength * exp(
               -greatest(0, extract(epoch from
                   %(clock)s::timestamptz - coalesce(n.last_accessed, n.recorded_at)))
               / 86400.0 / n.decay_tau_days
           ) as current
    from memory_notes n
    where n.status = 'active'
      and n.recorded_at <= %(clock)s::timestamptz
      and (
        (n.tier = 'episode' and n.consolidated_in is not null)
        or (n.tier = 'fact' and n.access_count <= %(max_access)s
            and not exists (
                select 1 from memory_links l join memory_notes t on t.id = l.src_id
                where l.dst_id = n.id and l.link_type = 'summarizes' and t.status = 'active'
            ))
      )
)
select id, tier::text as tier from decayed
where (tier = 'episode' and current < %(episode_theta)s)
   or (tier = 'fact' and current < %(fact_theta)s)
order by recorded_at, id
limit %(limit)s
"""


async def select_forgettable(
    conn: psycopg.AsyncConnection, *, clock: datetime, cfg: ForgetConfig
) -> list[tuple[UUID, str]]:
    if not cfg.enabled:
        return []
    cur = await conn.execute(
        _SELECT,
        {
            "clock": clock,
            "max_access": cfg.max_access,
            "episode_theta": cfg.episode_theta,
            "fact_theta": cfg.fact_theta,
            "limit": cfg.batch_limit,
        },
    )
    return [(r["id"], r["tier"]) for r in await cur.fetchall()]
