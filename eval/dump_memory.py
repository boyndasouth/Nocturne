"""Print what the eval database remembers about a topic: run from the repo root after an eval
run (the database keeps the last conversation's memory until the next run wipes it).

    python eval/run_eval.py --dataset longmemeval-s --questions 3a704032 --systems nocturne \\
        --run-name debug-3a704032
    python eval/dump_memory.py "peace lily" succulent

A fact or theme is listed if its text mentions a word, or if one of its source turns does.
The second catches facts that a merge or supersession reworded or replaced. For each note:
its status (active / superseded / archived), when it was true, its version history (what
each sleep cycle changed and why) and the notes that superseded it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "eval"]
for p in (ROOT / "backend", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from eval.config import EvalSettings  # noqa: E402

NOTES = """
with words as (select '%%' || w || '%%' as pat from unnest(%(words)s::text[]) as w),
hit_episodes as (
    select e.id from memory_notes e, words
    where e.tier = 'episode' and e.content ilike words.pat
)
select n.id, n.tier::text, n.status::text, n.content, n.valid_from, n.valid_to,
       n.version, n.strength,
       exists (select 1 from words where n.content ilike words.pat) as text_match,
       (select count(*) from unnest(n.source_episode_ids) s
         where s in (select id from hit_episodes)) as source_match
from memory_notes n
where n.tier = any(%(tiers)s::note_tier[])
  and (exists (select 1 from words where n.content ilike words.pat)
       or n.source_episode_ids && array(select id from hit_episodes))
order by n.tier, n.valid_from
"""
VERSIONS = """
select version, reason, content from memory_versions where note_id = %s order by version
"""
SUPERSEDED_BY = """
select n.content, n.status::text from memory_links l join memory_notes n on n.id = l.src_id
where l.dst_id = %s and l.link_type = 'supersedes'
"""


def _day(t: object) -> str:
    return "-" if t is None else f"{t:%Y-%m-%d}"


def dump(database_url: str, words: Sequence[str], tiers: Sequence[str]) -> int:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        rows = conn.execute(NOTES, {"words": list(words), "tiers": list(tiers)}).fetchall()
        if not rows:
            print(
                f"no {'/'.join(tiers)} notes mention {', '.join(words)} (or cite a turn that does)"
            )
            return 1
        for r in rows:
            why = "text" if r["text_match"] else f"{r['source_match']} source turn(s)"
            print(
                f"\n[{r['tier']} | {r['status']} | {_day(r['valid_from'])} → "
                f"{_day(r['valid_to'])} | v{r['version']} | matched by {why}]"
            )
            print(f"  {r['content']}")
            for v in conn.execute(VERSIONS, (r["id"],)).fetchall():
                print(f"    v{v['version']} before: {v['reason'] or '(no reason)'}")
                if v["content"] != r["content"]:
                    print(f"        was: {v['content']}")
            for s in conn.execute(SUPERSEDED_BY, (r["id"],)).fetchall():
                print(f"    superseded by ({s['status']}): {s['content']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("words", nargs="+", help="case-insensitive; a note matching any is shown")
    p.add_argument("--tiers", default="fact,theme", help="comma-separated: fact,theme,episode")
    args = p.parse_args(argv)
    url = EvalSettings().database_url
    if not url:
        raise SystemExit("EVAL_DATABASE_URL is not set")
    return dump(url, args.words, [t.strip() for t in args.tiers.split(",") if t.strip()])


if __name__ == "__main__":
    sys.exit(main())
