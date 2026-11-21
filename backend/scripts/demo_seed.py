"""Fill a database with a small, hand-written memory for trying the frontend without spending
LLM quota: three months of a student's conversations and the three nights of sleep that
consolidated them (facts learned, a move that supersedes an old address, merged duplicate
evidence, themes, links, forgotten turns, audit verdicts).

    python -m scripts.demo_seed postgresql://postgres:postgres@localhost:5432/nocturne_demo

The database must have no memory yet, or a name containing "demo" and --wipe (which
truncates every memory table). Migrations are applied first. Embeddings come from the
configured embedder, so /chat retrieval over the demo works; --hash skips loading it.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.llm.embeddings import Embedder, HashEmbedder
from scripts.migrate import migrate


def at(month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


@dataclass
class Night:
    clock: datetime
    turns: list[tuple[datetime, str, str]]
    facts: list[tuple[str, str, list[int], list[str]]] = field(default_factory=list)
    # (key, statement, indexes into this night's turns, keywords)


NIGHTS = [
    Night(
        at(1, 11, 3),
        [
            (at(1, 10, 9), "I live in Austin, near campus.", "Noted. Is that close to UT?"),
            (
                at(1, 10, 9, 5),
                "Yes, I'm a second-year CS student at UT Austin.",
                "Nice. What are you taking this semester?",
            ),
            (
                at(1, 10, 9, 12),
                "Linear algebra is the hard one. My midterm is on Friday the 16th.",
                "Want a study plan for eigenvalues before Friday?",
            ),
            (
                at(1, 10, 18),
                "I study best in the mornings at Epoch Coffee, evenings I'm useless.",
                "Then morning blocks it is.",
            ),
            (
                at(1, 10, 18, 20),
                "Also I'm vegetarian, so skip the brisket recommendations, ha.",
                "Understood, no brisket.",
            ),
            (
                at(1, 10, 21),
                "My cat Miso keeps walking across the keyboard while I do problem sets.",
                "Miso has opinions about linear algebra.",
            ),
            (
                at(1, 10, 21, 10),
                "I'm training for the Austin half marathon on Feb 15.",
                "How many miles a week are you running?",
            ),
        ],
        [
            ("austin", "The user lives in Austin, near the UT campus.", [0], ["austin", "home"]),
            ("ut", "The user is a second-year computer science student at UT Austin.", [1], []),
            ("midterm", "The user's linear algebra midterm is on January 16, 2026.", [2], []),
            ("mornings", "The user studies best in the mornings, at Epoch Coffee.", [3], []),
            ("veg", "The user is vegetarian.", [4], ["diet"]),
            ("miso", "The user has a cat named Miso.", [5], ["pet"]),
            ("half", "The user is training for the Austin half marathon on February 15.", [6], []),
        ],
    ),
    Night(
        at(2, 17, 3),
        [
            (
                at(2, 2, 10),
                "Got my linear algebra midterm back: 92!",
                "Congratulations. The morning blocks paid off.",
            ),
            (
                at(2, 2, 10, 30),
                "I started learning Rust for a systems class project.",
                "Good choice for systems work.",
            ),
            (
                at(2, 9, 8),
                "Did my last long run before the race, 12 miles at 8:40 pace.",
                "That's a solid taper point.",
            ),
            (
                at(2, 16, 14),
                "Finished the half marathon yesterday in 1:52:10!",
                "Great time for a first half.",
            ),
            (
                at(2, 16, 14, 5),
                "Miso slept through the whole celebration.",
                "Classic cat behaviour.",
            ),
        ],
        [
            ("score", "The user scored 92 on the linear algebra midterm.", [0], ["grades"]),
            ("rust", "The user is learning Rust for a systems class project.", [1], []),
            ("longrun", "The user ran 12 miles at an 8:40 pace on February 9.", [2], ["running"]),
            ("finish", "The user finished the Austin half marathon in 1:52:10.", [3], []),
        ],
    ),
    Night(
        at(3, 3, 3),
        [
            (
                at(3, 2, 18),
                "Big news, I moved to Dallas for a spring co-op at a fintech company.",
                "Congratulations! Is it remote-friendly or in office?",
            ),
            (
                at(3, 2, 18, 10),
                "In office. The team writes everything in Go, so Rust is on hold.",
                "Go will feel simpler after Rust.",
            ),
            (
                at(3, 2, 18, 30),
                "Looking for a new morning coffee spot in Dallas to study.",
                "Try somewhere near your office so mornings stay easy.",
            ),
            (
                at(3, 2, 19),
                "Thinking about the Dallas marathon in December. Full distance this time.",
                "Ambitious. You'd have nine months to build up.",
            ),
        ],
        [
            ("dallas", "The user lives in Dallas.", [0], ["dallas", "home"]),
            ("coop", "The user is doing a spring co-op at a fintech company in Dallas.", [0], []),
            ("go", "The user writes Go at work; learning Rust is on hold.", [1], []),
            ("marathon", "The user is considering the Dallas marathon in December.", [3], []),
        ],
    ),
]


def emb(embedder: Embedder, text: str) -> list[float]:
    return embedder.embed_documents([text])[0]


def seed(url: str, embedder: Embedder, *, wipe: bool, owner: str = "local") -> None:
    migrate(url, verbose=False)
    with psycopg.connect(url, row_factory=dict_row) as conn:
        # Rows default to this owner (migration 006); 'local' is the accounts-off memory.
        conn.execute("select set_config('nocturne.owner', %s, false)", (owner,))
        n = conn.execute(
            "select count(*) as n from memory_notes where owner_id = %s", (owner,)
        ).fetchone()
        if n and n["n"]:
            dbname = conn.info.dbname
            if not wipe or "demo" not in dbname:
                raise SystemExit(
                    f"{dbname} already has {n['n']} notes. Use an empty database, or one whose"
                    " name contains 'demo' with --wipe."
                )
            conn.execute(
                "truncate memory_links, memory_versions, memory_notes, sleep_cycles cascade"
            )

        def note(tier: str, content: str, *, valid_from: datetime, recorded_at: datetime,
                 sources: list[uuid.UUID] | None = None, keywords: list[str] | None = None,
                 consolidated_in: uuid.UUID | None = None, strength: float = 1.0,
                 access_count: int = 0) -> uuid.UUID:  # fmt: skip
            row = conn.execute(
                "insert into memory_notes (tier, content, keywords, embedding, valid_from,"
                " recorded_at, source_episode_ids, consolidated_in, strength, access_count)"
                " values (%s::note_tier, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s)"
                " returning id",
                (tier, content, keywords or [], str(emb(embedder, content)), valid_from,
                 recorded_at, sources or [], consolidated_in, strength, access_count),
            ).fetchone()  # fmt: skip
            assert row
            return row["id"]

        def edit(note_id: uuid.UUID, cycle: uuid.UUID, reason: str, **changes: object) -> None:
            conn.execute(
                "insert into memory_versions (note_id, version, content, snapshot,"
                " changed_by_cycle, reason, created_at)"
                " select id, version, content, to_jsonb(n) - 'search_tsv' - 'embedding', %s, %s,"
                " (select clock from sleep_cycles where cycle_id = %s)"
                " from memory_notes n where id = %s",
                (cycle, reason, cycle, note_id),
            )
            sets = ", ".join(
                f"{k} = %s" + ("::note_status" if k == "status" else "") for k in changes
            )
            conn.execute(
                f"update memory_notes set {sets}, version = version + 1 where id = %s",
                (*changes.values(), note_id),
            )

        def link(src: uuid.UUID, dst: uuid.UUID, kind: str, cycle: uuid.UUID) -> None:
            conn.execute(
                "insert into memory_links (src_id, dst_id, link_type, created_by_cycle)"
                " values (%s, %s, %s::link_type, %s) on conflict do nothing",
                (src, dst, kind, cycle),
            )

        facts: dict[str, uuid.UUID] = {}
        cycles: list[uuid.UUID] = []
        for night in NIGHTS:
            ids = [
                note("episode", f"User: {u}\nAssistant: {a}", valid_from=t, recorded_at=t)
                for t, u, a in night.turns
            ]
            row = conn.execute(
                "insert into sleep_cycles (started_at, finished_at, clock, status, episode_ids,"
                " episodes_processed, facts_created, llm_calls, tokens_used)"
                " values (%s, %s, %s, 'succeeded', %s, %s, %s, %s, %s) returning cycle_id",
                (night.clock, night.clock + timedelta(minutes=2), night.clock, ids, len(ids),
                 len(night.facts),
                 6 + len(ids) // 3, 9000 + 1500 * len(ids)),
            ).fetchone()  # fmt: skip
            assert row
            cid = row["cycle_id"]
            cycles.append(cid)
            for key, text, idx, kw in night.facts:
                facts[key] = note(
                    "fact",
                    text,
                    valid_from=night.turns[idx[0]][0],
                    recorded_at=night.clock,
                    sources=[ids[i] for i in idx],
                    keywords=kw,
                )
            conn.execute(
                "update memory_notes set consolidated_in = %s where id = any(%s)", (cid, ids)
            )

        c1, c2, c3 = cycles
        e2 = conn.execute(
            "select id from memory_notes where tier = 'episode' and consolidated_in = %s"
            " order by valid_from",
            (c2,),
        ).fetchall()

        # Night 2: the Miso turn is duplicate evidence; the midterm is over; a running theme.
        edit(facts["miso"], c2, "merged duplicate evidence from 1 episode(s): Miso slept "
             "through the whole celebration.",
             source_episode_ids=[*conn.execute("select source_episode_ids from memory_notes "
             "where id = %s", (facts["miso"],)).fetchone()["source_episode_ids"], e2[4]["id"]],
             strength=1.0)  # fmt: skip
        edit(facts["half"], c2, f"superseded by {facts['finish']}: the race is done",
             status="superseded", valid_to=at(2, 15, 9))  # fmt: skip
        link(facts["finish"], facts["half"], "supersedes", c2)
        link(facts["longrun"], facts["half"], "elaborates", c2)
        link(facts["score"], facts["midterm"], "elaborates", c2)
        link(facts["score"], facts["mornings"], "related", c2)
        running = note(
            "theme",
            "Running: the user trained for and finished the Austin half marathon (1:52:10) "
            "in February 2026.",
            valid_from=at(1, 10, 21, 10),
            recorded_at=NIGHTS[1].clock,
            keywords=["running", "half marathon"],
        )
        for k in ("half", "longrun", "finish"):
            link(running, facts[k], "summarizes", c2)
        studies = note(
            "theme",
            "Studies: second-year CS at UT Austin; linear algebra midterm went well (92); "
            "learning Rust for a systems project.",
            valid_from=at(1, 10, 9, 5),
            recorded_at=NIGHTS[1].clock,
            keywords=["school"],
        )
        for k in ("ut", "midterm", "score", "rust"):
            link(studies, facts[k], "summarizes", c2)
        conn.execute(
            "update sleep_cycles set merges = 1, supersessions = 1, links_created = 4,"
            " themes_created = 2, facts_created = 4 where cycle_id = %s",
            (c2,),
        )

        # Night 3: the move supersedes Austin; Go replaces Rust; themes re-summarised.
        edit(facts["austin"], c3, f"superseded by {facts['dallas']}: moved for a co-op",
             status="superseded", valid_to=at(3, 2, 18))  # fmt: skip
        link(facts["dallas"], facts["austin"], "supersedes", c3)
        edit(facts["rust"], c3, f"superseded by {facts['go']}: Rust is on hold",
             status="superseded", valid_to=at(3, 2, 18, 10))  # fmt: skip
        link(facts["go"], facts["rust"], "supersedes", c3)
        link(facts["go"], facts["coop"], "caused_by", c3)
        link(facts["dallas"], facts["coop"], "caused_by", c3)
        link(facts["marathon"], facts["finish"], "related", c3)
        edit(running, c3, "theme re-summarised over 4 facts",
             content="Running: after a 1:52:10 Austin half marathon in February 2026, the user "
             "is considering the full Dallas marathon in December.")  # fmt: skip
        link(running, facts["marathon"], "summarizes", c3)
        edit(studies, c3, "theme re-summarised over 5 facts",
             content="Studies and work: CS at UT Austin (92 on the linear algebra midterm), now "
             "on a spring co-op at a Dallas fintech writing Go; Rust is on hold.")  # fmt: skip
        link(studies, facts["coop"], "summarizes", c3)
        link(studies, facts["go"], "summarizes", c3)
        # The Forgetter: old small-talk turns, already consolidated, decay away.
        old = conn.execute(
            "select id from memory_notes where tier = 'episode' and consolidated_in = %s"
            " order by valid_from",
            (c1,),
        ).fetchall()
        for r in (old[3], old[5]):
            edit(r["id"], c3, "forgotten: decayed episode (consolidated into facts)",
                 status="archived")  # fmt: skip
        conn.execute(
            "update sleep_cycles set supersessions = 2, links_created = 5, themes_updated = 2,"
            " archived = 2, facts_created = 4, audit_score = 0.875, audit_items = 4, audit = %s"
            " where cycle_id = %s",
            (
                Jsonb(
                    {
                        "by_kind": {"supersede": 1.0, "link": 0.75, "theme": 1.0},
                        "verdicts": [
                            {"item": 1, "kind": "supersede", "verdict": "good",
                             "note": "Moving to Dallas replaces living in Austin."},
                            {"item": 2, "kind": "supersede", "verdict": "good",
                             "note": "Rust explicitly put on hold."},
                            {"item": 3, "kind": "link", "verdict": "questionable",
                             "note": "Writing Go is due to the co-op, but 'caused_by' is loose."},
                            {"item": 4, "kind": "theme", "verdict": "good",
                             "note": "Faithful to its facts."},
                        ],
                    }
                ),
                c3,
            ),
        )  # fmt: skip
        conn.execute(
            "update memory_notes set access_count = 3, last_accessed = %s where id = any(%s)",
            (at(3, 5), [facts["dallas"], facts["veg"], facts["mornings"]]),
        )
        conn.commit()
        print(f"seeded {len(facts)} facts, 2 themes and {len(cycles)} sleep cycles for {owner}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("database_url")
    p.add_argument("--wipe", action="store_true", help="empty a *demo* database first")
    p.add_argument("--hash", action="store_true", help="hash embeddings (no model download)")
    p.add_argument(
        "--owner",
        default="local",
        help="whose memory to fill: 'local' (accounts off) or a Supabase user id",
    )
    args = p.parse_args(argv)
    if args.hash:
        embedder: Embedder = HashEmbedder()
    else:
        from app.config import Settings
        from app.llm.embeddings import build_embedder

        embedder = build_embedder(Settings(database_url=args.database_url))
    seed(args.database_url, embedder, wipe=args.wipe, owner=args.owner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
