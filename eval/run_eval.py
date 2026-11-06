"""Nocturne evaluation (spec §9). Run from the repo root:

    python eval/run_eval.py --dataset longmemeval-oracle --limit 30
    python eval/run_eval.py --dataset longmemeval-oracle --limit 30 --fake   # no keys, no quota

Systems: full-context, naive-rag, nocturne-no-sleep, nocturne (default: all four).
Results go to eval/results/<run-name>/: results.csv, summary.csv, totals.csv,
conversations.csv, plus the raw *.jsonl the run resumes from. Rerunning the same command
resumes; cached LLM calls are free.

Needs EVAL_DATABASE_URL (a disposable database with "eval" in its name) for the nocturne
systems, plus the backend's GROQ_API_KEY / GEMINI_API_KEY in backend/.env.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

# `python eval/run_eval.py` puts eval/ itself on sys.path, where eval/datasets would shadow
# any installed `datasets` package. Import from the repo root (and backend/) instead.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "eval"]
for p in (ROOT / "backend", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.db import create_pool  # noqa: E402
from app.eventloop import loop_factory  # noqa: E402
from eval import datasets  # noqa: E402
from eval.config import EvalSettings  # noqa: E402
from eval.judge import Judge  # noqa: E402
from eval.llm import ResponseCache  # noqa: E402
from eval.memdb import EvalDatabase  # noqa: E402
from eval.report import build_reports  # noqa: E402
from eval.runner import (  # noqa: E402
    ALL_SYSTEMS,
    NOCTURNE_SYSTEMS,
    RunAborted,
    RunConfig,
    backend_settings,
    build_clients,
    build_systems,
    check_config,
    run,
    run_fingerprint,
    write_config,
)
from eval.systems import sleep_points  # noqa: E402


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="longmemeval-oracle", choices=sorted(datasets.DATASETS))
    p.add_argument("--data-path", type=Path, help="local copy of the dataset JSON (no download)")
    p.add_argument(
        "--systems", default=",".join(ALL_SYSTEMS), help="comma-separated (default: all)"
    )
    p.add_argument("--limit", type=int, help="number of questions, stratified by category")
    p.add_argument(
        "--split",
        default="all",
        choices=("all", "dev", "test"),
        help="dev for tuning, test for reporting (spec §11)",
    )
    p.add_argument("--categories", default="", help="comma-separated category filter")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", help="default: <dataset>[-fake]")
    p.add_argument(
        "--fake",
        action="store_true",
        help="offline dry run: scripted LLM + hash embedder, scores are meaningless",
    )
    p.add_argument(
        "--max-errors", type=int, default=3, help="stop after this many failures in a row"
    )
    p.add_argument("--no-cache", action="store_true", help="don't read or write the LLM cache")
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--report-only", action="store_true", help="rebuild CSVs from the JSONL")
    p.add_argument(
        "--plan", action="store_true", help="print the selection and LLM call counts, then exit"
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="also write N random verdicts to judge_sample.csv for hand-checking",
    )
    return p.parse_args(argv)


def plan(
    conversations: Sequence[datasets.Conversation], systems: Sequence[str], sleep_every: int = 1
) -> str:
    """What a run will cost, before spending anything. Sleep calls are a floor: one
    extraction per cycle, plus dedup/contradiction batches when similar facts exist."""
    from collections import Counter

    cats = Counter(q.category for c in conversations for q in c.questions)
    n_q = sum(cats.values())
    sessions = sum(len(c.sessions) for c in conversations)
    episodes = sum(len(c.episodes()) for c in conversations)
    lines = [
        f"{n_q} questions in {len(conversations)} conversations "
        f"({sessions} sessions, {episodes} episodes)",
        "  " + ", ".join(f"{k}: {v}" for k, v in sorted(cats.items())),
        f"answer calls: {n_q * len(systems)}   judge calls: {n_q * len(systems)}",
    ]
    if "nocturne" in systems:
        cycles = sum(len(sleep_points(c.session_ends(), sleep_every)) for c in conversations)
        lines.append(
            f"sleep cycles: {cycles} (every {sleep_every} session(s))   sleep calls: >= {cycles}"
            " (up to ~4x with dedup, conflict, link and theme judging)"
        )
    return "\n".join(lines)


async def amain(args: argparse.Namespace) -> int:
    es = EvalSettings()
    run_name = args.run_name or (args.dataset + ("-fake" if args.fake else ""))
    run_dir = es.results_dir / run_name
    if args.report_only:
        print(build_reports(run_dir, sample=args.sample, seed=args.seed))
        return 0

    systems_wanted = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems_wanted) - set(ALL_SYSTEMS)
    if unknown:
        raise SystemExit(f"unknown systems {sorted(unknown)}; choose from {ALL_SYSTEMS}")
    cfg = RunConfig(
        dataset=args.dataset,
        systems=systems_wanted,
        run_name=run_name,
        limit=args.limit,
        split=args.split,
        categories=[c.strip() for c in args.categories.split(",") if c.strip()],
        seed=args.seed,
        fake=args.fake,
        max_errors=args.max_errors,
    )

    conversations = datasets.select(
        datasets.load_dataset(cfg.dataset, args.data_path),
        limit=cfg.limit,
        split=cfg.split,
        categories=cfg.categories or None,
        seed=cfg.seed,
    )
    n_questions = sum(len(c.questions) for c in conversations)
    if not n_questions:
        raise SystemExit("no questions match the selection")
    if args.plan:
        print(plan(conversations, cfg.systems, es.sleep_every_sessions))
        return 0

    needs_db = bool(NOCTURNE_SYSTEMS & set(cfg.systems))
    if needs_db and not es.database_url:
        raise SystemExit(
            "EVAL_DATABASE_URL is not set. The nocturne systems need a disposable Postgres "
            "database with 'eval' in its name, e.g. with docker compose:\n"
            "  EVAL_DATABASE_URL=postgresql://nocturne:nocturne@localhost:5432/nocturne_eval\n"
            "Or run only the baselines: --systems full-context,naive-rag"
        )

    s = backend_settings(es, fake=cfg.fake)
    cache = None if args.no_cache else ResponseCache(es.cache_path)
    clients = build_clients(es, s, fake=cfg.fake, need_sleep="nocturne" in cfg.systems, cache=cache)
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(cfg, es, s, clients)
    check_config(run_dir, fingerprint, allow_change=args.allow_config_change)
    write_config(run_dir, cfg, fingerprint, n_questions)

    pool = None
    try:
        warm = getattr(clients.embedder, "warmup", None)
        if callable(warm):
            await asyncio.to_thread(warm)
        db = None
        if needs_db:
            assert es.database_url
            EvalDatabase.prepare_schema(es.database_url)
            pool = create_pool(s)
            await pool.open(wait=True, timeout=30)
            db = EvalDatabase(es.database_url, pool)
        systems = build_systems(cfg.systems, es=es, s=s, clients=clients, db=db, pool=pool)
        print(
            f"{run_name}: {n_questions} questions in {len(conversations)} conversations x "
            f"{len(systems)} systems -> {run_dir}"
        )
        try:
            await run(
                conversations,
                systems,
                clients=clients,
                judge=Judge(clients.judge, max_tokens=es.judge_max_tokens),
                run_dir=run_dir,
                dataset=cfg.dataset,
                split_of=datasets.split_of,
                max_errors=cfg.max_errors,
            )
        except RunAborted as exc:
            print(f"\nstopped: {exc}", file=sys.stderr)
            print(build_reports(run_dir, sample=args.sample, seed=args.seed))
            return 2
    finally:
        await clients.aclose()
        if pool is not None:
            await pool.close()
        if cache is not None:
            cache.close()

    print()
    print(build_reports(run_dir, sample=args.sample, seed=args.seed))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with asyncio.Runner(loop_factory=loop_factory) as runner:  # psycopg needs this on Windows
        try:
            return runner.run(amain(args))
        except RunAborted as exc:
            print(f"stopped: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\ninterrupted; rerun the same command to resume", file=sys.stderr)
            return 130


if __name__ == "__main__":
    sys.exit(main())
