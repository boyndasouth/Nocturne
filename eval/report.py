"""Turn a run's raw JSONL into the CSVs the README uses (spec §9 deliverable).

    results.csv        one row per (system, question): answer, verdict, latency, tokens
    conversations.csv  one row per (system, conversation): memory size, sleep work, tokens
    summary.csv        per system x category accuracy, plus overall rows and system totals
    judge_sample.csv   (--sample N) random verdicts with an empty column for hand-checking

Usage: python eval/report.py eval/results/<run> [--sample 20]
Pure functions over row dicts, so they're unit-tested without a database or an LLM.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

QUESTION_FIELDS = (
    "system",
    "dataset",
    "split",
    "conversation_id",
    "question_id",
    "category",
    "question_type",
    "abstention",
    "question",
    "gold",
    "answer",
    "correct",
    "rubric",
    "judge_raw",
    "retrieval_ms",
    "retrieved",
    "retrieved_via",
    "context_chars",
    "answer_tokens",
    "judge_tokens",
    "status",
    "error",
)
CONVERSATION_FIELDS = (
    "system",
    "dataset",
    "conversation_id",
    "episodes",
    "memory_items",
    "memory_chars",
    "memory_episode",
    "memory_fact",
    "memory_theme",
    "sleep_cycles",
    "facts_created",
    "merges",
    "supersessions",
    "links",
    "archived",
    "audit_score",
    "audit_items",
    "prep_seconds",
    "sleep_tokens",
    "sleep_calls",
    "sleep_cache_hits",
    "truncated_episodes",
    "status",
    "error",
)
SUMMARY_FIELDS = ("system", "category", "n", "correct", "accuracy", "unjudged")
TOTALS_FIELDS = (
    "system",
    "questions",
    "accuracy",
    "task_avg_accuracy",
    "errors",
    "retrieval_p50_ms",
    "retrieval_p95_ms",
    "answer_tokens",
    "sleep_tokens",
    "judge_tokens",
    "mean_memory_items",
    "mean_episodes",
    "memory_ratio",
    "audit_score",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest(rows: Iterable[dict[str, Any]], key: Sequence[str]) -> list[dict[str, Any]]:
    """Keep the last row per key: a resumed run appends retries after earlier errors."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        out[tuple(r.get(k) for k in key)] = r
    return list(out.values())


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    s = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(s)))
    return s[rank - 1]


def summarize(
    questions: Sequence[dict[str, Any]], conversations: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per (system, category) accuracy, and one totals row per system.

    Accuracy counts only answered questions (status ok); an unparseable verdict counts as
    wrong and is reported as `unjudged`. `task_avg_accuracy` is the mean of per-category
    accuracies, LongMemEval's headline, so small categories aren't drowned out."""
    by_cat: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_sys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in questions:
        by_sys[r["system"]].append(r)
        if r["status"] == "ok":
            by_cat[(r["system"], r["category"])].append(r)

    summary: list[dict[str, Any]] = []
    for (system, cat), rs in sorted(by_cat.items()):
        correct = sum(1 for r in rs if r["correct"] is True)
        summary.append(
            {
                "system": system,
                "category": cat,
                "n": len(rs),
                "correct": correct,
                "accuracy": round(correct / len(rs), 4),
                "unjudged": sum(1 for r in rs if r["correct"] is None),
            }
        )

    convs_by_sys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in conversations:
        if c["status"] == "ok":
            convs_by_sys[c["system"]].append(c)

    totals: list[dict[str, Any]] = []
    for system in sorted(by_sys):
        rs = by_sys[system]
        ok = [r for r in rs if r["status"] == "ok"]
        cat_accs = [s["accuracy"] for s in summary if s["system"] == system]
        lat = [float(r["retrieval_ms"]) for r in ok if r.get("retrieval_ms") is not None]
        convs = convs_by_sys.get(system, [])
        episodes = sum(int(c["episodes"]) for c in convs)
        items = sum(int(c["memory_items"]) for c in convs)
        totals.append(
            {
                "system": system,
                "questions": len(ok),
                "accuracy": round(sum(1 for r in ok if r["correct"] is True) / len(ok), 4)
                if ok
                else None,
                "task_avg_accuracy": round(sum(cat_accs) / len(cat_accs), 4) if cat_accs else None,
                "errors": len(rs) - len(ok),
                "retrieval_p50_ms": _round(percentile(lat, 50)),
                "retrieval_p95_ms": _round(percentile(lat, 95)),
                "answer_tokens": sum(int(r.get("answer_tokens") or 0) for r in ok),
                "sleep_tokens": sum(int(c.get("sleep_tokens") or 0) for c in convs),
                "judge_tokens": sum(int(r.get("judge_tokens") or 0) for r in ok),
                "mean_memory_items": round(items / len(convs), 1) if convs else None,
                "mean_episodes": round(episodes / len(convs), 1) if convs else None,
                # Memory items per episode ingested: < 1 means the store is more compact
                # than the raw log (goal 4: sub-linear growth).
                "memory_ratio": round(items / episodes, 3) if episodes else None,
                # Auditor quality (spec §9), weighted by how many verdicts each memory got.
                "audit_score": _weighted_audit(convs),
            }
        )
    return summary, totals


def _weighted_audit(convs: Sequence[dict[str, Any]]) -> float | None:
    pairs = [
        (float(c["audit_score"]), int(c.get("audit_items") or 0))
        for c in convs
        if c.get("audit_score") not in (None, "")
    ]
    n = sum(w for _, w in pairs)
    return round(sum(s * w for s, w in pairs) / n, 4) if n else None


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k) for k in fields})


def format_table(summary: Sequence[dict[str, Any]], totals: Sequence[dict[str, Any]]) -> str:
    """Accuracy by category (rows) x system (columns), for the console."""
    systems = [t["system"] for t in totals]
    cats = sorted({s["category"] for s in summary})
    acc = {(s["system"], s["category"]): s for s in summary}
    width = max([len(c) for c in cats] + [len("task-averaged"), 8])
    col = max([len(s) for s in systems] + [10])
    lines = [" " * width + "  " + "  ".join(s.rjust(col) for s in systems)]

    def cell(s: dict[str, Any] | None) -> str:
        return "-" if s is None else f"{s['accuracy']:.0%} ({s['correct']}/{s['n']})"

    for c in cats:
        lines.append(
            c.ljust(width) + "  " + "  ".join(cell(acc.get((s, c))).rjust(col) for s in systems)
        )
    for label, key in (("overall", "accuracy"), ("task-averaged", "task_avg_accuracy")):
        vals = [t[key] for t in totals]
        lines.append(
            label.ljust(width)
            + "  "
            + "  ".join(("-" if v is None else f"{v:.0%}").rjust(col) for v in vals)
        )
    return "\n".join(lines)


def build_reports(run_dir: Path, *, sample: int = 0, seed: int = 0) -> str:
    questions = latest(read_jsonl(run_dir / "questions.jsonl"), ("system", "question_id"))
    conversations = latest(
        read_jsonl(run_dir / "conversations.jsonl"), ("system", "conversation_id")
    )
    questions.sort(key=lambda r: (r["system"], r["category"], r["question_id"]))
    conversations.sort(key=lambda r: (r["system"], r["conversation_id"]))
    summary, totals = summarize(questions, conversations)

    write_csv(run_dir / "results.csv", questions, QUESTION_FIELDS)
    write_csv(run_dir / "conversations.csv", conversations, CONVERSATION_FIELDS)
    write_csv(run_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(run_dir / "totals.csv", totals, TOTALS_FIELDS)
    if sample:
        judged = [r for r in questions if r["status"] == "ok"]
        picked = random.Random(seed).sample(judged, min(sample, len(judged)))
        write_csv(
            run_dir / "judge_sample.csv",
            [{**r, "hand_verdict": ""} for r in picked],
            (
                "system",
                "question_id",
                "category",
                "question",
                "gold",
                "answer",
                "correct",
                "judge_raw",
                "hand_verdict",
            ),
        )
    return format_table(summary, totals)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--sample", type=int, default=0, help="write N random verdicts to hand-check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    print(build_reports(args.run_dir, sample=args.sample, seed=args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
