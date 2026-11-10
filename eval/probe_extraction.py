"""Extraction probe: measure the Extractor's recall on benchmark evidence turns for a few
dozen LLM calls instead of a full eval run. Run from the repo root:

    python eval/probe_extraction.py --dataset longmemeval-s --limit 10
    python eval/probe_extraction.py --dataset longmemeval-s --limit 10 --prompt v2

It replays a run's sleep schedule (same sleep points, same per-cycle episode cap, same
extraction batches and prompts as the real sleep graph) but calls the Extractor only on
batches that contain an evidence turn. Coverage is the fraction of evidence turns that
became the source of at least one fact: the same thing a full run's "evidence coverage"
reports, since later sleep steps carry a fact's sources along when they merge it.

The prompts are byte-identical to a full run's, so batches a full run already extracted
come from the LLM cache for free, and a probe of a run's settings reproduces its coverage.
Flags override the backend settings (backend/.env) for this probe only.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "eval"]
for p in (ROOT / "backend", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.agents.sleep.extractor import batch_episodes, extract_batch  # noqa: E402
from app.agents.sleep.graph import SleepConfig  # noqa: E402
from app.agents.sleep.state import EpisodeIn  # noqa: E402
from app.eventloop import loop_factory  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.llm.structured import InvalidJSONError  # noqa: E402
from eval import datasets  # noqa: E402
from eval.config import EvalSettings  # noqa: E402
from eval.datasets.schema import Conversation, Episode  # noqa: E402
from eval.llm import ResponseCache  # noqa: E402
from eval.runner import backend_settings, build_clients  # noqa: E402
from eval.systems import episode_id, sleep_points  # noqa: E402


@dataclass
class ProbeResult:
    conversation_id: str
    category: str
    evidence: int = 0
    covered: int = 0
    batches: int = 0  # extraction batches sent (after any splits)
    missed: list[Episode] = field(default_factory=list)
    evidence_facts: list[str] = field(default_factory=list)  # facts sourced from evidence


def sleep_batches(
    conv: Conversation, *, sleep_every: int, max_episodes: int, cfg: SleepConfig
) -> list[list[tuple[EpisodeIn, Episode]]]:
    """The extraction batches a full run makes for this conversation, in order. Mirrors
    NocturneSystem.prepare + the sleep runner: at each sleep point, the episodes recorded
    up to then, oldest first, at most `max_episodes` per cycle, batched as the graph does."""
    # Oldest first, ties in dataset order: the eval's episode ids sort that way, and the
    # sleep runner orders by (time, id).
    episodes = sorted(enumerate(conv.episodes()), key=lambda ie: ie[1].at)
    pairs = [
        (EpisodeIn(id=episode_id(conv.id, i), content=e.content, valid_from=e.at), e)
        for i, e in episodes
    ]
    by_id = {ep.id: (ep, e) for ep, e in pairs}
    out: list[list[tuple[EpisodeIn, Episode]]] = []
    i = 0
    for end in sleep_points(conv.session_ends(), sleep_every):
        pending: list[EpisodeIn] = []
        while i < len(pairs) and pairs[i][1].at <= end:
            pending.append(pairs[i][0])
            i += 1
        for start in range(0, len(pending), max_episodes):
            cycle = pending[start : start + max_episodes]
            for batch in batch_episodes(
                cycle,
                max_chars=cfg.extract_batch_chars,
                max_items=cfg.extract_batch_episodes,
                assistant_chars=cfg.extract_assistant_chars,
            ):
                out.append([by_id[ep.id] for ep in batch])
    return out


async def probe_conversation(
    conv: Conversation,
    llm: LLMClient,
    *,
    cfg: SleepConfig,
    sleep_every: int,
    max_episodes: int,
) -> ProbeResult:
    q = conv.questions[0]
    result = ProbeResult(conv.id, q.category)
    batches = sleep_batches(conv, sleep_every=sleep_every, max_episodes=max_episodes, cfg=cfg)
    evidence: dict[UUID, Episode] = {ep.id: e for batch in batches for ep, e in batch if e.evidence}
    result.evidence = len(evidence)
    covered: set[UUID] = set()
    queue = [[ep for ep, _ in b] for b in batches if any(e.evidence for _, e in b)]
    while queue:
        batch = queue.pop(0)
        try:
            facts, _ = await extract_batch(
                llm,
                batch,
                result.batches,
                max_tokens=cfg.max_tokens,
                assistant_chars=cfg.extract_assistant_chars,
                prompt=cfg.extract_prompt,
            )
        except InvalidJSONError:
            if len(batch) < 2:
                raise
            half = (len(batch) + 1) // 2  # same split as the sleep graph
            queue[:0] = [batch[:half], batch[half:]]
            result.batches += 1
            continue
        result.batches += 1
        for f in facts:
            hits = [i for i in f.source_episode_ids if i in evidence]
            covered.update(hits)
            if hits:
                result.evidence_facts.append(f.content)
    result.covered = len(covered)
    result.missed = [e for i, e in evidence.items() if i not in covered]
    return result


def _excerpt(e: Episode, n: int = 160) -> str:
    user = e.content.split("\nAssistant: ", 1)[0].replace("\n", " ")
    return user if len(user) <= n else user[: n - 1] + "…"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="longmemeval-s", choices=sorted(datasets.DATASETS))
    p.add_argument("--data-path", type=Path)
    p.add_argument("--limit", type=int, help="questions, picked exactly like run_eval.py")
    p.add_argument("--split", default="all", choices=("all", "dev", "test"))
    p.add_argument("--categories", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--questions", default="", help="comma-separated question ids instead")
    p.add_argument("--sleep-every", type=int, help="default: EVAL_SLEEP_EVERY_SESSIONS")
    p.add_argument("--batch-episodes", type=int, help="override EXTRACT_BATCH_EPISODES")
    p.add_argument("--batch-chars", type=int, help="override EXTRACT_BATCH_CHARS")
    p.add_argument("--assistant-chars", type=int, help="override EXTRACT_ASSISTANT_CHARS")
    p.add_argument("--prompt", choices=("v1", "v2"), help="override EXTRACT_PROMPT")
    p.add_argument("--show-facts", action="store_true", help="print facts from evidence turns")
    p.add_argument("--fake", action="store_true", help="scripted LLM, no keys (smoke test)")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args(argv)


async def amain(args: argparse.Namespace) -> int:
    es = EvalSettings()
    sleep_every = args.sleep_every or es.sleep_every_sessions
    overrides = {
        k: v
        for k, v in {
            "extract_batch_episodes": args.batch_episodes,
            "extract_batch_chars": args.batch_chars,
            "extract_assistant_chars": args.assistant_chars,
            "extract_prompt": args.prompt,
        }.items()
        if v is not None
    }
    s = backend_settings(es, fake=args.fake).model_copy(update=overrides)
    cfg = SleepConfig.from_settings(s)

    convs = datasets.load_dataset(args.dataset, args.data_path)
    if args.questions:
        wanted = {q.strip() for q in args.questions.split(",") if q.strip()}
        convs = [
            replace(c, questions=qs)
            for c in convs
            if (qs := tuple(q for q in c.questions if q.id in wanted))
        ]
    else:
        convs = datasets.select(
            convs,
            limit=args.limit,
            split=args.split,
            categories=[c.strip() for c in args.categories.split(",") if c.strip()] or None,
            seed=args.seed,
        )
    if not convs:
        raise SystemExit("no questions match the selection")

    print(
        f"probe: {len(convs)} conversations, sleep every {sleep_every} sessions, batches of "
        f"<= {cfg.extract_batch_episodes} episodes / {cfg.extract_batch_chars} chars, "
        f"assistant replies {cfg.extract_assistant_chars or 'whole'}, "
        f"prompt {cfg.extract_prompt}"
    )
    cache = None if args.no_cache else ResponseCache(es.cache_path)
    clients = build_clients(es, s, fake=args.fake, need_sleep=True, cache=cache)
    assert clients.sleep is not None
    total_ev = total_cov = 0
    try:
        for conv in convs:
            r = await probe_conversation(
                conv,
                clients.sleep,
                cfg=cfg,
                sleep_every=sleep_every,
                max_episodes=s.sleep_max_episodes,
            )
            if not r.evidence:
                print(f"{r.conversation_id:16} {r.category:27} no evidence turns")
                continue
            total_ev += r.evidence
            total_cov += r.covered
            print(
                f"{r.conversation_id:16} {r.category:27} {r.covered}/{r.evidence} covered"
                f"   ({r.batches} batch{'es' if r.batches != 1 else ''})"
            )
            for e in r.missed:
                print(f"    missed {e.at:%Y-%m-%d}: {_excerpt(e)}")
            if args.show_facts:
                for f in r.evidence_facts:
                    print(f"    fact: {f}")
    finally:
        meter = clients.sleep.meter
        await clients.aclose()
        if cache is not None:
            cache.close()
    pct = f"{total_cov / total_ev:.0%}" if total_ev else "-"
    print(
        f"\nevidence coverage {total_cov}/{total_ev} ({pct})   sleep calls {meter.calls} "
        f"({meter.calls - meter.cache_hits} new, {meter.cache_hits} cached)   "
        f"tokens {meter.tokens:,}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
