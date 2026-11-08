"""Runs systems over a benchmark and appends raw rows to JSONL (spec §9 protocol).

For each conversation, each system builds its memory (`prepare`), then answers every
selected question, and the judge grades each answer. Rows are appended and flushed as they
happen, so an interrupted run (Ctrl-C, a daily quota) resumes where it stopped: questions
with an `ok` row are skipped, errored ones are retried. With the LLM cache, finished work
costs nothing on resume, including sleep cycles, which rebuild the same memory from cache.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.llm.client import LLMClient, OpenAICompatClient, QuotaExhaustedError
from app.llm.embeddings import Embedder, HashEmbedder, build_embedder
from app.llm.providers import build_online_client, build_sleep_client
from eval.base import CachedEmbedder, System
from eval.baselines.full_context import FullContextSystem
from eval.baselines.naive_rag import NaiveRagSystem
from eval.baselines.no_sleep import nocturne_no_sleep
from eval.config import BACKEND_ENV, ROOT, EvalSettings
from eval.datasets.schema import Conversation
from eval.fake import FakeLLM
from eval.judge import Judge
from eval.llm import EvalLLM, Meter, ResponseCache
from eval.memdb import EvalDatabase
from eval.systems import NocturneSystem

ALL_SYSTEMS = ("full-context", "naive-rag", "nocturne-no-sleep", "nocturne")
NOCTURNE_SYSTEMS = frozenset({"nocturne-no-sleep", "nocturne"})


class RunAborted(RuntimeError):
    pass


@dataclass
class RunConfig:
    dataset: str
    systems: list[str]
    run_name: str
    limit: int | None = None
    split: str = "all"
    categories: list[str] = field(default_factory=list)
    seed: int = 0
    fake: bool = False
    max_errors: int = 3  # consecutive failures before giving up (e.g. a daily quota is gone)


# --- Building clients and systems ----------------------------------------------------------


def backend_settings(eval_settings: EvalSettings, *, fake: bool) -> Settings:
    """The app's settings (from backend/.env), pointed at the eval database, with the idle
    trigger off. Fake runs use the hash embedder so no model download is needed."""
    overrides: dict[str, Any] = {"sleep_auto": False}
    # Never fall back to the app's DATABASE_URL: the eval database is wiped constantly.
    overrides["database_url"] = eval_settings.database_url or "postgresql://unset/eval"
    if fake:
        overrides["embedding_backend"] = "hash"
    env_file = BACKEND_ENV if BACKEND_ENV.exists() else None
    return Settings(_env_file=env_file, **overrides)  # type: ignore[call-arg]


def build_judge_client(es: EvalSettings, s: Settings) -> OpenAICompatClient:
    keys = {
        "groq": (s.groq_api_key, s.groq_base_url, "GROQ_API_KEY"),
        "gemini": (s.gemini_api_key, s.gemini_base_url, "GEMINI_API_KEY"),
        "openrouter": (s.openrouter_api_key, s.openrouter_base_url, "OPENROUTER_API_KEY"),
    }
    key, base_url, env_name = keys[es.judge_provider]
    if key is None:
        raise RuntimeError(f"{env_name} is not set (needed for the judge).")
    extra: dict[str, object] = {}
    if es.judge_reasoning_effort:
        extra["reasoning_effort"] = es.judge_reasoning_effort
        if es.judge_provider == "groq":
            extra["include_reasoning"] = False
    return OpenAICompatClient(
        name=f"judge-{es.judge_provider}",
        base_url=base_url,
        api_key=key.get_secret_value(),
        model=es.judge_model,
        rpm=es.judge_rpm,
        max_retries=s.llm_max_retries,
        timeout_s=s.llm_timeout_s,
        extra_params=extra,
    )


@dataclass
class Clients:
    answer: EvalLLM
    sleep: EvalLLM | None
    judge: EvalLLM
    embedder: Embedder

    async def aclose(self) -> None:
        for c in (self.answer, self.sleep, self.judge):
            if c is not None:
                await c.aclose()


def build_clients(
    es: EvalSettings, s: Settings, *, fake: bool, need_sleep: bool, cache: ResponseCache | None
) -> Clients:
    from app.llm.tracing import Tracer

    tracer = Tracer()  # eval keeps its own logs; don't spend the Langfuse quota
    if fake:
        answer: LLMClient = FakeLLM("fake-answer")
        sleep: LLMClient | None = FakeLLM("fake-sleep")
        judge: LLMClient = FakeLLM("fake-judge")
        embedder: Embedder = HashEmbedder(s.embedding_dim)
    else:
        answer = (
            build_online_client(s, tracer)
            if es.answer_provider == "online"
            else build_sleep_client(s, tracer)
        )
        sleep = build_sleep_client(s, tracer) if need_sleep else None
        judge = build_judge_client(es, s)
        embedder = build_embedder(s)
    return Clients(
        answer=EvalLLM(answer, cache=cache, extra_key=_extras(answer)),
        sleep=EvalLLM(sleep, cache=cache, extra_key=_extras(sleep)) if sleep else None,
        judge=EvalLLM(judge, cache=cache, extra_key=_extras(judge)),
        embedder=CachedEmbedder(embedder),
    )


def _extras(llm: LLMClient | None) -> dict[str, Any]:
    """Provider extras (e.g. reasoning_effort) change outputs, so they're part of the key."""
    return dict(getattr(llm, "_extra", {}) or {})


def build_systems(
    names: Sequence[str],
    *,
    es: EvalSettings,
    s: Settings,
    clients: Clients,
    db: EvalDatabase | None,
    pool: Any,
) -> list[System]:
    out: list[System] = []
    for name in names:
        if name == "full-context":
            out.append(
                FullContextSystem(
                    answer_llm=clients.answer,
                    budget_tokens=es.full_context_tokens,
                    max_tokens=s.online_max_tokens,
                )
            )
        elif name == "naive-rag":
            out.append(
                NaiveRagSystem(
                    answer_llm=clients.answer,
                    embedder=clients.embedder,
                    top_k=s.retrieval_top_k,
                    max_tokens=s.online_max_tokens,
                    budget_chars=s.context_budget_chars,
                    item_max_chars=s.context_item_max_chars,
                )
            )
        elif name in NOCTURNE_SYSTEMS:
            assert db is not None and pool is not None
            common: dict[str, Any] = {
                "settings": s,
                "db": db,
                "pool": pool,
                "embedder": clients.embedder,
                "answer_llm": clients.answer,
            }
            if name == "nocturne-no-sleep":
                out.append(nocturne_no_sleep(**common))
            else:
                out.append(
                    NocturneSystem(
                        name,
                        sleep=True,
                        sleep_llm=clients.sleep,
                        sleep_every=es.sleep_every_sessions,
                        **common,
                    )
                )
        else:
            raise ValueError(f"unknown system {name!r}; choose from {ALL_SYSTEMS}")
    return out


# --- Run bookkeeping -------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def run_fingerprint(cfg: RunConfig, es: EvalSettings, s: Settings, clients: Clients) -> dict:
    """Everything that changes results. `limit` and `systems` are left out on purpose:
    raising --limit only adds questions (stratified picks are nested), and adding a system
    only adds rows."""
    return {
        "dataset": cfg.dataset,
        "split": cfg.split,
        "categories": sorted(cfg.categories),
        "seed": cfg.seed,
        "fake": cfg.fake,
        "answer_model": f"{clients.answer.name}:{clients.answer.model}",
        "sleep_model": f"{clients.sleep.name}:{clients.sleep.model}" if clients.sleep else None,
        "judge_model": f"{clients.judge.name}:{clients.judge.model}",
        "embedding": "hash" if cfg.fake else f"{s.embedding_backend}:{s.embedding_model}",
        "full_context_tokens": es.full_context_tokens,
        "sleep_every_sessions": es.sleep_every_sessions,
        "backend": {
            k: getattr(s, k)
            for k in (
                "retrieval_top_k",
                "retrieval_theme_k",
                "retrieval_theme_decay",
                "retrieval_link_decay",
                "retrieval_expand_from",
                "retrieval_temporal",
                "retrieval_temporal_weight",
                "retrieval_fallback_min_score",
                "retrieval_fact_slots",
                "retrieval_fact_slot_ratio",
                "context_budget_chars",
                "context_item_max_chars",
                "retrieval_candidate_pool",
                "recent_turns",
                "score_alpha",
                "score_beta",
                "score_gamma",
                "score_delta",
                "recency_tau_days",
                "reinforce_eta",
                "online_max_tokens",
                "online_reasoning_effort",
                "sleep_max_tokens",
                "sleep_reasoning_effort",
                "extract_batch_chars",
                "extract_batch_episodes",
                "extract_assistant_chars",
                "judge_batch_groups",
                "neighbors_k",
                "dedup_auto_sim",
                "dedup_judge_sim",
                "conflict_sim",
                "enable_dedup",
                "enable_contradiction",
                "enable_linker",
                "link_sim",
                "enable_abstractor",
                "theme_sim",
                "theme_min_facts",
                "theme_match_overlap",
                "theme_max_per_cycle",
                "theme_batch_clusters",
                "theme_facts_shown",
                "theme_max_facts",
                "reinforce_kappa",
                "enable_forgetter",
                "forget_episode_theta",
                "forget_fact_theta",
                "forget_max_access",
                "enable_auditor",
                "audit_sample",
            )
        },
    }


def check_config(run_dir: Path, fingerprint: dict, *, allow_change: bool) -> None:
    path = run_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8")).get("fingerprint")
        if old is not None and old != fingerprint and not allow_change:
            changed = sorted(
                k for k in set(old) | set(fingerprint) if old.get(k) != fingerprint.get(k)
            )
            raise RunAborted(
                f"{run_dir.name} was run with a different configuration (changed: "
                f"{', '.join(changed)}). Mixing them would make the numbers meaningless. "
                "Use a new --run-name, or --allow-config-change if you're sure."
            )


def write_config(run_dir: Path, cfg: RunConfig, fingerprint: dict, n_questions: int) -> None:
    payload = {
        "run": asdict(cfg),
        "questions_selected": n_questions,
        "fingerprint": fingerprint,
        "fingerprint_sha": hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
        "git_commit": _git_commit(),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()

    def write(self, row: dict[str, Any]) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def completed(run_dir: Path) -> set[tuple[str, str]]:
    from eval.report import read_jsonl

    done: set[tuple[str, str]] = set()
    for r in read_jsonl(run_dir / "questions.jsonl"):
        key = (r["system"], r["question_id"])
        if r["status"] == "ok":
            done.add(key)
        else:
            done.discard(key)
    return done


def quota_error(exc: BaseException) -> QuotaExhaustedError | None:
    """The daily-quota error behind `exc`, if any (sleep failures arrive wrapped)."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, QuotaExhaustedError):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


# --- The loop --------------------------------------------------------------------------------


async def run(
    conversations: Sequence[Conversation],
    systems: Sequence[System],
    *,
    clients: Clients,
    judge: Judge,
    run_dir: Path,
    dataset: str,
    split_of: Any,
    max_errors: int,
    log: Any = print,
) -> None:
    done = completed(run_dir)
    qrows = JsonlWriter(run_dir / "questions.jsonl")
    crows = JsonlWriter(run_dir / "conversations.jsonl")
    total = sum(len(c.questions) for c in conversations) * len(systems)
    progress = len(
        [1 for c in conversations for q in c.questions for s in systems if (s.name, q.id) in done]
    )
    streak = 0

    def failure(what: str, exc: BaseException) -> None:
        nonlocal streak
        streak += 1
        log(f"  ! {what}: {_err(exc)}")
        quota = quota_error(exc)
        if quota is not None:
            raise RunAborted(
                f"{quota} Rows so far are saved; rerun the same command once it resets "
                "(cached calls are free)."
            ) from exc
        if streak >= max_errors:
            raise RunAborted(
                f"{streak} failures in a row (last: {_err(exc)}). Rows so far are saved; "
                "rerun the same command to resume (cached calls are free)."
            ) from exc

    try:
        for conv in conversations:
            for system in systems:
                todo = [q for q in conv.questions if (system.name, q.id) not in done]
                if not todo:
                    continue
                sleep_before = clients.sleep.meter.snapshot() if clients.sleep else Meter()
                try:
                    prep = await system.prepare(conv)
                except Exception as exc:
                    crows.write(
                        {
                            "system": system.name,
                            "dataset": dataset,
                            "conversation_id": conv.id,
                            "status": "error",
                            "error": _err(exc),
                        }
                    )
                    for q in todo:
                        qrows.write(
                            _question_row(
                                system.name, dataset, split_of, q, status="error", error=_err(exc)
                            )
                        )
                    failure(f"{system.name} prepare {conv.id}", exc)
                    continue
                sleep_used = (
                    clients.sleep.meter.snapshot() - sleep_before if clients.sleep else Meter()
                )
                crows.write(
                    {
                        "system": system.name,
                        "dataset": dataset,
                        "conversation_id": conv.id,
                        "episodes": prep.episodes,
                        "memory_items": prep.memory_items,
                        "memory_chars": prep.memory_chars,
                        "memory_episode": prep.by_tier.get("episode", 0),
                        "memory_fact": prep.by_tier.get("fact", 0),
                        "memory_theme": prep.by_tier.get("theme", 0),
                        "sleep_cycles": prep.sleep_cycles,
                        "facts_created": prep.facts_created,
                        "merges": prep.merges,
                        "supersessions": prep.supersessions,
                        "links": prep.links,
                        "archived": prep.archived,
                        "audit_score": prep.audit_score,
                        "audit_items": prep.audit_items,
                        "prep_seconds": round(prep.seconds, 2),
                        "sleep_tokens": sleep_used.tokens,
                        "sleep_calls": sleep_used.calls,
                        "sleep_cache_hits": sleep_used.cache_hits,
                        "sleep_calls_by_stage": sleep_used.stages() or None,
                        "truncated_episodes": getattr(system, "truncated_episodes", None),
                        "status": "ok",
                        "error": None,
                    }
                )

                for q in todo:
                    progress += 1
                    t0 = time.perf_counter()
                    a0, j0 = clients.answer.meter.snapshot(), clients.judge.meter.snapshot()
                    try:
                        ans = await system.answer(q)
                        verdict = await judge.grade(q, ans.text)
                    except Exception as exc:
                        qrows.write(
                            _question_row(
                                system.name, dataset, split_of, q, status="error", error=_err(exc)
                            )
                        )
                        failure(f"{system.name} {q.id}", exc)
                        continue
                    streak = 0
                    row = _question_row(system.name, dataset, split_of, q, status="ok")
                    row.update(
                        answer=ans.text,
                        correct=verdict.correct,
                        rubric=verdict.rubric,
                        judge_raw=verdict.raw[:500],
                        retrieval_ms=None
                        if ans.retrieval_ms is None
                        else round(ans.retrieval_ms, 2),
                        retrieved=ans.retrieved,
                        retrieved_via=ans.retrieved_via,
                        context_chars=ans.context_chars,
                        answer_tokens=(clients.answer.meter.snapshot() - a0).tokens,
                        judge_tokens=(clients.judge.meter.snapshot() - j0).tokens,
                    )
                    qrows.write(row)
                    mark = {True: "ok", False: "x", None: "?"}[verdict.correct]
                    log(
                        f"[{progress}/{total}] {system.name:<18} {q.category:<26} "
                        f"{mark:<2} {q.id} ({time.perf_counter() - t0:.1f}s)"
                    )
    finally:
        qrows.close()
        crows.close()


def _question_row(
    system: str, dataset: str, split_of: Any, q: Any, *, status: str, error: str | None = None
) -> dict[str, Any]:
    return {
        "system": system,
        "dataset": dataset,
        "split": split_of(q.id),
        "conversation_id": q.conversation_id,
        "question_id": q.id,
        "category": q.category,
        "question_type": q.question_type,
        "abstention": q.abstention,
        "question": q.question,
        "gold": q.answer,
        "answer": None,
        "correct": None,
        "status": status,
        "error": error,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
