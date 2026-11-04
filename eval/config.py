"""Eval settings. Read from the environment and backend/.env, with an EVAL_ prefix.

Backend knobs (retrieval weights, sleep thresholds, models) come from the normal backend
Settings, so an eval run measures exactly what the app would run. Three LLM roles, each on
its own free-tier quota bucket by default:

    answer  - the online model (Groq), shared by every system so only memory differs
    sleep   - the sleep model (Gemini / OpenRouter), only nocturne's sleep cycles use it
    judge   - a separate model from the answerer (default: Groq gpt-oss-20b; on Groq each
              model has its own limits, so judging doesn't eat the answer quota)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENV = ROOT / "backend" / ".env"


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EVAL_", env_file=BACKEND_ENV, env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str | None = Field(
        None,
        description="A disposable Postgres+pgvector database whose name contains 'eval'. It "
        "is wiped before every conversation. Required for the nocturne systems.",
    )

    answer_provider: Literal["online", "sleep"] = "online"

    judge_provider: Literal["groq", "gemini", "openrouter"] = "groq"
    judge_model: str = "openai/gpt-oss-20b"
    judge_reasoning_effort: Literal["low", "medium", "high", ""] = "low"
    judge_rpm: int = 25
    judge_max_tokens: int = 600

    # Raw-history budget for the full-context baseline (see baselines/full_context.py).
    # 5,000 keeps one request under Groq's free 8K tokens/minute with room for the answer.
    full_context_tokens: int = 5_000

    cache_path: Path = ROOT / "eval" / ".cache" / "llm_cache.sqlite"
    results_dir: Path = ROOT / "eval" / "results"
