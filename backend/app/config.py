"""Application settings, loaded from environment variables or a `.env` file in backend/."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.memory.scoring import ScoreWeights


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Core -------------------------------------------------------------------------------
    database_url: str = Field(..., description="Postgres URL (Supabase session pooler or Neon).")
    api_key: SecretStr | None = Field(None, description="If set, required in the X-API-Key header.")
    db_pool_min: int = 1
    db_pool_max: int = 5

    # --- Online LLM (Groq, OpenAI-compatible) -------------------------------------------------
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Groq's recommended replacement for llama-3.3-70b-versatile (retired 2026-08-16).
    # Alternative: openai/gpt-oss-20b (faster and cheaper, weaker).
    online_model: str = "openai/gpt-oss-120b"
    # For reasoning models (gpt-oss): low keeps chat latency down. Set to "" for
    # non-reasoning models, which reject the parameter.
    online_reasoning_effort: Literal["low", "medium", "high", ""] = "low"
    online_rpm: int = 30
    online_max_tokens: int = 1500  # leaves room for reasoning tokens

    # --- Sleep LLM (used from Phase 2) --------------------------------------------------------
    sleep_provider: Literal["gemini", "openrouter"] = "gemini"
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    sleep_model: str = "gemini-2.5-flash"
    sleep_rpm: int = 8

    llm_max_retries: int = 5
    llm_timeout_s: float = 60.0

    # --- Embeddings ---------------------------------------------------------------------------
    # "hash" is a deterministic, non-semantic embedder for tests/CI only.
    embedding_backend: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # must match vector(N) in migrations/001_init.sql

    # --- Tracing (Langfuse); disabled when keys are absent ------------------------------------
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # --- Retrieval & scoring (spec §5) --------------------------------------------------------
    retrieval_top_k: int = 8
    retrieval_candidate_pool: int = 40
    recent_turns: int = 4
    score_alpha: float = 0.6
    score_beta: float = 0.2
    score_gamma: float = 0.1
    score_delta: float = 0.1
    recency_tau_days: float = 30.0
    reinforce_eta: float = 0.2

    @property
    def score_weights(self) -> ScoreWeights:
        return ScoreWeights(
            alpha=self.score_alpha,
            beta=self.score_beta,
            gamma=self.score_gamma,
            delta=self.score_delta,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # database_url comes from env
