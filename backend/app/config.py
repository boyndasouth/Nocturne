"""Application settings, loaded from environment variables or a `.env` file in backend/."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.memory.scoring import ScoreWeights

if TYPE_CHECKING:
    from app.memory.retriever import RetrievalConfig


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
    # ~500 free requests/day vs ~20 for the Flash models (per-model limits; check AI Studio).
    sleep_model: str = "gemini-3.5-flash-lite"
    sleep_rpm: int = 8
    sleep_max_tokens: int = 8192  # JSON outputs for a whole batch, plus any thinking tokens
    # Passed as `reasoning_effort` when set (Gemini's OpenAI endpoint accepts it).
    sleep_reasoning_effort: Literal["none", "low", "medium", "high", ""] = ""

    llm_max_retries: int = 5
    llm_timeout_s: float = 60.0

    # --- Embeddings ---------------------------------------------------------------------------
    # "hash" is a deterministic, non-semantic embedder for tests/CI only.
    embedding_backend: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str | None = None  # default: fastembed's cache in the temp dir
    embedding_dim: int = 384  # must match vector(N) in migrations/001_init.sql

    # --- Tracing (Langfuse); disabled when keys are absent ------------------------------------
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # --- Retrieval & scoring (spec §5) --------------------------------------------------------
    retrieval_top_k: int = 8
    retrieval_candidate_pool: int = 40
    # Hierarchical retrieval (spec §5): themes first, then member facts and one link hop.
    retrieval_theme_k: int = 2
    retrieval_theme_decay: float = 0.85  # a theme's member facts inherit its score x this
    retrieval_link_decay: float = 0.7  # a linked fact inherits the linking fact's score x this
    retrieval_expand_from: int = 8  # top facts whose links are followed
    retrieval_temporal: bool = True  # parse "in March", "last week", ... into a time window
    retrieval_temporal_weight: float = 0.15
    retrieval_fallback_min_score: float = 0.45  # weaker best match -> search archived turns
    # At least this many of the top_k slots go to facts, when there are facts scoring at
    # least `ratio` x the best result (so a question can't be answered from raw turns alone).
    retrieval_fact_slots: int = 3
    retrieval_fact_slot_ratio: float = 0.6
    # Phase 4 options (see app/memory/retriever.py). Dev run Nov 2026: extra facts fixed a
    # question and broke none; split had nothing to fix on dev but costs nothing when unused.
    retrieval_extra_facts: int = 10  # more facts after the top k, for counting questions
    retrieval_split_queries: bool = True  # search each side of "A or B" separately too
    retrieval_split_slots: int = 2
    # Dev at temperature 0: 80% vs 75% (the plants counting question fixed, none broken).
    retrieval_strip_time_phrase: bool = True  # search without "in the last month" etc.
    # v2 adds near-miss, counting and comparison rules. Dev at temperature 0: 75% vs 70%
    # for v1 (one counting question fixed, none broken). Baselines use the same version.
    responder_prompt: Literal["v1", "v2", "v3"] = "v2"  # v3: + advice/recommendation rules
    responder_temperature: float = 0.2  # the eval answers at EVAL_ANSWER_TEMPERATURE (0)
    # Memory block given to the answer model; the eval's naive-RAG baseline gets the same.
    context_budget_chars: int = 12_000
    context_item_max_chars: int = 3_000
    recent_turns: int = 4
    score_alpha: float = 0.6
    score_beta: float = 0.2
    score_gamma: float = 0.1
    score_delta: float = 0.1
    recency_tau_days: float = 30.0
    reinforce_eta: float = 0.2
    reinforce_kappa: float = 1.5  # τ grows by this on every access (spacing effect)
    # Forgetter (spec §3.2 step 6). Notes start at τ = 30 days; untouched, a consolidated
    # episode is archived after ~1 τ, an unused fact outside any theme after ~5 τ.
    enable_forgetter: bool = True
    forget_episode_theta: float = 0.37
    forget_fact_theta: float = 0.0067
    forget_max_access: int = 1
    # Auditor (step 7): an LLM judge spot-checks this many of each cycle's decisions.
    enable_auditor: bool = True
    audit_sample: int = 8

    # --- Sleep cycle (spec §3.2) -------------------------------------------------------------
    sleep_auto: bool = True  # in-process idle/backlog trigger
    sleep_idle_minutes: float = 10.0  # (a) idle for N minutes ...
    sleep_idle_min_episodes: int = 3  # ... with at least this many pending episodes
    sleep_backlog_episodes: int = 30  # (b) M pending episodes triggers regardless of idle
    sleep_poll_seconds: float = 60.0
    sleep_retry_backoff_minutes: float = 10.0  # after a failed cycle; doubles each time
    sleep_max_episodes: int = 200  # per cycle; the rest wait for the next one
    sleep_max_attempts: int = 3  # resume a failed cycle at most this many times
    sleep_stale_minutes: float = 30.0  # a "running" cycle with no heartbeat is presumed dead
    sleep_recursion_limit: int = 500  # LangGraph steps; each batch is one step

    # Big batches keep a sleep cycle to a handful of calls (free tiers cap requests per day).
    # A batch whose answer is cut off at sleep_max_tokens is split in two and retried.
    extract_batch_chars: int = 40_000  # prompt chars per extraction call
    extract_batch_episodes: int = 25  # 60 dropped ~1 in 10 evidence turns (probe, Nov 2026)
    extract_assistant_chars: int = 400  # assistant reply shown to the Extractor; 0 = all
    extract_prompt: Literal["v1", "v2"] = "v2"  # v2: also extract facts said in passing
    judge_batch_groups: int = 40  # dedup / contradiction / linker groups per LLM call
    neighbors_k: int = 4
    # Cosine-similarity thresholds (tune on a dev split; see tests/unit/test_dedup.py).
    dedup_auto_sim: float = 0.97  # ≥ this: merge without asking the LLM
    dedup_judge_sim: float = 0.82  # ≥ this: ask the LLM whether it's a duplicate
    conflict_sim: float = 0.65  # ≥ this: ask the LLM whether the facts conflict
    # Ablation switches (spec §9).
    enable_dedup: bool = True
    enable_contradiction: bool = True
    # Linker (spec §3.2 step 4): new facts are linked to active facts at least this similar.
    enable_linker: bool = True
    link_sim: float = 0.70
    # Abstractor (step 5): average-linkage clusters of active facts become themes.
    enable_abstractor: bool = True
    theme_sim: float = 0.72  # average cosine similarity within a cluster
    theme_min_facts: int = 3
    theme_match_overlap: float = 0.5  # share of members for a cluster to keep its theme
    theme_max_per_cycle: int = 12  # themes (re)written per cycle; the rest wait
    theme_batch_clusters: int = 12  # clusters per LLM call
    theme_facts_shown: int = 25  # members the LLM sees per cluster
    theme_max_facts: int = 2000  # newest active facts considered for clustering

    @property
    def retrieval_config(self) -> RetrievalConfig:
        from app.memory.retriever import RetrievalConfig

        return RetrievalConfig(
            top_k=self.retrieval_top_k,
            candidate_pool=self.retrieval_candidate_pool,
            theme_k=self.retrieval_theme_k,
            theme_decay=self.retrieval_theme_decay,
            link_decay=self.retrieval_link_decay,
            expand_from=self.retrieval_expand_from,
            temporal=self.retrieval_temporal,
            temporal_weight=self.retrieval_temporal_weight,
            fallback_min_score=self.retrieval_fallback_min_score,
            fact_slots=self.retrieval_fact_slots,
            fact_slot_ratio=self.retrieval_fact_slot_ratio,
            extra_facts=self.retrieval_extra_facts,
            split_queries=self.retrieval_split_queries,
            split_slots=self.retrieval_split_slots,
            strip_time_phrase=self.retrieval_strip_time_phrase,
        )

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
