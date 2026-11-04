"""Factories for the online (Groq) and sleep (Gemini / OpenRouter) clients."""

from __future__ import annotations

from app.config import Settings
from app.llm.client import OpenAICompatClient
from app.llm.tracing import Tracer


class ConfigError(RuntimeError):
    pass


def build_online_client(settings: Settings, tracer: Tracer) -> OpenAICompatClient:
    if settings.groq_api_key is None:
        raise ConfigError("GROQ_API_KEY is not set (needed for /chat).")
    extra: dict[str, object] = {}
    if settings.online_reasoning_effort:
        # Groq returns gpt-oss reasoning in message.reasoning; we never use it, so skip it.
        extra = {"reasoning_effort": settings.online_reasoning_effort, "include_reasoning": False}
    return OpenAICompatClient(
        name="groq",
        base_url=settings.groq_base_url,
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.online_model,
        rpm=settings.online_rpm,
        tracer=tracer,
        max_retries=settings.llm_max_retries,
        timeout_s=settings.llm_timeout_s,
        extra_params=extra,
    )


def build_sleep_client(settings: Settings, tracer: Tracer) -> OpenAICompatClient:
    if settings.sleep_provider == "gemini":
        key, base_url = settings.gemini_api_key, settings.gemini_base_url
        env_name = "GEMINI_API_KEY"
    else:
        key, base_url = settings.openrouter_api_key, settings.openrouter_base_url
        env_name = "OPENROUTER_API_KEY"
    if key is None:
        raise ConfigError(f"{env_name} is not set (needed for sleep cycles).")
    extra: dict[str, object] = {}
    if settings.sleep_reasoning_effort:
        extra["reasoning_effort"] = settings.sleep_reasoning_effort
    return OpenAICompatClient(
        name=settings.sleep_provider,
        base_url=base_url,
        api_key=key.get_secret_value(),
        model=settings.sleep_model,
        rpm=settings.sleep_rpm,
        tracer=tracer,
        max_retries=settings.llm_max_retries,
        timeout_s=max(settings.llm_timeout_s, 120.0),  # batch prompts are long
        extra_params=extra,
    )
