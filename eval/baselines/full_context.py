"""Full-context baseline: put as much of the raw history as fits into one prompt.

"Fits" is a token budget, not the model's context window: free-tier tokens-per-minute caps
can be far below the window (Groq's free gpt-oss-120b allows 8K tokens per minute, so a
single bigger request is rejected). When the history doesn't fit, the *oldest* turns are
dropped, and `truncated_episodes` in the results says how many. A run with a budget far
below the history size is a "recent history" baseline, not an upper bound: raise
EVAL_FULL_CONTEXT_TOKENS with a large-context answer model for the real reference.
"""

from __future__ import annotations

import time

from app.llm.client import LLMClient
from eval.base import PrepStats, SystemAnswer, answer_with_context
from eval.datasets.schema import Conversation, Episode, Question

CHARS_PER_TOKEN = 4  # rough English average; good enough for a budget


def render(episodes: list[Episode]) -> list[str]:
    return [
        f"[M{i}] ({e.at:%Y-%m-%d %H:%M} UTC, {e.at:%A}) {e.content}"
        for i, e in enumerate(episodes, start=1)
    ]


def fit_recent(lines: list[str], budget_tokens: int) -> tuple[list[str], int]:
    """Keep the newest lines whose total size fits the budget; returns (kept, dropped)."""
    budget = budget_tokens * CHARS_PER_TOKEN
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    kept.reverse()
    return kept, len(lines) - len(kept)


class FullContextSystem:
    def __init__(
        self,
        *,
        answer_llm: LLMClient,
        budget_tokens: int,
        max_tokens: int,
        name: str = "",
        prompt: str = "v1",
        temperature: float = 0.2,
    ) -> None:
        self.name = name or "full-context"
        self._llm = answer_llm
        self._budget = budget_tokens
        self._max_tokens = max_tokens
        self._prompt = prompt
        self._temperature = temperature
        self._context = ""
        self.truncated_episodes = 0

    async def prepare(self, conv: Conversation) -> PrepStats:
        t0 = time.perf_counter()
        episodes = conv.episodes()
        lines = render(episodes)
        kept, self.truncated_episodes = fit_recent(lines, self._budget)
        self._context = (
            "Conversation history (oldest first):\n" + "\n".join(kept)
            if kept
            else "(no memories yet)"
        )
        return PrepStats(
            episodes=len(episodes),
            memory_items=len(kept),
            memory_chars=sum(len(e.content) for e in episodes),
            by_tier={"episode": len(episodes)},
            seconds=time.perf_counter() - t0,
        )

    async def answer(self, q: Question) -> SystemAnswer:
        text = await answer_with_context(
            self._llm,
            q.question,
            self._context,
            now=q.asked_at,
            max_tokens=self._max_tokens,
            prompt=self._prompt,
            temperature=self._temperature,
        )
        return SystemAnswer(text=text, context_chars=len(self._context))

    async def aclose(self) -> None:
        pass
