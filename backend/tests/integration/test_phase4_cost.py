"""Phase 4: a sleep cycle whose extraction batch is too big for one answer splits the batch
and still consolidates everything, and the failed attempt is counted."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.llm.client import ChatMessage, LLMResponse
from app.llm.prompts import EXTRACTOR_SYSTEMS
from tests.integration.conftest import FakeLLM
from tests.integration.test_phase2 import JAN, JAN_SLEEP, chat, facts, sleep, sleep_brain


class TruncatingLLM(FakeLLM):
    """The scripted sleep brain, except an extraction prompt with more than `max_episodes`
    episodes comes back cut off at max_tokens."""

    def __init__(self, max_episodes: int) -> None:
        super().__init__(sleep_brain)
        self.max_episodes = max_episodes
        self.batch_sizes: list[int] = []

    async def chat(
        self, messages: Sequence[ChatMessage], *, trace_name: str = "llm.chat", **kw: object
    ) -> LLMResponse:
        if messages[0]["content"] in EXTRACTOR_SYSTEMS:
            n = len(re.findall(r"^\[\d+\] \(", messages[-1]["content"], re.M))
            self.batch_sizes.append(n)
            if n > self.max_episodes:
                self.trace_names.append(trace_name)
                return LLMResponse(
                    text='{"facts": [{"statement": "The user', model=self.model,
                    input_tokens=10, output_tokens=5, latency_ms=1.0, finish_reason="length",
                )  # fmt: skip
        return await super().chat(messages, trace_name=trace_name, **kw)


def test_oversized_extraction_batch_is_split_and_nothing_is_lost(make_client, fresh_db):
    sleep_llm = TruncatingLLM(max_episodes=2)
    client = make_client(sleep_llm=sleep_llm)
    for i, msg in enumerate(["I live in Austin", "hello", "My exam is on Friday", "thanks"]):
        chat(client, msg, JAN.replace(minute=i))

    r = sleep(client, JAN_SLEEP)

    assert r["status"] == "succeeded"
    # one batch of 4 -> cut off -> two batches of 2
    assert sleep_llm.batch_sizes == [4, 2, 2]
    assert sorted(f["content"] for f in facts(fresh_db)) == [
        "The user lives in Austin.",
        "The user's linear algebra exam is on Friday.",
    ]
    assert r["cycle"]["episodes_processed"] == 4
    # the truncated call is part of the cycle's cost
    assert r["cycle"]["llm_calls"] == len(sleep_llm.trace_names)


def test_single_episode_that_cannot_fit_fails_the_cycle(make_client, fresh_db):
    sleep_llm = TruncatingLLM(max_episodes=0)
    client = make_client(sleep_llm=sleep_llm)
    chat(client, "I live in Austin", JAN)
    chat(client, "My exam is on Friday", JAN.replace(minute=1))

    r = sleep(client, JAN_SLEEP)

    assert r["status"] == "failed" and "cut off" in r["cycle"]["error"]
    assert sleep_llm.batch_sizes == [2, 1]  # split once, then the single episode gives up
    assert facts(fresh_db) == []
