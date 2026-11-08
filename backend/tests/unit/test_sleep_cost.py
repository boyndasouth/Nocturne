"""Phase 4 sleep cost: the Extractor sees trimmed assistant replies, batches are sized on
that text, and a reply cut off at max_tokens fails fast instead of paying for a repair."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.agents.sleep.extractor import batch_episodes, episode_text, format_episodes
from app.agents.sleep.state import EpisodeIn
from app.llm.client import LLMResponse
from app.llm.structured import InvalidJSONError, OutputTruncatedError, chat_json, extract_json

T0 = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)


def ep(content: str) -> EpisodeIn:
    return EpisodeIn(id=uuid4(), content=content, valid_from=T0)


REPLY = "word " * 200  # 1000 chars


def test_assistant_reply_is_trimmed_and_user_text_kept():
    text = episode_text(
        "User: I moved to Dallas last week\nAssistant: " + REPLY, assistant_chars=50
    )
    user, reply = text.split("\nAssistant: ")
    assert user == "User: I moved to Dallas last week"
    assert reply.endswith(" …") and len(reply) <= 52
    assert not reply[:-2].endswith(" ")  # cut on a word boundary


def test_trim_is_off_at_zero_and_leaves_other_episodes_alone():
    turn = "User: hi\nAssistant: " + REPLY
    assert episode_text(turn, assistant_chars=0) == turn
    note = "Lecture 3: attention. " * 20  # an ingested note has no assistant part
    assert episode_text(note, assistant_chars=50) == note
    dialogue = "Caroline: I went to a support group yesterday."
    assert episode_text(dialogue, assistant_chars=10) == dialogue
    # an assistant turn with no user turn before it
    lone = episode_text("Assistant: " + REPLY, assistant_chars=40)
    assert lone.startswith("Assistant: ") and len(lone) < 60


def test_short_replies_are_untouched():
    turn = "User: my exam is Friday\nAssistant: Good luck!"
    assert episode_text(turn, assistant_chars=400) == turn


def test_every_episode_is_capped_even_without_trimming():
    huge = "User: " + "x" * 9000
    assert len(episode_text(huge)) < 4100


def test_batches_are_sized_on_the_trimmed_text():
    eps = [ep("User: I live in Austin\nAssistant: " + REPLY) for _ in range(10)]
    # Full replies: ~1030 chars each, so 3 per 3,500-char batch.
    assert [len(b) for b in batch_episodes(eps, max_chars=3500, max_items=50)] == [3, 3, 3, 1]
    # Trimmed to 100 chars: ~130 each, all ten fit.
    trimmed = batch_episodes(eps, max_chars=3500, max_items=50, assistant_chars=100)
    assert [len(b) for b in trimmed] == [10]


def test_prompt_shows_trimmed_replies():
    prompt = format_episodes([ep("User: hi\nAssistant: " + REPLY)], assistant_chars=30)
    assert prompt.startswith("[1] (2026-03-02 14:00 UTC, Monday)\nUser: hi\nAssistant: word")
    assert len(prompt) < 120


class ScriptedLLM:
    name, model = "fake", "fake-model"

    def __init__(self, *replies: tuple[str, str | None]) -> None:
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, messages, **_: object) -> LLMResponse:
        self.calls += 1
        text, finish = self.replies.pop(0)
        return LLMResponse(
            text=text, model=self.model, input_tokens=100, output_tokens=50, latency_ms=1.0,
            finish_reason=finish,
        )  # fmt: skip

    async def aclose(self) -> None:
        pass


async def test_truncated_reply_raises_without_a_repair_call():
    from app.agents.sleep.extractor import ExtractionOutput

    llm = ScriptedLLM(('{"facts": [{"statement": "The user li', "length"))
    with pytest.raises(OutputTruncatedError) as err:
        await chat_json(
            llm, [{"role": "user", "content": "x"}], ExtractionOutput, trace_name="t", max_tokens=9
        )
    assert llm.calls == 1
    assert (err.value.usage.calls, err.value.usage.tokens) == (1, 150)


async def test_invalid_json_after_repair_reports_its_cost():
    from app.agents.sleep.extractor import ExtractionOutput

    llm = ScriptedLLM(("not json", "stop"), ("still not json", "stop"))
    with pytest.raises(InvalidJSONError) as err:
        await chat_json(
            llm, [{"role": "user", "content": "x"}], ExtractionOutput, trace_name="t", max_tokens=9
        )
    assert not isinstance(err.value, OutputTruncatedError)
    assert llm.calls == 2 and err.value.usage.calls == 2


async def test_valid_reply_marked_length_is_still_accepted():
    """A complete JSON object that happens to end at the limit is fine."""
    from app.agents.sleep.extractor import ExtractionOutput

    llm = ScriptedLLM(('{"facts": []}', "length"))
    out, usage = await chat_json(
        llm, [{"role": "user", "content": "x"}], ExtractionOutput, trace_name="t", max_tokens=9
    )
    assert out.facts == [] and usage.calls == 1
    assert extract_json('{"facts": []}') == {"facts": []}
