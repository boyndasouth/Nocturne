"""JSON-mode chat with Pydantic validation and one repair round-trip.

Free-tier models occasionally wrap JSON in code fences, add a preamble, or miss a field. We
strip the noise, validate, and on failure send the validation error back once before giving
up with an InvalidJSONError (which fails the sleep step, so the cycle can resume later).

Output cut off at max_tokens is different: asking again would produce the same too-long
answer, so it raises OutputTruncatedError straight away. Callers that can shrink the request
(the Extractor halves its batch) catch InvalidJSONError and retry with less.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.client import ChatMessage, LLMClient, LLMError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class Usage:
    tokens: int = 0
    calls: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.tokens + other.tokens, self.calls + other.calls)


class InvalidJSONError(LLMError):
    """The model's reply didn't parse as the requested schema, even after a repair round.
    `usage` is what the failed attempts cost, so callers that retry can still count it."""

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or Usage()


class OutputTruncatedError(InvalidJSONError):
    """The reply hit max_tokens mid-JSON. A smaller request is the only fix."""


def extract_json(text: str) -> object:
    """Parse the first JSON object in `text`, tolerating code fences and surrounding prose."""
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


async def chat_json(
    llm: LLMClient,
    messages: Sequence[ChatMessage],
    model: type[T],
    *,
    trace_name: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> tuple[T, Usage]:
    convo = list(messages)
    usage = Usage()
    last_error = ""
    for attempt in range(2):
        resp = await llm.chat(
            convo,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            trace_name=trace_name if attempt == 0 else f"{trace_name}.repair",
        )
        usage += Usage(resp.input_tokens + resp.output_tokens, 1)
        try:
            return model.model_validate(extract_json(resp.text)), usage
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)[:1500]
            if resp.finish_reason == "length":
                raise OutputTruncatedError(
                    f"{trace_name}: output cut off at max_tokens={max_tokens} "
                    f"({resp.output_tokens} tokens)",
                    usage,
                ) from exc
            convo = [
                *messages,
                {"role": "assistant", "content": resp.text[:4000]},
                {
                    "role": "user",
                    "content": "That wasn't valid JSON for the requested schema. Error:\n"
                    f"{last_error}\nReply again with only the corrected JSON object.",
                },
            ]
    raise InvalidJSONError(f"{trace_name}: invalid JSON after repair: {last_error}", usage)
