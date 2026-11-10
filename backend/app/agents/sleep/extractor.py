"""Extractor (spec §3.2, step 1): episodes -> atomic fact candidates with provenance."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime

from pydantic import BaseModel, Field, field_validator

from app.agents.sleep.state import EpisodeIn, FactCandidate
from app.llm.client import LLMClient
from app.llm.prompts import EXTRACTOR_USER, extractor_system
from app.llm.structured import Usage, chat_json

MAX_FACT_CHARS = 500
_EPISODE_PREVIEW_CHARS = 4000
# How the Episode Writer lays out a chat turn (app.agents.episode_writer.format_turn).
ASSISTANT_MARKER = "\nAssistant: "
ASSISTANT_PREFIX = "Assistant: "


class _ExtractedFact(BaseModel):
    statement: str
    keywords: list[str] = Field(default_factory=list)
    sources: list[int] = Field(default_factory=list)
    valid_from: str | None = None

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, v: object) -> object:
        return [] if v is None else v

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, int):
            return [v]
        return v


class ExtractionOutput(BaseModel):
    facts: list[_ExtractedFact] = Field(default_factory=list)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # End on a word boundary when one is reasonably close.
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " …"


def _trim_assistant(content: str, assistant_chars: int) -> str:
    if assistant_chars <= 0:
        return content
    if content.startswith(ASSISTANT_PREFIX):
        return ASSISTANT_PREFIX + _clip(content[len(ASSISTANT_PREFIX) :], assistant_chars)
    user, sep, reply = content.partition(ASSISTANT_MARKER)
    return user + sep + _clip(reply, assistant_chars) if sep else content


def episode_text(content: str, *, assistant_chars: int = 0) -> str:
    """What the extractor sees of one episode.

    Facts come only from what the user said (EXTRACTOR_SYSTEM); the assistant's reply is there
    to make the user's words understandable ("I'll take the second one"). On chat benchmarks
    the replies are ~85% of the text, so `assistant_chars` > 0 keeps just their opening. The
    stored episode is untouched: retrieval and the Responder still see the full reply.
    Episodes without an "Assistant:" part (notes, dialogue) are only capped in length.

    Trimming off reproduces Phase 3's prompt text byte for byte, and trimming on reproduces
    Phase 4.1's, so runs made with either replay from the LLM cache."""
    text = _trim_assistant(content, assistant_chars)
    if assistant_chars > 0:
        return _clip(text, _EPISODE_PREVIEW_CHARS)
    if len(text) > _EPISODE_PREVIEW_CHARS:
        text = text[:_EPISODE_PREVIEW_CHARS].rstrip() + " …"
    return text


def batch_episodes(
    episodes: Sequence[EpisodeIn], *, max_chars: int, max_items: int, assistant_chars: int = 0
) -> list[list[EpisodeIn]]:
    """Consecutive batches under both limits, measured on the text the prompt will hold
    (Phase 3, untrimmed: on the raw text capped at the preview size). An oversized episode
    gets a batch to itself (and is truncated in the prompt)."""
    batches: list[list[EpisodeIn]] = []
    current: list[EpisodeIn] = []
    size = 0
    for ep in episodes:
        n = (
            len(episode_text(ep.content, assistant_chars=assistant_chars))
            if assistant_chars > 0
            else min(len(ep.content), _EPISODE_PREVIEW_CHARS)
        )
        if current and (size + n > max_chars or len(current) >= max_items):
            batches.append(current)
            current, size = [], 0
        current.append(ep)
        size += n
    if current:
        batches.append(current)
    return batches


def format_episodes(batch: Sequence[EpisodeIn], *, assistant_chars: int = 0) -> str:
    lines = []
    for i, ep in enumerate(batch, start=1):
        text = episode_text(ep.content, assistant_chars=assistant_chars)
        lines.append(f"[{i}] ({ep.valid_from:%Y-%m-%d %H:%M} UTC, {ep.valid_from:%A})\n{text}")
    return "\n\n".join(lines)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        d = date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def to_candidates(
    output: ExtractionOutput, batch: Sequence[EpisodeIn], batch_index: int
) -> list[FactCandidate]:
    """Validate the model's output against the batch:
    - drop facts without a valid source episode (no provenance, no memory),
    - valid_from defaults to the earliest source episode. An explicit date from the model
      ("I moved in March") is used only if it isn't after the latest source episode: the
      model may backdate a fact the user dated, but not start one in the future,
    - drop exact duplicates within the batch."""
    out: list[FactCandidate] = []
    seen: set[str] = set()
    for i, fact in enumerate(output.facts):
        content = " ".join(fact.statement.split())[:MAX_FACT_CHARS]
        if not content or content.lower() in seen:
            continue
        sources = [batch[s - 1] for s in dict.fromkeys(fact.sources) if 1 <= s <= len(batch)]
        if not sources:
            continue
        earliest = min(ep.valid_from for ep in sources)
        latest = max(ep.valid_from for ep in sources)
        explicit = _parse_date(fact.valid_from)
        valid_from = explicit if explicit is not None and explicit <= latest else earliest
        keywords = list(dict.fromkeys(k.strip().lower() for k in fact.keywords if k.strip()))[:8]
        seen.add(content.lower())
        out.append(
            FactCandidate(
                key=f"{batch_index}.{i}",
                content=content,
                keywords=keywords,
                valid_from=valid_from,
                source_episode_ids=[ep.id for ep in sources],
            )
        )
    return out


async def extract_batch(
    llm: LLMClient,
    batch: Sequence[EpisodeIn],
    batch_index: int,
    *,
    max_tokens: int,
    assistant_chars: int = 0,
    prompt: str = "v1",
) -> tuple[list[FactCandidate], Usage]:
    output, usage = await chat_json(
        llm,
        [
            {
                "role": "system",
                "content": extractor_system(trimmed=assistant_chars > 0, version=prompt),
            },
            {
                "role": "user",
                "content": EXTRACTOR_USER.format(
                    episodes=format_episodes(batch, assistant_chars=assistant_chars)
                ),
            },
        ],
        ExtractionOutput,
        trace_name="sleep.extract",
        max_tokens=max_tokens,
    )
    return to_candidates(output, batch, batch_index), usage
