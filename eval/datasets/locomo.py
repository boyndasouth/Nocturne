"""LoCoMo loader (Maharana et al., 2024; https://github.com/snap-research/locomo).

locomo10.json has 10 long two-person dialogues (~19 sessions each) with ~200 QA pairs each.
Turns are between two people, not a user and an assistant, so conversations use the
"dialogue" style: one episode per utterance, "Name: text".

Category numbers aren't named in the file. This follows the mapping most follow-up work
uses (e.g. the Mem0 paper): 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
5 adversarial. Category 5 questions have no `answer`; the premise is false or unanswerable,
so they're scored as abstention.

Known gap: Nocturne's extractor is written for a user/assistant chat and extracts facts
"the user stated". On dialogues it needs a multi-speaker variant before its LoCoMo numbers
mean much; the baselines and the no-sleep system aren't affected.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from eval.datasets.schema import Conversation, Question, Session, Turn

BENCHMARK = "locomo"
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
CATEGORIES = tuple(CATEGORY_NAMES.values())

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def parse_date(raw: str) -> datetime:
    """'1:56 pm on 8 May, 2023' -> 2023-05-08 13:56 UTC."""
    text = " ".join(raw.replace(",", " ").split()).lower()
    for fmt in ("%I:%M %p on %d %B %Y", "%I:%M%p on %d %B %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"unrecognised LoCoMo date: {raw!r}")


def _turn_text(t: dict[str, Any]) -> str:
    text = str(t.get("text") or "").strip()
    caption = t.get("blip_caption")
    if caption:  # image turns: keep what the photo showed
        text = f"{text} [shares a photo: {caption}]".strip()
    return text


def parse_sample(item: dict[str, Any]) -> Conversation:
    conv_id = str(item["sample_id"])
    convo: dict[str, Any] = item["conversation"]

    sessions: list[Session] = []
    for key, turns in convo.items():
        m = _SESSION_KEY_RE.match(key)
        if not m or not turns:
            continue
        date_raw = convo.get(f"{key}_date_time")
        if not date_raw:
            continue
        sessions.append(
            Session(
                id=f"{conv_id}:S{m.group(1)}",
                date=parse_date(str(date_raw)),
                turns=tuple(Turn(speaker=str(t["speaker"]), text=_turn_text(t)) for t in turns),
            )
        )
    sessions.sort(key=lambda s: (s.date, int(s.id.rsplit("S", 1)[1])))
    if not sessions:
        raise ValueError(f"{conv_id}: no sessions")

    # No question dates in LoCoMo: ask the day after the last session.
    asked_at = sessions[-1].date + timedelta(days=1)
    questions: list[Question] = []
    for i, qa in enumerate(item.get("qa") or ()):
        cat = int(qa["category"])
        adversarial = cat == 5
        answer = qa.get("answer")
        if adversarial or answer is None:
            answer = (
                "The question has a false premise or isn't answered in the conversation. "
                f"(Distractor answer: {qa.get('adversarial_answer', 'n/a')})"
            )
        questions.append(
            Question(
                id=f"{conv_id}:q{i}",
                conversation_id=conv_id,
                question=str(qa["question"]),
                answer=str(answer),
                category=CATEGORY_NAMES.get(cat, f"category-{cat}"),
                question_type=str(cat),
                asked_at=asked_at,
                abstention=adversarial,
                evidence=tuple(str(e) for e in qa.get("evidence") or ()),
            )
        )
    return Conversation(
        id=conv_id,
        benchmark=BENCHMARK,
        style="dialogue",
        sessions=tuple(sessions),
        questions=tuple(questions),
    )


def load(path: str | Path) -> list[Conversation]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_sample(item) for item in data]
