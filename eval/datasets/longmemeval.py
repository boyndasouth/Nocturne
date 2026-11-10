"""LongMemEval loader (Wu et al., 2024; https://github.com/xiaowu0162/LongMemEval).

Each instance is one question with its own chat history ("haystack"), so it becomes one
Conversation with one Question. Files:
- longmemeval_oracle.json: only the evidence sessions (cheap; use first)
- longmemeval_s_cleaned.json: ~50 sessions (~115k tokens) per question

Dates look like "2023/05/20 (Sat) 02:21" and have no zone; they're read as UTC. Oracle
sessions are not sorted by date, so they're sorted here.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.datasets.jsonstream import iter_json_array
from eval.datasets.schema import Conversation, Question, Session, Turn

BENCHMARK = "longmemeval"
CATEGORIES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
    "abstention",
)

_WEEKDAY_RE = re.compile(r"\s*\([A-Za-z]{3}\)\s*")
_DATE_FORMATS = ("%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d %H:%M")


def parse_date(raw: str) -> datetime:
    text = _WEEKDAY_RE.sub(" ", raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"unrecognised LongMemEval date: {raw!r}") from None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def parse_instance(item: dict[str, Any]) -> Conversation:
    qid = str(item["question_id"])
    abstention = qid.endswith("_abs")
    qtype = str(item["question_type"])

    sessions = [
        Session(
            id=str(sid),
            date=parse_date(date),
            turns=tuple(
                Turn(
                    speaker=str(t["role"]),
                    text=str(t.get("content") or ""),
                    evidence=bool(t.get("has_answer")),
                )
                for t in turns
            ),
        )
        for sid, date, turns in zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
            strict=True,
        )
    ]
    sessions.sort(key=lambda s: (s.date, s.id))

    question = Question(
        id=qid,
        conversation_id=qid,
        question=str(item["question"]),
        answer=str(item["answer"]),
        category="abstention" if abstention else qtype,
        question_type=qtype,
        asked_at=parse_date(str(item["question_date"])),
        abstention=abstention,
        evidence=tuple(str(s) for s in item.get("answer_session_ids") or ()),
    )
    return Conversation(
        id=qid, benchmark=BENCHMARK, style="chat", sessions=tuple(sessions), questions=(question,)
    )


def load(path: str | Path) -> list[Conversation]:
    """Streamed one instance at a time: LongMemEval-S needs ~400 MB this way instead of
    ~2.4 GB with json.load (which can raise MemoryError on a laptop)."""
    return [parse_instance(item) for item in iter_json_array(path)]
