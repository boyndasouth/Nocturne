from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from app.memory.models import Tier
from app.memory.retriever import is_conversational, overlaps_window
from app.memory.temporal import TimeWindow, parse_time_window

NOW = datetime(2023, 6, 15, 12, tzinfo=UTC)  # a Thursday


def window(q: str, now: datetime = NOW) -> tuple[date, date] | None:
    w = parse_time_window(q, now)
    return None if w is None else (w.start.date(), w.end.date())


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Where did I live in March?", (date(2023, 3, 1), date(2023, 4, 1))),
        ("What did I do in December?", (date(2022, 12, 1), date(2023, 1, 1))),  # latest one
        ("What did I read during March 2021?", (date(2021, 3, 1), date(2021, 4, 1))),
        ("What did I do on May 5th, 2023?", (date(2023, 5, 5), date(2023, 5, 6))),
        ("What happened on 5 May?", (date(2023, 5, 5), date(2023, 5, 6))),
        ("What happened on 2023-04-02?", (date(2023, 4, 2), date(2023, 4, 3))),
        ("How many books did I read in 2022?", (date(2022, 1, 1), date(2023, 1, 1))),
        ("What did I buy two weeks ago?", (date(2023, 5, 25), date(2023, 6, 9))),
        ("What did I do 3 days ago?", (date(2023, 6, 11), date(2023, 6, 14))),
        ("What did I do last week?", (date(2023, 6, 1), date(2023, 6, 16))),
        ("What did I eat last month?", (date(2023, 5, 1), date(2023, 6, 1))),
        ("Anything this month?", (date(2023, 6, 1), date(2023, 7, 1))),
        ("What did I do last year?", (date(2022, 1, 1), date(2023, 1, 1))),
        ("What happened last Saturday?", (date(2023, 6, 10), date(2023, 6, 11))),
        ("Where was I yesterday?", (date(2023, 6, 14), date(2023, 6, 15))),
        ("Where did I live before May?", (date(1900, 1, 1), date(2023, 5, 1))),
        ("What have I learned since March 2023?", (date(2023, 3, 1), date(2023, 6, 16))),
        ("What happened after 2020?", (date(2021, 1, 1), date(2023, 6, 16))),
    ],
)
def test_time_phrases(question, expected):
    assert window(question) == expected


@pytest.mark.parametrize(
    "question",
    ["May I ask about my plans?", "When is my exam?", "Tell me about my marbles",
     "What's the date today?", "I have 2 bikes", "Jan said hi"],
)  # fmt: skip
def test_no_time_phrase(question):
    assert window(question) is None


def test_january_last_month_wraps_the_year():
    assert window("last month", datetime(2024, 1, 10, tzinfo=UTC)) == (
        date(2023, 12, 1),
        date(2024, 1, 1),
    )


def test_window_overlap_rules():
    w = TimeWindow(datetime(2023, 3, 1, tzinfo=UTC), datetime(2023, 4, 1, tzinfo=UTC), "in March")
    jan, apr = datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 4, 1, tzinfo=UTC)
    assert overlaps_window(Tier.FACT, jan, None, w)  # still true in March
    assert overlaps_window(Tier.FACT, jan, apr, w)
    assert not overlaps_window(Tier.FACT, jan, datetime(2023, 3, 1, tzinfo=UTC), w)  # ended
    assert not overlaps_window(Tier.FACT, apr, None, w)  # began later
    assert not overlaps_window(Tier.EPISODE, jan, None, w)  # an episode is a moment
    assert overlaps_window(Tier.EPISODE, datetime(2023, 3, 9, tzinfo=UTC), None, w)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Can you remind me what you recommended for dinner?", True),
        ("You told me a ratio for tea tree oil, what was it?", True),
        ("In our previous conversation, which hotel did we pick?", True),
        ("What did we discuss about my resume?", True),
        ("How many bikes do I own?", False),
        ("Where do you think I should travel?", False),
    ],
)
def test_conversational_questions(question, expected):
    assert is_conversational(question) is expected
