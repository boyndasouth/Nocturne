"""Time references in questions -> a time window (spec §5 step 5).

"Where did I live in March?", "what did I do last week?", "two months ago", "before May
2023". Pure and rule-based: no LLM on the online path. The retriever uses the window to
(1) bring back facts that were true then even if they've since been superseded, and
(2) boost notes whose valid time overlaps it. It's a soft signal, never a hard filter, so a
wrong parse costs a little ranking, not a missing answer.

Windows are half-open [start, end) in UTC. Relative phrases resolve against `now`, the
question's time (simulated time in benchmark replays).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "a few": 3, "several": 4,
}  # fmt: skip
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_NUM_ALT = "|".join(sorted(_NUMBERS, key=len, reverse=True))
_MOD = r"(?P<mod>before|after|since|until|by)\s+"

_ISO_RE = re.compile(r"\b(?P<y>(?:19|20)\d\d)[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})\b")
# "May 5", "May 5th, 2023", "5 May", "5th of May 2023"
_DAY_MONTH_RE = re.compile(
    rf"\b(?:(?P<m1>{_MONTH_ALT})\.?\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"|(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m2>{_MONTH_ALT})\.?)"
    r"(?:,?\s+(?P<y>(?:19|20)\d\d))?\b",
    re.IGNORECASE,
)
# "in March", "during March 2023", "March 2023". A bare month needs a preposition or a year
# ("May I ask" isn't a date).
_MONTH_RE = re.compile(
    rf"\b(?:(?P<prep>in|during|of|on|since|before|after|until|by|around|from)\s+"
    rf"(?:early\s+|mid[- ]|late\s+)?)?(?P<m>{_MONTH_ALT})\.?(?:\s+(?P<y>(?:19|20)\d\d))?\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(
    r"\b(?P<prep>in|during|of|since|before|after|until|by)\s+(?P<y>(?:19|20)\d\d)\b", re.I
)
_AGO_RE = re.compile(
    rf"\b(?P<n>\d+|{_NUM_ALT})\s+(?P<u>day|week|month|year)s?\s+ago\b", re.IGNORECASE
)
_LAST_RE = re.compile(
    r"\b(?P<the>the\s+)?(?P<w>last|this|past|previous)\s+(?P<u>week|month|year)\b", re.I
)
# "in the past two weeks", "over the last 3 months": a rolling window ending today.
_ROLLING_RE = re.compile(
    rf"\b(?:last|past|previous)\s+(?P<n>\d+|{_NUM_ALT})\s+(?P<u>day|week|month|year)s\b",
    re.IGNORECASE,
)
_LAST_WEEKDAY_RE = re.compile(rf"\b(?:last|on)\s+(?P<d>{'|'.join(_WEEKDAYS)})\b", re.I)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.I)
_MOD_PREFIX_RE = re.compile(_MOD + r"$", re.IGNORECASE)


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime
    phrase: str

    def overlaps(self, valid_from: datetime, valid_to: datetime | None) -> bool:
        return valid_from < self.end and (valid_to is None or valid_to > self.start)


def _day(y: int, m: int, d: int) -> tuple[datetime, datetime] | None:
    try:
        start = datetime(y, m, d, tzinfo=UTC)
    except ValueError:
        return None
    return start, start + timedelta(days=1)


def _month(y: int, m: int) -> tuple[datetime, datetime]:
    start = datetime(y, m, 1, tzinfo=UTC)
    end = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=UTC)
    return start, end


def _latest_year(month: int, day: int | None, now: datetime) -> int:
    """A date without a year means the most recent one not after `now`."""
    if (month, day or 1) > (now.month, now.day):
        return now.year - 1
    return now.year


def _modified(
    mod: str | None, span: tuple[datetime, datetime], now: datetime
) -> tuple[datetime, datetime]:
    start, end = span
    if mod in ("before", "until", "by"):
        return datetime(1900, 1, 1, tzinfo=UTC), start if mod == "before" else end
    if mod in ("after", "since"):
        return (end if mod == "after" else start), max(now + timedelta(days=1), end)
    return span


def _mod_before(text: str, pos: int) -> str | None:
    m = _MOD_PREFIX_RE.search(text[max(0, pos - 12) : pos])
    return m.group("mod").lower() if m else None


def parse_time_window(query: str, now: datetime) -> TimeWindow | None:
    """The first time reference in `query`, most specific patterns first."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    text = query
    today = datetime(now.year, now.month, now.day, tzinfo=UTC)

    if m := _ISO_RE.search(text):
        span = _day(int(m["y"]), int(m["m"]), int(m["d"]))
        if span:
            return TimeWindow(*_modified(_mod_before(text, m.start()), span, now), m.group())

    if m := _DAY_MONTH_RE.search(text):
        month = _MONTHS[(m["m1"] or m["m2"]).lower().rstrip(".")]
        day = int(m["d1"] or m["d2"])
        year = int(m["y"]) if m["y"] else _latest_year(month, day, now)
        span = _day(year, month, day)
        if span:
            return TimeWindow(*_modified(_mod_before(text, m.start()), span, now), m.group())

    for m in _MONTH_RE.finditer(text):
        if not (m["prep"] or m["y"]):
            continue
        month = _MONTHS[m["m"].lower().rstrip(".")]
        year = int(m["y"]) if m["y"] else _latest_year(month, None, now)
        prep = (m["prep"] or "").lower()
        mod = prep if prep in ("since", "before", "after", "until", "by") else None
        return TimeWindow(*_modified(mod, _month(year, month), now), m.group().strip())

    if m := _YEAR_RE.search(text):
        y = int(m["y"])
        span = (datetime(y, 1, 1, tzinfo=UTC), datetime(y + 1, 1, 1, tzinfo=UTC))
        prep = m["prep"].lower()
        mod = prep if prep in ("since", "before", "after", "until", "by") else None
        return TimeWindow(*_modified(mod, span, now), m.group())

    if m := _AGO_RE.search(text):
        raw = m["n"].lower()
        n = int(raw) if raw.isdigit() else _NUMBERS[raw]
        days = n * _UNIT_DAYS[m["u"].lower()]
        slack = max(1, days // 2)  # "3 weeks ago" is roughly, not exactly, 21 days
        center = today - timedelta(days=days)
        return TimeWindow(center - timedelta(days=slack), center + timedelta(days=slack + 1),
                          m.group())  # fmt: skip

    if m := _ROLLING_RE.search(text):
        raw = m["n"].lower()
        days = (int(raw) if raw.isdigit() else _NUMBERS[raw]) * _UNIT_DAYS[m["u"].lower()]
        return TimeWindow(today - timedelta(days=days), today + timedelta(days=1), m.group())

    if m := _LAST_RE.search(text):
        which, unit = m["w"].lower(), m["u"].lower()
        # "in the last month" is the past 30 days; plain "last month" is the calendar month.
        if m["the"] and which in ("last", "previous"):
            which = "past"
        if unit == "week":
            start = today - timedelta(days=14 if which in ("last", "previous") else 7)
            return TimeWindow(start, today + timedelta(days=1), m.group())
        if unit == "month":
            if which == "this":
                return TimeWindow(*_month(now.year, now.month), m.group())
            if which == "past":
                return TimeWindow(today - timedelta(days=30), today + timedelta(days=1), m.group())
            prev = (now.year - (now.month == 1), (now.month - 2) % 12 + 1)
            return TimeWindow(*_month(*prev), m.group())
        year = now.year - (which in ("last", "previous"))
        if which == "past":
            return TimeWindow(today - timedelta(days=365), today + timedelta(days=1), m.group())
        return TimeWindow(datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC),
                          m.group())  # fmt: skip

    if m := _LAST_WEEKDAY_RE.search(text):
        target = _WEEKDAYS.index(m["d"].lower())
        back = (today.weekday() - target) % 7 or 7
        start = today - timedelta(days=back)
        return TimeWindow(start, start + timedelta(days=1), m.group())

    if m := _YESTERDAY_RE.search(text):
        return TimeWindow(today - timedelta(days=1), today, m.group())

    return None
