"""Fitting memories into a fixed context budget.

Two pieces, shared by the Responder and the naive-RAG baseline so every memory-based
system gets the same amount of text:

- `fair_caps`: split a character budget across items by water-filling. Items shorter than
  their share keep their full text, and what they don't use goes to the longer ones.
- `excerpt`: shorten one memory to a cap by keeping what matters for the question. A chat
  turn keeps its "User: ..." line, then the run of sentences that shares the most words
  with the question, rare words counting most (often deep inside a long assistant reply: a
  chess move, a dosage, a name in a list), marked with "…" where text was cut. With no
  overlap at all, it falls back to the beginning, as plain truncation would.

No LLM and no embeddings: this is on the online path and must stay cheap.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

ELLIPSIS = "…"

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset(
    """
    a about after again all am an and any are as at be been before being between both but by
    can could did do does doing done for from had has have having he her here hers him his how
    i if in into is it its just me more most my no nor not now of off on once only or other our
    out over own same she should so some such than that the their them then there these they
    this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours remember remind tell told said say know think
    please want like get got one ones also us
    """.split()
)


def _stem(w: str) -> str:
    """Crude plural folding ("doctors" ~ "doctor", "bikes" ~ "bike"); no stemmer dependency."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_RE.findall(text.lower())}


def query_terms(query: str) -> set[str]:
    """Content words of the question: no stopwords; short tokens only if they carry a digit
    ("27", "kg2", "5k")."""
    out = set()
    for w in _WORD_RE.findall(query.lower()):
        if w in _STOPWORDS:
            continue
        if len(w) >= 3 or any(ch.isdigit() for ch in w):
            out.add(_stem(w))
    return out


def _scores(segments: Sequence[str], terms: set[str]) -> list[float]:
    """Query-word overlap per segment, each word weighted by how rare it is in this text
    (1 / number of segments containing it): "move" in a game log counts for little,
    "kg2" a lot."""
    words = [_words(s) & terms for s in segments]
    df = {t: sum(1 for w in words if t in w) for t in terms}
    return [sum(1.0 / df[t] for t in w) for w in words]


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[: max(0, n - 2)].rstrip() + " " + ELLIPSIS


def excerpt(text: str, terms: set[str], max_chars: int) -> str:
    """`text` shortened to about `max_chars`, keeping the part that matches `terms`."""
    if len(text) <= max_chars:
        return text
    head, body = "", text
    if text.startswith("User: "):
        first, _, rest = text.partition("\n")
        head, body = _cut(first, max(80, max_chars // 3)), rest
    room = max_chars - len(head) - (1 if head else 0)
    if room < 40:
        return _cut(head or text, max_chars)

    speaker = ""
    m = re.match(r"(Assistant|User): ", body)
    if m:
        speaker, body = m.group(0), body[m.end() :]
        room -= len(speaker)
    segs = [s for s in _SENTENCE_RE.split(body) if s.strip()]
    scores = _scores(segs, terms)
    if not segs:
        return _cut(head or text, max_chars)
    if not terms or max(scores) == 0:
        kept = _cut(body, room)
        return "\n".join(p for p in (head, speaker + kept) if p)

    best = scores.index(max(scores))
    if len(segs[best]) > room:  # one long sentence: centre on its first matching word
        seg = segs[best]
        low = seg.lower()
        pos = min(
            (low.find(t) for t in terms if t in low),
            default=0,
        )
        start = max(0, min(pos - room // 3, len(seg) - room))
        window = seg[start : start + room - 4].strip()
        middle = (ELLIPSIS + " " if start > 0 else "") + window + " " + ELLIPSIS
        lo = hi = best
    else:
        lo = hi = best
        used = len(segs[best])
        while True:  # grow towards the better-scoring neighbour; ties go right
            options = []
            if hi + 1 < len(segs):
                options.append((scores[hi + 1], 1, hi + 1))
            if lo - 1 >= 0:
                options.append((scores[lo - 1], 0, lo - 1))
            options = [o for o in options if used + 1 + len(segs[o[2]]) <= room - 4]
            if not options:
                break
            _, _, j = max(options)
            used += 1 + len(segs[j])
            lo, hi = min(lo, j), max(hi, j)
        middle = " ".join(segs[lo : hi + 1])
        if lo > 0:
            middle = ELLIPSIS + " " + middle
        if hi < len(segs) - 1:
            middle = middle + " " + ELLIPSIS
    return "\n".join(p for p in (head, speaker + middle) if p)


def fair_caps(
    lengths: Sequence[int], budget: int, *, item_max: int, item_min: int = 200
) -> list[int]:
    """Per-item character caps: the largest common cap c with sum(min(len, c)) <= budget,
    clipped to [item_min, item_max]. Items at or under c keep their full length."""
    if not lengths:
        return []
    lo, hi = 0, max(max(lengths), 1)
    while lo < hi:  # largest c that fits
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    cap = max(item_min, min(item_max, lo))
    return [min(n, cap) for n in lengths]


def fit(texts: Sequence[str], query: str, *, budget: int, item_max: int) -> list[str]:
    """Every text excerpted to its fair share of `budget`."""
    terms = query_terms(query)
    caps = fair_caps([len(t) for t in texts], budget, item_max=item_max)
    return [t if len(t) <= c else excerpt(t, terms, c) for t, c in zip(texts, caps, strict=True)]
