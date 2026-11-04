import math
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.agents.episode_writer import chunk_text, format_turn, strip_citations
from app.agents.responder import build_memory_context, extract_citations
from app.llm.embeddings import HashEmbedder
from app.memory.models import MemoryNote, ScoredNote, Status, Tier

T = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def note(content: str) -> MemoryNote:
    return MemoryNote(
        id=uuid4(), tier=Tier.EPISODE, content=content, strength=1.0, access_count=0,
        last_accessed=None, valid_from=T, valid_to=None, recorded_at=T, status=Status.ACTIVE,
        version=1,
    )  # fmt: skip


def scored(n: MemoryNote) -> ScoredNote:
    return ScoredNote(note=n, score=0.5, semantic_sim=0.5, keyword_score=0.0, recency=1.0)


# --- responder ------------------------------------------------------------------------------


def test_memory_context_numbers_retrieved_then_recent_without_duplicates():
    a, b, c = note("alpha"), note("beta"), note("gamma")
    text, refmap = build_memory_context([scored(a), scored(b)], [b, c])
    assert refmap == {1: a.id, 2: b.id, 3: c.id}
    assert "[M1]" in text and "[M3]" in text and text.count("beta") == 1
    assert text.index("Relevant memories") < text.index("Most recent conversation")


def test_memory_context_when_empty():
    text, refmap = build_memory_context([], [])
    assert refmap == {} and "no memories" in text


def test_memory_context_truncates_long_notes():
    text, _ = build_memory_context([scored(note("x" * 5000))], [])
    assert len(text) < 1000 and text.endswith("…")


def test_extract_citations_keeps_order_dedupes_and_drops_hallucinated_refs():
    ids = {1: uuid4(), 2: uuid4()}
    cites = extract_citations("See [M2], also [M1] and [M2] again, plus [M9].", ids)
    assert [(c.ref, c.note_id) for c in cites] == [("M2", ids[2]), ("M1", ids[1])]


# --- episode writer -------------------------------------------------------------------------


def test_format_turn_strips_citations_and_handles_missing_answer():
    assert format_turn(" hi ", "Your exam is Friday [M2].") == (
        "User: hi\nAssistant: Your exam is Friday."
    )
    assert format_turn("hi", None) == "User: hi"
    assert strip_citations("a [M1][M2] b") == "a b"


def test_chunk_text_packs_paragraphs_under_limit():
    text = "\n\n".join(["p" * 300] * 5)
    chunks = chunk_text(text, max_chars=700)
    assert all(len(c) <= 700 for c in chunks)
    assert len(chunks) == 3
    assert "".join(c.replace("\n", "") for c in chunks) == "p" * 1500


def test_chunk_text_splits_oversized_paragraphs_and_skips_blanks():
    long_para = " ".join(["This is a sentence."] * 100)
    chunks = chunk_text(f"\n\n   \n\n{long_para}", max_chars=200)
    assert chunks and all(len(c) <= 200 for c in chunks)
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_hard_splits_unbroken_text():
    chunks = chunk_text("x" * 1000, max_chars=300)
    assert all(len(c) <= 300 for c in chunks) and sum(map(len, chunks)) == 1000


# --- hash embedder --------------------------------------------------------------------------


def test_hash_embedder_is_deterministic_normalised_and_lexical():
    e = HashEmbedder(384)
    v1, v2 = e.embed_query("pgvector index"), e.embed_query("pgvector index")
    assert v1 == v2 and len(v1) == 384
    assert math.sqrt(sum(x * x for x in v1)) == pytest.approx(1.0)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    q = e.embed_query("my exam is on friday")
    assert cos(q, e.embed_query("exam friday")) > cos(q, e.embed_query("bananas are yellow"))


def test_hash_embedder_never_returns_zero_vector():
    assert any(HashEmbedder(8).embed_query("!!!"))
