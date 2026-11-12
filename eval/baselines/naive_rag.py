"""Naive RAG baseline: one chunk per raw turn, top-k by cosine similarity, nothing else.

No keyword search, no strength or recency, no consolidation, and no database: an in-memory
matrix with the same embedder as Nocturne. Its context gets the same character budget and
excerpting as Nocturne's Responder (app/memory/excerpt.py), so the systems differ in what
they retrieve, not in how much text the answer model reads. The gap between this and
nocturne-no-sleep is what hybrid search and scoring buy; the gap to nocturne is what sleep
buys.
"""

from __future__ import annotations

import time

import numpy as np

from app.llm.client import LLMClient
from app.llm.embeddings import Embedder, aembed_documents, aembed_query
from app.memory.excerpt import fit
from eval.base import PrepStats, SystemAnswer, answer_with_context
from eval.datasets.schema import Conversation, Episode, Question


def top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Indices of the k rows most cosine-similar to `query`, best first. Ties go to the
    later row (newer turn)."""
    if matrix.shape[0] == 0 or k <= 0:
        return []
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
    sims = (matrix @ query) / np.where(norms == 0, 1.0, norms)
    order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), -i))
    return order[:k]


class NaiveRagSystem:
    def __init__(
        self,
        *,
        answer_llm: LLMClient,
        embedder: Embedder,
        top_k: int,
        max_tokens: int,
        budget_chars: int = 12_000,
        item_max_chars: int = 3_000,
        name: str = "naive-rag",
        prompt: str = "v1",
        temperature: float = 0.2,
    ) -> None:
        self._budget = budget_chars
        self._item_max = item_max_chars
        self.name = name
        self._llm = answer_llm
        self._embedder = embedder
        self._k = top_k
        self._max_tokens = max_tokens
        self._prompt = prompt
        self._temperature = temperature
        self._chunks: list[Episode] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)

    async def prepare(self, conv: Conversation) -> PrepStats:
        t0 = time.perf_counter()
        self._chunks = conv.episodes()
        vecs = await aembed_documents(self._embedder, [c.content for c in self._chunks])
        self._matrix = np.asarray(vecs, dtype=np.float32).reshape(len(vecs), self._embedder.dim)
        return PrepStats(
            episodes=len(self._chunks),
            memory_items=len(self._chunks),
            memory_chars=sum(len(c.content) for c in self._chunks),
            by_tier={"episode": len(self._chunks)},
            seconds=time.perf_counter() - t0,
        )

    async def answer(self, q: Question) -> SystemAnswer:
        t0 = time.perf_counter()
        qvec = np.asarray(await aembed_query(self._embedder, q.question), dtype=np.float32)
        # Only what had happened by the time of the question (matters for replays).
        visible = [i for i, c in enumerate(self._chunks) if c.at <= q.asked_at]
        hits = [visible[i] for i in top_k(self._matrix[visible], qvec, self._k)]
        retrieval_ms = (time.perf_counter() - t0) * 1000

        labels = [f"[M{n}] ({self._chunks[i].at:%Y-%m-%d %H:%M} UTC)" for n, i in
                  enumerate(hits, start=1)]  # fmt: skip
        texts = fit(
            [self._chunks[i].content for i in hits],
            q.question,
            budget=max(0, self._budget - sum(len(lb) + 2 for lb in labels)),
            item_max=self._item_max,
        )
        lines = [f"{lb} {t}" for lb, t in zip(labels, texts, strict=True)]
        context = (
            "Relevant memories (most relevant first):\n" + "\n".join(lines)
            if lines
            else "(no memories yet)"
        )
        text = await answer_with_context(
            self._llm,
            q.question,
            context,
            now=q.asked_at,
            max_tokens=self._max_tokens,
            prompt=self._prompt,
            temperature=self._temperature,
        )
        return SystemAnswer(
            text=text,
            retrieval_ms=retrieval_ms,
            retrieved=len(hits),
            context_chars=len(context),
            context=context,
        )

    async def aclose(self) -> None:
        pass
