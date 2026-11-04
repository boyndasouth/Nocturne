"""Embedding backends.

Embedding calls are synchronous and CPU-bound. Async callers should wrap them in
`asyncio.to_thread`, which `aembed_query` and `aembed_documents` do.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.config import Settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """bge-small-en-v1.5 via fastembed (ONNX Runtime, no torch). fastembed L2-normalises its
    output, so cosine distance is meaningful in pgvector.

    fastembed ships a quantized ONNX export of bge-small, so its vectors differ slightly from
    sentence-transformers'. After switching, run `python -m scripts.reembed` once."""

    # BGE v1.5 recommends this instruction for retrieval queries (not documents). fastembed's
    # query_embed doesn't add it for BGE models, so we do.
    BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, dim: int, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.dim = dim
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        self._load()

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    model = TextEmbedding(self.model_name, cache_dir=self._cache_dir)
                    actual = len(next(iter(model.embed(["dimension probe"]))))
                    if actual != self.dim:
                        raise ValueError(
                            f"{self.model_name} produces {actual}-d vectors, "
                            f"but EMBEDDING_DIM={self.dim}"
                        )
                    self._model = model
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._load().embed(list(texts), batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        prefix = self.BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return next(iter(self._load().embed([prefix + text]))).tolist()


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic feature-hashing embedder for tests and CI. NOT semantic: texts are only
    similar if they share words. Never use it for evaluation."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:  # a zero vector has no cosine distance in pgvector
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hash":
        return HashEmbedder(settings.embedding_dim)
    return FastEmbedEmbedder(
        settings.embedding_model, settings.embedding_dim, settings.embedding_cache_dir
    )


async def aembed_query(embedder: Embedder, text: str) -> list[float]:
    return await asyncio.to_thread(embedder.embed_query, text)


async def aembed_documents(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    return await asyncio.to_thread(embedder.embed_documents, list(texts))
