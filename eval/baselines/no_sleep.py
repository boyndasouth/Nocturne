"""Nocturne without sleep: the real Responder and retriever (hybrid search + §5 scoring +
recent turns) over raw episodes only. Isolates what consolidation adds."""

from __future__ import annotations

from typing import Any

from eval.systems import NocturneSystem


def nocturne_no_sleep(**kwargs: Any) -> NocturneSystem:
    return NocturneSystem("nocturne-no-sleep", sleep=False, sleep_llm=None, **kwargs)
