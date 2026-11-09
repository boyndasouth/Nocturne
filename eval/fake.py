"""A deterministic stand-in LLM for `--fake` dry runs and tests. It speaks every prompt the
harness sends (responder, extractor, dedup, contradiction, judge) well enough to exercise
the full pipeline with no API keys and no quota. Its scores mean nothing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from app.llm.client import ChatMessage, LLMResponse
from app.llm.prompts import (
    ABSTRACTOR_SYSTEM,
    AUDITOR_SYSTEM,
    CONTRADICTION_SYSTEM,
    DEDUP_SYSTEM,
    EXTRACTOR_SYSTEM,
    EXTRACTOR_SYSTEM_UNTRIMMED,
    LINKER_SYSTEM,
    RESPONDER_SYSTEM,
)

_EPISODE_RE = re.compile(r"^\[(\d+)\] \([^)]*\)\n(.*?)(?=^\[\d+\] |\Z)", re.M | re.S)
_GROUP_RE = re.compile(r"^Group (\d+)$", re.M)
_CLUSTER_RE = re.compile(r"^Cluster (\d+)\n- \([^)]*\) (.*)$", re.M)
_MEMORY_RE = re.compile(r"^\s*\[M\d+\][^)]*\)\s*(.+)$", re.M)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


class FakeLLM:
    def __init__(self, name: str = "fake", model: str = "fake-1") -> None:
        self.name = name
        self.model = model

    async def chat(
        self, messages: Sequence[ChatMessage], *, trace_name: str = "llm.chat", **_: object
    ) -> LLMResponse:
        text = self._reply(messages)
        prompt_chars = sum(len(m["content"]) for m in messages)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=prompt_chars // 4,
            output_tokens=len(text) // 4 + 1,
            latency_ms=0.0,
        )

    async def aclose(self) -> None:
        pass

    def _reply(self, messages: Sequence[ChatMessage]) -> str:
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if system in (EXTRACTOR_SYSTEM, EXTRACTOR_SYSTEM_UNTRIMMED):
            facts = []
            for n, body in _EPISODE_RE.findall(user):
                first = body.strip().splitlines()[0] if body.strip() else ""
                said = first.split(":", 1)[1].strip() if ":" in first else first
                if said:
                    facts.append({"statement": said[:300], "keywords": [], "sources": [int(n)]})
            return json.dumps({"facts": facts})
        if system == DEDUP_SYSTEM:
            groups = [int(g) for g in _GROUP_RE.findall(user)]
            return json.dumps({"decisions": [{"group": g, "duplicate_of": None} for g in groups]})
        if system == CONTRADICTION_SYSTEM:
            groups = [int(g) for g in _GROUP_RE.findall(user)]
            return json.dumps(
                {"decisions": [{"group": g, "conflicts_with": [], "reason": ""} for g in groups]}
            )
        if system == LINKER_SYSTEM:
            groups = [int(g) for g in _GROUP_RE.findall(user)]
            return json.dumps(
                {
                    "decisions": [
                        {"group": g, "links": [{"to": "A", "type": "related"}]} for g in groups
                    ]
                }
            )
        if system == ABSTRACTOR_SYSTEM:
            themes = [
                {
                    "cluster": int(n),
                    "title": " ".join(first.split()[:5]),
                    "summary": first[:200],
                    "keywords": first.split()[:3],
                }
                for n, first in _CLUSTER_RE.findall(user)
            ]
            return json.dumps({"themes": themes})
        if system == AUDITOR_SYSTEM:
            items = re.findall(r"^Item (\d+)$", user, re.M)
            return json.dumps(
                {"verdicts": [{"item": int(i), "verdict": "good", "note": ""} for i in items]}
            )
        if system == RESPONDER_SYSTEM:
            hits = _MEMORY_RE.findall(user)
            return hits[0][:400] if hits else "I don't have that in memory."
        return self._judge(user)

    @staticmethod
    def _judge(prompt: str) -> str:
        if "unanswerable" in prompt:
            return "yes" if "don't have" in prompt.split("Model Response:")[-1] else "no"
        gold = re.search(r"(?:Correct Answer|Rubric): (.*?)\n\nModel Response: ", prompt, re.S)
        resp = prompt.split("Model Response: ", 1)[-1].rsplit("\n\n", 1)[0]
        if gold is None:
            return "no"
        return "yes" if _norm(gold.group(1)) and _norm(gold.group(1)) in _norm(resp) else "no"
