"""Cache compatibility: eval runs replay from an LLM cache keyed by the exact prompt, so a
re-run of an old configuration is free only if it sends the same bytes. These hashes pin the
Extractor's system prompt as Phase 3 (no trimming) and Phase 4.1 (trimming) sent it.

If you change the Extractor prompt on purpose, update the hash: old runs will then cost
quota to re-run, so note that in the commit message."""

import hashlib

from app.agents.sleep.extractor import batch_episodes, episode_text
from app.llm.prompts import EXTRACTOR_SYSTEM, EXTRACTOR_SYSTEM_UNTRIMMED, EXTRACTOR_USER
from tests.unit.test_sleep_cost import ep

PHASE3_SYSTEM = "6c08933a27911216da0d6163c83da2596c019304279297cefb6c53826e00cbca"
PHASE41_SYSTEM = "e353d63f5e120ca67583b617ddfc832cf447fbf933dc587494d3c19abd9467a8"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_extractor_prompts_match_the_runs_they_replay():
    assert sha(EXTRACTOR_SYSTEM_UNTRIMMED) == PHASE3_SYSTEM
    assert sha(EXTRACTOR_SYSTEM) == PHASE41_SYSTEM
    assert EXTRACTOR_USER == "Episodes (oldest first):\n\n{episodes}"


def test_long_episodes_are_cut_the_way_each_phase_cut_them():
    long = "User: " + "abc " * 1200  # 4806 chars, no assistant part
    # Phase 3: hard cut at 4000 chars; batches sized on min(raw length, 4000).
    assert episode_text(long) == long[:4000].rstrip() + " …"
    assert [len(b) for b in batch_episodes([ep(long)] * 3, max_chars=8000, max_items=9)] == [2, 1]
    # Phase 4.1: cut on a word boundary; batches sized on the prompt text.
    trimmed = episode_text(long, assistant_chars=400)
    assert trimmed.endswith("abc …") and len(trimmed) <= 4002


def test_v2_adds_the_asides_rule_and_v1_is_untouched():
    from app.llm.prompts import EXTRACTOR_SYSTEMS, extractor_system

    assert extractor_system(trimmed=True) == EXTRACTOR_SYSTEM
    assert extractor_system(trimmed=False) == EXTRACTOR_SYSTEM_UNTRIMMED
    for trimmed in (True, False):
        v1 = extractor_system(trimmed=trimmed, version="v1")
        v2 = extractor_system(trimmed=trimmed, version="v2")
        assert "in passing" in v2 and "in passing" not in v1
        assert v2.replace(v2[v2.index("\n- Read every") : v2.index("\n- From ingested")], "") == v1
    assert len(EXTRACTOR_SYSTEMS) == 4


def test_unknown_prompt_version_is_refused():
    import pytest

    from app.llm.prompts import extractor_system

    with pytest.raises(ValueError):
        extractor_system(trimmed=True, version="v9")
