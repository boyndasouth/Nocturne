"""The extraction probe makes exactly the batches a full run makes, so its numbers match a
run's evidence coverage and its calls replay from the run's cache."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.agents.sleep.graph import SleepConfig
from eval.datasets.longmemeval import load
from eval.fake import FakeLLM
from eval.probe_extraction import main, probe_conversation, sleep_batches
from eval.systems import episode_id

FIXTURE = Path(__file__).parent / "fixtures" / "longmemeval_mini.json"


def test_episode_ids_sort_in_dataset_order():
    ids = [episode_id("conv-1", i) for i in range(300)]
    assert sorted(ids, key=lambda u: u.bytes) == ids  # Postgres compares uuids bytewise
    assert episode_id("conv-1", 3) == episode_id("conv-1", 3)
    assert episode_id("conv-1", 3) != episode_id("conv-2", 3)
    assert isinstance(ids[0], UUID)


def test_batches_follow_the_sleep_schedule():
    conv = {c.id: c for c in load(FIXTURE)}["ku_001"]  # 2 sessions, 3 episodes
    cfg = SleepConfig(extract_batch_episodes=60)
    every_session = sleep_batches(conv, sleep_every=1, max_episodes=200, cfg=cfg)
    assert [len(b) for b in every_session] == [2, 1]  # one cycle per session
    once = sleep_batches(conv, sleep_every=10, max_episodes=200, cfg=cfg)
    assert [len(b) for b in once] == [3]
    capped = sleep_batches(conv, sleep_every=10, max_episodes=2, cfg=cfg)
    assert [len(b) for b in capped] == [2, 1]  # sleep_max_episodes splits the cycle
    assert [e.evidence for b in once for _, e in b] == [False, False, True]


async def test_probe_counts_covered_evidence():
    conv = {c.id: c for c in load(FIXTURE)}["tr_002"]
    llm = FakeLLM("fake-sleep")
    r = await probe_conversation(conv, llm, cfg=SleepConfig(), sleep_every=1, max_episodes=200)
    assert (r.evidence, r.covered, r.missed) == (2, 2, [])
    assert r.batches == 2 and r.evidence_facts


def test_cli_smoke(tmp_path, monkeypatch, capsys):
    # Defaults only: ignore backend/.env and settings exported in the shell.
    import os

    from app.config import Settings
    from eval.config import EvalSettings

    fields = {f.upper() for f in Settings.model_fields}
    for key in list(os.environ):
        if key.upper().startswith("EVAL_") or key.upper() in fields:
            monkeypatch.delenv(key)
    monkeypatch.setitem(EvalSettings.model_config, "env_file", None)
    monkeypatch.setattr("eval.runner.BACKEND_ENV", tmp_path / "no.env")
    monkeypatch.setenv("EVAL_CACHE_PATH", str(tmp_path / "c.sqlite"))
    args = ["--dataset", "longmemeval-s", "--data-path", str(FIXTURE), "--fake"]
    assert main([*args, "--sleep-every", "1", "--prompt", "v2"]) == 0
    out = capsys.readouterr().out
    assert "prompt v2" in out and "evidence coverage 3/3 (100%)" in out
    assert main([*args, "--sleep-every", "1", "--prompt", "v2"]) == 0
    assert "(0 new, 3 cached)" in capsys.readouterr().out  # a re-probe is free
