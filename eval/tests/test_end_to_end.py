"""The whole harness, offline: fake LLM + hash embedder, real Postgres for the nocturne
systems. Skipped without TEST_DATABASE_URL (same disposable database as backend tests)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from eval.run_eval import main

FIXTURE = Path(__file__).parent / "fixtures" / "longmemeval_mini.json"
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture
def eval_env(tmp_path, monkeypatch):
    if not TEST_DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    monkeypatch.setenv("EVAL_DATABASE_URL", TEST_DB_URL)
    monkeypatch.setenv("EVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("EVAL_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    return tmp_path / "results"


def _csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


ARGS = [
    "--dataset",
    "longmemeval-oracle",
    "--data-path",
    str(FIXTURE),
    "--fake",
    "--run-name",
    "e2e",
]


def test_fake_run_produces_results_for_every_system_and_resumes(eval_env):
    assert main(ARGS) == 0
    run = eval_env / "e2e"

    results = _csv(run / "results.csv")
    assert len(results) == 3 * 4
    assert {r["system"] for r in results} == {
        "full-context",
        "naive-rag",
        "nocturne-no-sleep",
        "nocturne",
    }
    assert all(r["status"] == "ok" and r["correct"] in ("True", "False") for r in results)
    assert {r["category"] for r in results} == {
        "knowledge-update",
        "temporal-reasoning",
        "abstention",
    }

    convs = {(c["system"], c["conversation_id"]): c for c in _csv(run / "conversations.csv")}
    slept = convs[("nocturne", "ku_001")]
    assert int(slept["sleep_cycles"]) == 2  # one per session
    assert int(slept["memory_fact"]) > 0 and int(slept["sleep_tokens"]) > 0
    awake = convs[("nocturne-no-sleep", "ku_001")]
    assert (awake["memory_fact"], awake["sleep_tokens"]) == ("0", "0")

    totals = {t["system"]: t for t in _csv(run / "totals.csv")}
    assert totals["naive-rag"]["retrieval_p95_ms"] != ""
    assert totals["full-context"]["retrieval_p95_ms"] == ""

    lines_before = (run / "questions.jsonl").read_text().count("\n")
    assert main(ARGS) == 0  # resume: nothing left to do
    assert (run / "questions.jsonl").read_text().count("\n") == lines_before


def test_changed_configuration_is_refused(eval_env):
    assert main(ARGS) == 0
    assert main([*ARGS, "--seed", "7"]) == 2


def test_refuses_a_database_that_is_not_disposable(eval_env, monkeypatch):
    monkeypatch.setenv("EVAL_DATABASE_URL", "postgresql://u:p@localhost:5432/nocturne")
    from eval.memdb import UnsafeDatabaseError

    with pytest.raises(UnsafeDatabaseError):
        main([*ARGS, "--run-name", "unsafe"])
