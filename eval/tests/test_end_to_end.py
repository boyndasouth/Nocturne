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
    # Run on defaults: ignore backend/.env and any EVAL_* or backend setting exported in the
    # shell (e.g. EVAL_SLEEP_EVERY_SESSIONS=10 left over from a real run).
    from app.config import Settings
    from eval.config import EvalSettings

    fields = {f.upper() for f in Settings.model_fields}
    for key in list(os.environ):
        if key.upper().startswith("EVAL_") or key.upper() in fields:
            monkeypatch.delenv(key)
    monkeypatch.setitem(EvalSettings.model_config, "env_file", None)
    monkeypatch.setattr("eval.runner.BACKEND_ENV", tmp_path / "no.env")
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
    assert slept["audit_score"] == ""  # nothing merged, linked or themed: nothing to audit
    # The fake Extractor turns every user turn into a fact, so the one evidence turn is covered.
    assert (slept["evidence_episodes"], slept["evidence_covered"]) == ("1", "1")
    awake = convs[("nocturne-no-sleep", "ku_001")]
    assert (awake["memory_fact"], awake["sleep_tokens"]) == ("0", "0")
    assert awake["evidence_episodes"] == ""  # no sleep, nothing to measure

    totals = {t["system"]: t for t in _csv(run / "totals.csv")}
    assert totals["naive-rag"]["retrieval_p95_ms"] != ""
    assert totals["full-context"]["retrieval_p95_ms"] == ""
    assert float(totals["nocturne"]["evidence_coverage"]) == 1.0
    assert totals["nocturne-no-sleep"]["evidence_coverage"] == ""

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


def test_daily_quota_stops_the_run_on_the_first_error(eval_env, monkeypatch):
    from app.llm.client import QuotaExhaustedError
    from eval.fake import FakeLLM

    async def no_quota(self, messages, **kwargs):
        if self.name == "fake-sleep":
            raise QuotaExhaustedError("gemini: daily quota used up for gemini-x")
        return await original(self, messages, **kwargs)

    original = FakeLLM.chat
    monkeypatch.setattr(FakeLLM, "chat", no_quota)
    assert main([*ARGS, "--run-name", "quota", "--max-errors", "10"]) == 2
    rows = _csv(eval_env / "quota" / "results.csv")
    errors = [r for r in rows if r["status"] == "error"]
    assert len(errors) == 1 and errors[0]["system"] == "nocturne"  # stopped at once


def test_linker_and_abstractor_run_in_the_eval(eval_env, monkeypatch):
    monkeypatch.setenv("EVAL_SLEEP_EVERY_SESSIONS", "2")
    monkeypatch.setenv("THEME_SIM", "0.05")  # hash embeddings: let the fixture cluster
    monkeypatch.setenv("THEME_MIN_FACTS", "2")
    monkeypatch.setenv("LINK_SIM", "0.05")
    assert main([*ARGS, "--run-name", "structure", "--systems", "nocturne"]) == 0
    convs = {c["conversation_id"]: c for c in _csv(eval_env / "structure" / "conversations.csv")}
    ku = convs["ku_001"]  # 2 sessions, 3 user turns
    assert ku["sleep_cycles"] == "1"  # every 2 sessions -> one cycle
    assert int(ku["links"]) > 0 and int(ku["memory_theme"]) > 0
    assert float(ku["audit_score"]) == 1.0 and int(ku["audit_items"]) > 0  # fake judge: "good"


def test_runs_made_before_a_setting_existed_still_resume(eval_env, monkeypatch):
    import json

    assert main(ARGS) == 0
    config = eval_env / "e2e" / "config.json"
    data = json.loads(config.read_text())
    for key in ("extract_assistant_chars", "extract_prompt"):
        data["fingerprint"]["backend"].pop(key)  # as written by an older version
    config.write_text(json.dumps(data))
    monkeypatch.setenv("EXTRACT_ASSISTANT_CHARS", "0")  # what those runs used
    monkeypatch.setenv("EXTRACT_PROMPT", "v1")
    assert main(ARGS) == 0
    monkeypatch.setenv("EXTRACT_PROMPT", "v2")  # a real change is still refused
    assert main(ARGS) == 2
