# Nocturne

A long-term memory agent. A fast online agent answers questions; an offline "sleep" agent
consolidates, links, corrects, and prunes memories between sessions.

**Status: Phase 3, step 1 of 4 (eval harness + baselines).** Chat with episode memory and
hybrid retrieval, a checkpointed LangGraph sleep cycle that extracts facts, merges duplicates,
and supersedes contradicted facts with bitemporal history, and a benchmark harness that
compares it against baselines. Linking, themes, and forgetting come next in Phase 3.

## What works now

| Piece | Where |
|---|---|
| Schema: notes, links, versions, sleep_cycles (bitemporal; hard deletes blocked in the DB) | `backend/migrations/001_init.sql` |
| Groq / Gemini / OpenRouter client: rate limiting, Retry-After-aware backoff, Langfuse generations | `backend/app/llm/client.py` |
| bge-small embeddings + hybrid search (pgvector cosine ∪ Postgres full-text) | `backend/app/llm/embeddings.py`, `backend/app/memory/store.py` |
| §5 scoring: `α·sim + β·keyword + γ·strength + δ·recency` | `backend/app/memory/scoring.py` |
| `POST /chat`, `POST /ingest`, `GET /health` | `backend/app/api/` |
| Sleep graph: Extractor → Deduplicator → Contradiction Resolver → Apply (LangGraph, Postgres checkpoints) | `backend/app/agents/sleep/` |
| Versioned edits: full-row JSON snapshot before every change; `supersedes` links; valid time vs. transaction time | `backend/app/memory/store.py`, `migrations/002_sleep_cycle.sql` |
| `POST /sleep`, `GET /sleep/cycles[/{id}]`, idle/backlog trigger, GitHub Actions cron | `backend/app/api/sleep.py`, `backend/app/agents/sleep/trigger.py`, `.github/workflows/sleep-cron.yml` |
| Responder sees current facts, plus the older facts they replaced, with dates | `backend/app/agents/responder.py` |
| Eval harness: LongMemEval + LoCoMo loaders, 4 systems, official LongMemEval judge rubrics, resumable runs, LLM response cache, CSV reports | `eval/` |

## Setup (Windows, PowerShell)

> **OneDrive tip:** create the virtualenv *outside* your OneDrive folder (or exclude it from
> sync). Syncing thousands of `site-packages` files is slow and can lock files mid-install.

```powershell
cd nocturne\backend
python -m venv $HOME\.venvs\nocturne
& $HOME\.venvs\nocturne\Scripts\Activate.ps1
pip install -e ".[dev,embeddings]"     # all prebuilt wheels (ONNX Runtime, no torch)
copy .env.example .env                 # fill in DATABASE_URL, GROQ_API_KEY, GEMINI_API_KEY
python -m scripts.migrate              # pgvector, tables, and LangGraph checkpoint tables
uvicorn app.main:create_app --factory --reload --loop app.eventloop:loop_factory
```

Keep the `--loop` flag: on Windows, psycopg can't run on Python's default event loop.
Without the flag the server stops at startup and prints this command.

**Database:** a Supabase or Neon free project. Running the migration enables pgvector, so you
don't need to touch the dashboard. On Supabase, use the **Session pooler** connection string.
Prefer local? `docker compose up -d` gives you the same setup in Docker.

The first start downloads bge-small (~70 MB, quantized ONNX) into fastembed's cache.

**Upgrading from Phase 1:** run `python -m scripts.migrate`, then `python -m scripts.reembed`
once. Embeddings moved from sentence-transformers to fastembed, whose bge-small vectors differ
slightly. Your existing episodes are consolidated by the first sleep cycle.

## Try it

```powershell
$h = @{ "Content-Type" = "application/json" }   # add "X-API-Key" if you set API_KEY
irm localhost:8000/chat -Method Post -Headers $h -Body '{"message":"My linear algebra exam is on Friday"}'
# stop and restart uvicorn, then:
irm localhost:8000/chat -Method Post -Headers $h -Body '{"message":"When is my exam?"}'
```

The second answer cites the first turn (`citations[].note_id`), even after a restart. With
Langfuse keys set, each request shows up as a `chat` trace containing `retrieve` and
`responder` spans. Interactive docs: http://localhost:8000/docs

### Sleep (Phase 2 exit check)

```powershell
irm localhost:8000/chat -Method Post -Headers $h -Body '{"message":"I live in Austin"}'
irm "localhost:8000/sleep?wait=true" -Method Post -Headers $h
irm localhost:8000/chat -Method Post -Headers $h -Body '{"message":"Actually I moved to Dallas last week"}'
irm "localhost:8000/sleep?wait=true" -Method Post -Headers $h
irm localhost:8000/chat -Method Post -Headers $h -Body '{"message":"Where do I live, and where did I live before?"}'
irm localhost:8000/sleep/cycles -Headers $h      # what each cycle did, tokens spent
```

Without `?wait=true`, `/sleep` returns 202 and runs in the background. You rarely need to call
it: the server sleeps on its own after `SLEEP_IDLE_MINUTES` idle (with at least
`SLEEP_IDLE_MIN_EPISODES` pending) or once `SLEEP_BACKLOG_EPISODES` pile up. After deploy, the
`Sleep cron` workflow covers the hours when the free host is asleep (setup notes in the file).

For benchmark replay, pass `{"occurred_at": "..."}` to `/sleep` as well as `/chat`: facts are
recorded at that time, so replayed questions can't see facts from the future.

## Evaluation

The harness replays a benchmark's history, builds each system's memory, asks the questions,
and has an LLM judge grade the answers (spec §9). Every system answers with the same model
and the same Responder prompt, so the only difference between rows is the memory.

| System | Memory |
|---|---|
| `full-context` | Raw history in the prompt, newest first until `EVAL_FULL_CONTEXT_TOKENS` |
| `naive-rag` | One chunk per turn, top-k cosine, nothing else |
| `nocturne-no-sleep` | The real retriever (hybrid + §5 scoring + recent turns) over episodes |
| `nocturne` | The same, after a sleep cycle at the end of every session |

```powershell
docker compose up -d                 # creates the nocturne_eval database (see docker-compose.yml)
$env:EVAL_DATABASE_URL = "postgresql://nocturne:nocturne@localhost:5432/nocturne_eval"
python eval/run_eval.py --limit 30 --plan          # what it will cost, spends nothing
python eval/run_eval.py --limit 30 --fake          # offline dry run: no keys, no quota
python eval/run_eval.py --limit 30                 # the real thing (LongMemEval oracle)
python eval/run_eval.py --limit 30 --report-only --sample 20   # CSVs + 20 verdicts to hand-check
```

Results land in `eval/results/<run>/`: `summary.csv` (accuracy per system × category),
`totals.csv` (overall and task-averaged accuracy, retrieval p50/p95, tokens online vs.
offline, memory items per episode), `results.csv` (every answer with its verdict), and
`conversations.csv` (memory size and sleep work per conversation).

- **Resumable.** Rows are appended as they finish. Rerun the same command after a rate limit
  or Ctrl-C and it picks up where it stopped; every LLM call is cached in `eval/.cache/`, so
  finished work (including sleep cycles) is free the second time.
- **Fair splits.** Each question is assigned to `dev` (20%) or `test` by a hash of its id.
  Tune on `--split dev`, report on `--split test` (spec §11).
- **Guarded.** A run refuses to continue with a different model or setting (`config.json`
  holds the fingerprint), and the eval database must have "eval" in its name because it's
  wiped before every conversation. Never point it at your real memory.
- **Three quota buckets.** Answers use the online model (Groq), sleep uses Gemini, and the
  judge defaults to Groq `openai/gpt-oss-20b`, which has its own limits separate from the
  answer model. Override with `EVAL_JUDGE_PROVIDER` / `EVAL_JUDGE_MODEL`.
- **Datasets** download on first use into `eval/datasets/data/` (`--data-path` uses a local
  copy). Start with `longmemeval-oracle` (evidence sessions only); `longmemeval-s` has ~50
  sessions per question and `locomo` ~27 per conversation, so sleep costs grow fast. Check
  with `--plan`.

Known limits: the full-context budget defaults to 5,000 tokens because Groq's free tier caps
a single request at its 8K tokens-per-minute limit, so on long histories it's a "recent
history" baseline rather than an upper bound (`truncated_episodes` says how much it dropped).
On LoCoMo, a two-person dialogue, the extractor still assumes a user/assistant chat.

## Tests

Works from the repo root or `backend\`; from the root it also runs the eval harness tests.

```powershell
pytest -q                                   # unit tests; integration tests skip
$env:TEST_DATABASE_URL = "postgresql://nocturne:nocturne@localhost:5432/nocturne_test"
pytest -q                                   # + integration tests (WIPES that database)
```

CI (`.github/workflows/ci.yml`) runs everything against a pgvector service container.

## Design notes

- **Short citation refs.** The prompt labels memories `[M1]`, `[M2]`, … and the server maps
  them back to UUIDs, because LLMs mangle UUIDs. Refs the model invents are dropped.
- **No lost input.** If the LLM call fails, the user's message is still saved as an episode
  and `/chat` returns 502.
- **Replay-safe.** `occurred_at` on `/chat` sets simulated time. Retrieval then hides anything
  recorded after that time, so benchmark replays can't see future sessions.
- **OR-semantics full-text.** `plainto_tsquery` ANDs every word, which makes natural-language
  questions match almost nothing. The query ORs the lexemes instead.
- **Pooler-safe.** Prepared statements are off, so transaction-mode poolers work.

### Sleep cycle

- **The LLM decides *whether* facts conflict; valid time decides *which wins*.** The fact
  that became true later supersedes the other, so notes ingested today about January become
  history instead of overriding March. Ordering by time also makes supersession cycles
  impossible.
- **Narrow auto-merge band.** Only cosine ≥ 0.97 merges without asking. "Lives in Austin" vs.
  "lives in Dallas" can score ~0.9 with bge-small, so anything looser would silently merge
  contradictions. 0.82–0.97 goes to an LLM judge that is told to answer "not a duplicate"
  when unsure. Thresholds are settings; tune them on a dev split in Phase 4.
- **One batch per checkpointed step.** A free-tier rate limit on batch 7 of 9 resumes at batch
  7; batches 1–6 aren't paid for again. Failed cycles are resumed (up to
  `SLEEP_MAX_ATTEMPTS`) before new work is picked up.
- **All-or-nothing writes.** Every step before Apply only reads memory. Apply writes facts,
  merges, supersessions, version snapshots, the episodes' `consolidated_in`, and the cycle's
  `succeeded` status in one transaction, so a resumed cycle can't double-apply.
- **One cycle at a time, across processes.** A partial unique index allows a single `running`
  row in `sleep_cycles`; a cycle whose heartbeat stops (crashed process) is taken over after
  `SLEEP_STALE_MINUTES`.
- **Facts come from the user, not the assistant.** The extractor uses assistant replies only
  as context, so a hallucinated answer can't become a "memory".
- **Checkpoints are deleted after success** to keep free-tier storage flat. Each cycle's
  episode ids, token count, and LLM calls stay in `sleep_cycles`.
