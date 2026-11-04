# Nocturne

A long-term memory agent. A fast online agent answers questions; an offline "sleep" agent
consolidates, links, corrects, and prunes memories between sessions.

**Status: Phase 1 (foundations).** Chat with episode memory, hybrid retrieval, and tracing.
The sleep cycle arrives in Phase 2.

## What works now

| Piece | Where |
|---|---|
| Schema: notes, links, versions, sleep_cycles (bitemporal; hard deletes blocked in the DB) | `backend/migrations/001_init.sql` |
| Groq / Gemini / OpenRouter client: rate limiting, Retry-After-aware backoff, Langfuse generations | `backend/app/llm/client.py` |
| bge-small embeddings + hybrid search (pgvector cosine ∪ Postgres full-text) | `backend/app/llm/embeddings.py`, `backend/app/memory/store.py` |
| §5 scoring: `α·sim + β·keyword + γ·strength + δ·recency` | `backend/app/memory/scoring.py` |
| `POST /chat`, `POST /ingest`, `GET /health` | `backend/app/api/` |

## Setup (Windows, PowerShell)

> **OneDrive tip:** create the virtualenv *outside* your OneDrive folder (or exclude it from
> sync). Syncing thousands of `site-packages` files is slow and can lock files mid-install.

```powershell
cd nocturne\backend
python -m venv $HOME\.venvs\nocturne
& $HOME\.venvs\nocturne\Scripts\Activate.ps1
pip install -e ".[dev,embeddings]"     # all prebuilt wheels; no compiler needed
copy .env.example .env                 # then fill in DATABASE_URL and GROQ_API_KEY
python -m scripts.migrate              # creates the pgvector extension and tables
uvicorn app.main:create_app --factory --reload --loop app.eventloop:loop_factory
```

Keep the `--loop` flag: on Windows, psycopg can't run on Python's default event loop.
Without the flag the server stops at startup and prints this command.

**Database:** a Supabase or Neon free project. Running the migration enables pgvector, so you
don't need to touch the dashboard. On Supabase, use the **Session pooler** connection string.
Prefer local? `docker compose up -d` gives you the same setup in Docker.

The first start downloads bge-small (~130 MB) into the Hugging Face cache.

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

## Tests

Works from the repo root or `backend\`.

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
