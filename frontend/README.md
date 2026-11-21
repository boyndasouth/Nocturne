# Nocturne frontend

React + TypeScript + Vite. It serves two kinds of reader: the person the memory is about,
who sees it addressed to them ("You live in Dallas.") and can correct or forget any of it,
and developers, who turn on **Settings → Show developer details** to see memories as stored
for the model, ids, scores, model calls and tokens.

Three views over the backend API:

- **Chat**: talk to the agent; numbered citations open the memory they came from. After a
  sleep, a note says what changed overnight.
- **Memory**: "About you" groups current facts under the themes that summarize them; Map
  (force graph) and List show everything, including outdated, faded and forgotten memories.
  Each note shows its history (every version and the night or person that changed it), its
  connections, and what you said that it came from, with **Correct** and **Forget**.
- **Nights**: one page per sleep cycle: what it learned, what stopped being true, what it
  heard again, themes it wrote, links it made, what it let go of, and the Auditor's verdicts.
  "Sleep now" starts a cycle.

Forgetting stops the agent using a memory for good: the note and the conversation turns it
came from (unless another fact relies on them) get the status `forgotten`, which no
retrieval path reads. Faded memories (`archived` by the sleep agent) can still come back
when nothing current answers. The rows stay so the history is traceable.

## Run locally

```powershell
cd frontend
npm install
copy .env.example .env.local   # VITE_API_URL=http://localhost:8000
npm run dev                    # http://localhost:5173
```

The backend must allow the page's origin: `CORS_ORIGINS` in `backend/.env` defaults to the
Vite dev server. If the backend has an `API_KEY`, enter it under Connection settings (it's
kept in this browser only).

To look around without spending LLM quota, seed a separate database with a hand-written
demo memory and point the backend at it:

```powershell
docker exec -it <postgres container> psql -U postgres -c "create database nocturne_demo"
cd backend
python -m scripts.demo_seed postgresql://postgres:postgres@localhost:5432/nocturne_demo
```

## Accounts (Supabase Auth)

With accounts on, everyone who signs in gets a private memory. The backend checks each
request's Supabase token, and Postgres row-level security keeps every query, including the
sleep agent's, inside that person's rows. With accounts off it's one memory, as before.

1. In Supabase, open **Authentication → Providers**: Email is on by default (sign-in links
   and passwords). Optionally enable GitHub or Google and list them in `VITE_AUTH_PROVIDERS`.
2. **Authentication → URL Configuration**: set the Site URL to the frontend's address and add
   it (and `http://localhost:5173` for local use) under Redirect URLs.
3. Frontend `.env.local` (or Vercel environment variables): `VITE_SUPABASE_URL` and
   `VITE_SUPABASE_ANON_KEY` from **Project Settings → API**. The anon/publishable key is
   meant for browsers; never use the service_role key here.
4. Backend `.env`: `AUTH_MODE=supabase`, `SUPABASE_URL`, and a long random `API_KEY` (the
   service key the scheduled job uses for `POST /sleep/all`). Run `python -m scripts.migrate`.

Existing memory belongs to the account `local`. To fill a signed-in account with the demo,
use its user id (Authentication → Users): `python -m scripts.demo_seed <url> --owner <id>`.

**Developer details** (Settings) adds an answer trace under each reply: every memory
retrieval found, its score and how it was reached, which ones made it into the prompt and
were cited, and the exact prompt the model saw.

## Deploy (Vercel)

Root directory `frontend`, framework preset Vite, environment variables `VITE_API_URL` = the
backend URL, plus the `VITE_SUPABASE_*` ones above for accounts. Then add the Vercel URL to the backend's `CORS_ORIGINS`. Routing is hash-based,
so no rewrites are needed.
