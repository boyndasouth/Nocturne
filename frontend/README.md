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

## Deploy (Vercel)

Root directory `frontend`, framework preset Vite, environment variable `VITE_API_URL` = the
backend URL. Then add the Vercel URL to the backend's `CORS_ORIGINS`. Routing is hash-based,
so no rewrites are needed.
