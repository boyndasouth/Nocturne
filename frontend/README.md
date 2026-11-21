# Nocturne frontend

React + TypeScript + Vite. Three views over the backend API:

- **Chat**: talk to the agent; numbered citations open the memory they came from.
- **Memory**: every fact and theme as a map (force graph) or a list, with each note's
  history (every version and the night that changed it), its connections, and what you said
  that it came from.
- **Nights**: one page per sleep cycle: what it learned, what stopped being true, what it
  heard again, themes it wrote, links it made, what it let go of, and the Auditor's verdicts.
  "Sleep now" starts a cycle.

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
