# Deploying Nocturne (free tier)

Four free services, set up once, in this order:

| Piece | Service | What it does |
|---|---|---|
| Database + sign-in | **Supabase** | Postgres with pgvector holds every memory; Supabase Auth signs people in |
| API | **Render** (free web service, Docker) | The FastAPI backend: chat, sleep, memory explorer |
| Web app | **Vercel** (Hobby) | The React frontend |
| Scheduled sleep | **GitHub Actions** | Wakes the API every 6 hours and sleeps every account with new conversations |

Plus your existing **Groq** and **Gemini** keys. Nothing here needs a credit card.

What the free tiers mean for visitors:

- Render's free service **spins down after 15 idle minutes** and takes about a minute to wake.
  The first visit after a quiet spell shows "Can't reach the backend" until it's up; a reload a
  minute later works. The scheduled job wakes it every 6 hours anyway.
- The free LLM quotas are shared by everyone. The API protects them with daily limits (30
  messages and 3 manual sleeps per account; a total answer-token and sleep-request budget a
  little under Groq's and Gemini's free quotas). When the chat model's budget is spent, Gemini
  answers instead; when both are spent, people are told to come back tomorrow (midnight UTC).
- Anyone can open the **read-only demo memory** without signing up. That costs no LLM quota,
  so it's the best link to put on a résumé.

Have these open in a text file as you go: Supabase project URL, publishable key, database
URL, Render URL, Vercel URL, and the Render-generated `API_KEY`.

---

## 1. Supabase: database and sign-in

1. **Restore or create the project.** A paused project: open it and click **Restore project**.
   Free projects pause after about a week without use. The scheduled job (step 4) queries the
   database every 6 hours, which should count as use; if it pauses anyway, restore it again.
2. **Database URL.** Click **Connect** (top of the project page) → **Session pooler** → copy
   the URI and put your database password in it. It looks like
   `postgresql://postgres.abcdefgh:PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres`.
   Use the *session* pooler (port 5432), not the transaction pooler (6543): Nocturne switches
   each connection to the signed-in person's account, which needs session mode.
3. **Project URL and key.** **Settings → API Keys**: copy the **Project URL**
   (`https://abcdefgh.supabase.co`) and the **Publishable key** (`sb_publishable_…`; a legacy
   **anon** key also works). Never use a secret or `service_role` key in the frontend.
4. **Email sign-in without your own mail server.** **Authentication → Sign In / Providers →
   Email**: keep Email enabled and turn **Confirm email** off. Supabase's built-in mailer only
   sends to your own team (about 2 emails an hour), so new visitors couldn't confirm their
   address. Nocturne's sign-in page therefore leads with email + password; sign-in links stay
   hidden until you set up your own SMTP (optional, see the end).
5. **URL configuration** comes in step 3, once you know the Vercel URL.

The schema is created by the API itself on its first start (migrations run before it serves).

## 2. Render: the API

1. Push this repo to GitHub (it's already there if you've been pushing).
2. In Render: **New → Blueprint** → connect GitHub → choose the Nocturne repository. Render
   reads `render.yaml` and proposes one free web service, `nocturne-api`.
3. Fill in the values it asks for:

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | the session pooler URI from 1.2 |
   | `SUPABASE_URL` | the project URL from 1.3 |
   | `CORS_ORIGINS` | leave as `http://localhost:5173` for now; you'll set the Vercel URL in step 3 |
   | `GROQ_API_KEY` | your Groq key |
   | `GEMINI_API_KEY` | your Gemini key |

   `API_KEY` is generated for you (it's the service key for the scheduled job). The limits,
   `AUTH_MODE=supabase`, `DEMO_OWNER=demo` and `EMBEDDING_THREADS=1` are preset.
4. **Apply.** The first build takes several minutes (it installs the embedding model into the
   image). When the log shows `applied 007_usage.sql` and `Nocturne … ready (accounts=supabase…`,
   open `https://<your-service>.onrender.com/health`. You should see `"database": true`.
5. Copy the generated **API_KEY**: service → **Environment** → reveal `API_KEY`.

If the service restarts with an out-of-memory error, check that `EMBEDDING_THREADS` is `1`.

### Fill the demo memory

From your laptop, in the backend venv, run the seed against the **Supabase** database
(the same `DATABASE_URL` you gave Render):

```powershell
cd C:\Users\shiwi\Nocturne\backend
python -m scripts.demo_seed "postgresql://postgres.abcdefgh:PASSWORD@aws-0-us-east-1.pooler.supabase.com:5432/postgres" --owner demo
```

It prints `seeded 15 facts, 2 themes and 3 sleep cycles for demo`. It only adds to the `demo`
account and refuses to run if that account already has memories.

## 3. Vercel: the web app

1. In Vercel: **Add New → Project** → import the Nocturne repository.
2. **Root Directory**: `frontend`. The framework preset is detected as **Vite**.
3. **Environment Variables**:

   | Name | Value |
   |---|---|
   | `VITE_API_URL` | `https://<your-service>.onrender.com` (no trailing slash) |
   | `VITE_SUPABASE_URL` | the project URL from 1.3 |
   | `VITE_SUPABASE_ANON_KEY` | the publishable key from 1.3 |
   | `VITE_DEMO` | `1` |

4. **Deploy**, and note the URL (e.g. `https://nocturne-abc.vercel.app`).
5. Back in **Render** → `nocturne-api` → **Environment**: set `CORS_ORIGINS` to that URL
   (no trailing slash; add `,http://localhost:5173` if you still develop locally). Save; Render
   redeploys.
6. Back in **Supabase** → **Authentication → URL Configuration**: set **Site URL** to the
   Vercel URL, and add it under **Redirect URLs** (plus `http://localhost:5173` for local use).

`VITE_` values are baked in at build time: after changing one in Vercel, redeploy.

## 4. GitHub: the scheduled sleep

Repository **Settings → Secrets and variables → Actions**:

- **Secrets** tab → New repository secret: `NOCTURNE_API_KEY` = the Render `API_KEY`.
- **Variables** tab → New repository variable: `NOCTURNE_URL` = the Render URL (no trailing
  slash).

Then **Actions → Sleep cron → Run workflow** once. It should wake the API and finish green;
the log ends with `{"waiting":0,...}` (or a number of accounts started).

## 5. Check it end to end

1. Open the Vercel URL. You get the sign-in page with **Explore a demo memory**.
2. **Explore a demo memory**: Memory shows the student's facts grouped by theme; Nights shows
   three nights. Chat, Sleep now, Correct and Forget aren't offered.
3. Back to sign-in → **Create an account** with an email and password. You're signed in with an
   empty memory; the rail shows your email.
4. Chat a few messages ("I moved to Austin last month", "I'm training for a half marathon").
   The composer shows how many messages are left today.
5. **Nights → Sleep now**. Within a minute or two a new night lists what it learned.
6. Ask "Where do I live?" and open the citation.

If something fails, the page and the Render log usually say which setting is wrong:

| You see | Likely cause |
|---|---|
| "Can't reach the backend" | Render is waking up (wait a minute), `VITE_API_URL` is wrong, or `CORS_ORIGINS` doesn't exactly match the Vercel URL |
| Sign-in page doesn't appear | `VITE_SUPABASE_URL`/`VITE_SUPABASE_ANON_KEY` missing in Vercel, or not redeployed after adding them |
| "Your session has ended" right after sign-in | Render log: `couldn't fetch Supabase signing keys` → `SUPABASE_URL` wrong; `HS256 token but SUPABASE_JWT_SECRET isn't set` → older project, add the legacy JWT secret (Settings → JWT Keys) as `SUPABASE_JWT_SECRET` |
| Sign-up says to confirm your email | **Confirm email** is still on (step 1.4) |
| "Explore a demo memory" shows nothing | The seed wasn't run against the Supabase database, or `DEMO_OWNER` isn't `demo` |
| Render log: `missing migration` | The start command didn't run migrations; redeploy |

## Optional

- **Sign-in links by email.** Needs an email provider with a domain you control (Resend,
  Brevo and others have free tiers). Configure it in **Authentication → Emails → SMTP
  Settings**, then raise **Authentication → Rate Limits** for emails (it starts at 30 per hour).
  Set `VITE_EMAIL_LINKS=1` in Vercel and redeploy.
- **GitHub or Google sign-in.** Enable the provider in **Authentication → Sign In / Providers**
  and set `VITE_AUTH_PROVIDERS=github` (comma-separated) in Vercel.
- **Different limits.** Change `LIMIT_CHATS_PER_DAY`, `LIMIT_SLEEPS_PER_DAY`,
  `ANSWER_TOKENS_PER_DAY` and `SLEEP_REQUESTS_PER_DAY` in Render (0 = no limit). Keep the two
  budgets under your providers' free quotas; Groq's is a rolling 24 hours, so leave a margin.
