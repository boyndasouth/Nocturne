// Typed client for the Nocturne backend. Shapes mirror backend/app/api/*.py.

export type Tier = "episode" | "fact" | "theme";
export type Status = "active" | "superseded" | "archived";
export type LinkType =
  | "related"
  | "elaborates"
  | "example_of"
  | "caused_by"
  | "supersedes"
  | "summarizes";

export interface Note {
  id: string;
  tier: Tier;
  status: Status;
  content: string;
  keywords: string[];
  strength: number;
  access_count: number;
  last_accessed: string | null;
  valid_from: string;
  valid_to: string | null;
  recorded_at: string;
  version: number;
  source_count: number;
}

export interface NotePage {
  items: Note[];
  total: number;
  limit: number;
  offset: number;
}

export interface HistoryEntry {
  version: number;
  content: string;
  status: Status;
  valid_from: string | null;
  valid_to: string | null;
  strength: number | null;
  source_count: number | null;
  ended: { cycle_id: string | null; cycle_clock: string | null; at: string; reason: string | null } | null;
}

export interface NoteLink {
  link_type: LinkType;
  direction: "out" | "in";
  weight: number;
  created_by_cycle: string | null;
  other: Note;
}

export interface NoteDetail {
  note: Note;
  created_by_cycle: string | null;
  history: HistoryEntry[];
  links: NoteLink[];
  sources: Note[];
}

export interface Edge {
  src_id: string;
  dst_id: string;
  link_type: LinkType;
  weight: number;
  created_by_cycle: string | null;
}

export interface Graph {
  nodes: Note[];
  edges: Edge[];
  total: number;
}

export interface Stats {
  notes: Record<Tier, Record<Status, number>>;
  links: number;
  cycles: number;
  pending_episodes: number;
}

export interface Cycle {
  cycle_id: string;
  status: "running" | "succeeded" | "failed";
  started_at: string;
  finished_at: string | null;
  clock: string;
  episodes_processed: number;
  facts_created: number;
  merges: number;
  supersessions: number;
  links_created: number;
  themes_created: number;
  themes_updated: number;
  audit_score: number | null;
  audit_items: number;
  archived: number;
  tokens_used: number;
  llm_calls: number;
  attempts: number;
  error: string | null;
}

export type EditKind = "superseded" | "archived" | "theme_updated" | "merged" | "edited";

export interface CycleEdit {
  note_id: string;
  tier: Tier;
  kind: EditKind;
  version: number;
  reason: string | null;
  before: string;
  after: string;
  status_now: Status;
}

export interface AuditVerdict {
  item: number;
  kind: string;
  verdict: "good" | "questionable" | "wrong";
  note: string;
}

export interface CycleChanges {
  cycle_id: string;
  clock: string;
  created: Note[];
  episodes: Note[];
  edits: CycleEdit[];
  episodes_forgotten: number;
  links: {
    src_id: string;
    dst_id: string;
    link_type: LinkType;
    weight: number;
    src_content: string;
    dst_content: string;
  }[];
  audit: { by_kind?: Record<string, number>; verdicts?: AuditVerdict[]; error?: string } | null;
}

export interface ChatResponse {
  answer: string;
  citations: { ref: string; note_id: string }[];
  retrieved: { id: string; tier: Tier; score: number; preview: string }[];
  episode_id: string;
}

export interface SleepResponse {
  status: "nothing_to_do" | "started" | "succeeded" | "failed";
  resumed: boolean;
  cycle: Cycle | null;
}

// --- connection settings ------------------------------------------------------------------

const KEY = "nocturne.connection";

export interface Connection {
  url: string;
  apiKey: string;
}

const defaultUrl = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export function loadConnection(): Connection {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const c = JSON.parse(raw) as Partial<Connection>;
      return { url: c.url || defaultUrl, apiKey: c.apiKey ?? "" };
    }
  } catch {
    // storage blocked: fall through to defaults
  }
  return { url: defaultUrl, apiKey: "" };
}

export function saveConnection(c: Connection): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(c));
  } catch {
    // not persisted; still used for this visit
  }
}

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function explain(err: unknown, conn: Connection): string {
  if (err instanceof ApiError) {
    if (err.status === 401) return "The API key was refused. Enter the backend's API_KEY in Settings.";
    if (err.status === 404) return "Not found. It may belong to a different database.";
    if (err.status === 409) return "A sleep cycle is already running. Wait for it to finish.";
    if (err.status === 502 || err.status === 503)
      return `The backend couldn't reach its language model: ${err.message}`;
    return `The backend answered ${err.status}: ${err.message}`;
  }
  return `Can't reach the backend at ${conn.url}. Check the URL in Settings, that the server is running, and that CORS_ORIGINS includes ${window.location.origin}.`;
}

export function client(conn: Connection) {
  const base = conn.url.replace(/\/+$/, "");

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (conn.apiKey) headers["X-API-Key"] = conn.apiKey;
    if (init?.body) headers["Content-Type"] = "application/json";
    const res = await fetch(base + path, { ...init, headers });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = (await res.json()) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        // not JSON
      }
      throw new ApiError(res.status, detail);
    }
    return (await res.json()) as T;
  }

  const qs = (p: Record<string, string | number | undefined>) => {
    const s = new URLSearchParams();
    for (const [k, v] of Object.entries(p)) if (v !== undefined && v !== "") s.set(k, String(v));
    const out = s.toString();
    return out ? `?${out}` : "";
  };

  return {
    stats: () => request<Stats>("/memory/stats"),
    notes: (p: { tier?: string; status?: string; q?: string; limit?: number; offset?: number }) =>
      request<NotePage>(`/memory/notes${qs(p)}`),
    note: (id: string) => request<NoteDetail>(`/memory/notes/${id}`),
    graph: (p: { tier?: string; status?: string; limit?: number }) =>
      request<Graph>(`/memory/graph${qs(p)}`),
    cycles: (limit = 100) => request<Cycle[]>(`/sleep/cycles${qs({ limit })}`),
    cycle: (id: string) => request<Cycle>(`/sleep/cycles/${id}`),
    changes: (id: string) => request<CycleChanges>(`/sleep/cycles/${id}/changes`),
    chat: (message: string) =>
      request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify({ message }) }),
    sleep: () => request<SleepResponse>("/sleep", { method: "POST" }),
  };
}

export type Client = ReturnType<typeof client>;
