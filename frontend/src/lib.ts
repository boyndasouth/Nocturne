import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { Client, Connection, LinkType, Me, Note } from "./api";
import { toYou } from "./voice";

// --- app context ---------------------------------------------------------------------------

export interface AppCtx {
  api: Client;
  conn: Connection;
  go: (route: string) => void;
  openNote: (id: string) => void;
  refreshKey: number;
  bump: () => void;
  /** Developer details: stored wording, ids, scores and costs. */
  dev: boolean;
  /** Who is signed in and today's limits; null until loaded. */
  me: Me | null;
  /** Exploring the read-only demo memory: no chat, sleep, correct or forget. */
  readOnly: boolean;
}

export const Ctx = createContext<AppCtx | null>(null);

export function useApp(): AppCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useApp outside provider");
  return ctx;
}

/** How a stored memory is shown: as stored for developers, addressed to the reader otherwise. */
export function useText(): (s: string) => string {
  const { dev } = useApp();
  return dev ? (s) => s : toYou;
}

// --- routing (hash based: works on any static host) ----------------------------------------

export function useHashRoute(): [string[], (r: string) => void] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [parts, setParts] = useState(read);
  useEffect(() => {
    const on = () => setParts(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = useCallback((r: string) => {
    window.location.hash = `/${r}`;
  }, []);
  return [parts, go];
}

// --- data loading --------------------------------------------------------------------------

export interface Loaded<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

export function useLoad<T>(fn: () => Promise<T>, deps: unknown[]): Loaded<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const seq = useRef(0);
  useEffect(() => {
    const mine = ++seq.current;
    setLoading(true);
    fn().then(
      (d) => {
        if (mine !== seq.current) return;
        setData(d);
        setError(null);
        setLoading(false);
      },
      (e: unknown) => {
        if (mine !== seq.current) return;
        setError(e);
        setLoading(false);
      },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { data, error, loading, reload: () => setTick((t) => t + 1) };
}

// --- formatting ----------------------------------------------------------------------------

const dayFmt = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});
const timeFmt = new Intl.DateTimeFormat(undefined, {
  hour: "numeric",
  minute: "2-digit",
  timeZone: "UTC",
});

export const day = (iso: string | null | undefined) => (iso ? dayFmt.format(new Date(iso)) : "");
export const dayTime = (iso: string) => `${day(iso)}, ${timeFmt.format(new Date(iso))} UTC`;

export function validRange(n: { valid_from: string | null; valid_to: string | null }): string {
  if (!n.valid_from) return "";
  if (!n.valid_to) return `True since ${day(n.valid_from)}`;
  const a = day(n.valid_from);
  const b = day(n.valid_to);
  return a === b ? `True on ${a}` : `True from ${a} to ${b}`;
}

export const plural = (n: number, one: string, many = `${one}s`) =>
  `${n.toLocaleString()} ${n === 1 ? one : many}`;

export const tierName: Record<Note["tier"], string> = {
  episode: "Conversation turn",
  fact: "Fact",
  theme: "Theme",
};

export const statusName: Record<Note["status"], string> = {
  active: "Current",
  superseded: "Outdated",
  archived: "Faded",
  forgotten: "Forgotten",
};

/** Storage that may be unavailable (private windows, blocked site data). */
export function stored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
export function store(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // kept for this visit only
  }
}

/** How a link reads from this note's side. */
export function linkPhrase(type: LinkType, dir: "out" | "in"): string {
  const out: Record<LinkType, string> = {
    supersedes: "Replaces",
    summarizes: "Summarizes",
    related: "Related to",
    elaborates: "Adds detail to",
    example_of: "Is an example of",
    caused_by: "Caused by",
  };
  const inn: Record<LinkType, string> = {
    supersedes: "Replaced by",
    summarizes: "Part of theme",
    related: "Related to",
    elaborates: "Detailed by",
    example_of: "Has example",
    caused_by: "Led to",
  };
  return (dir === "out" ? out : inn)[type];
}

/** Episode text is "User: ...\nAssistant: ..."; show the user's side. */
export function episodeQuote(content: string): string {
  const m = content.match(/User:\s*([\s\S]*?)(?:\n\s*Assistant:|$)/);
  return (m ? m[1] : content).trim();
}
