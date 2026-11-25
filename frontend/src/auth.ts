// Supabase Auth in the browser. Accounts are on when the build has a Supabase URL and key;
// otherwise the app talks to a single local memory, as before.
import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";

const url = (import.meta.env.VITE_SUPABASE_URL as string | undefined)?.trim();
const key = (import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined)?.trim();

export const supabase: SupabaseClient | null =
  url && key ? createClient(url, key, {
        // PKCE: sign-in links come back as ?code=…, which leaves the app's #/routes alone.
        auth: { flowType: "pkce", persistSession: true, detectSessionInUrl: true },
      }) : null;

export const authEnabled = supabase !== null;

export type Provider = "github" | "google";
export const providers: Provider[] = ((import.meta.env.VITE_AUTH_PROVIDERS as string | undefined) ?? "")
  .split(",")
  .map((p) => p.trim().toLowerCase())
  .filter((p): p is Provider => p === "github" || p === "google");

/** Sign-in links by email need your own SMTP in Supabase: its built-in sender only reaches
 * your team. Off by default, so the page leads with passwords. */
export const emailLinks = ["1", "true", "yes"].includes(
  String(import.meta.env.VITE_EMAIL_LINKS ?? "").trim().toLowerCase(),
);

/** The current access token (supabase-js refreshes it before it expires). */
export async function accessToken(): Promise<string | null> {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

export type { Session };

// --- the read-only demo memory ------------------------------------------------------------

/** The sign-in page offers it when the build says the backend has one (DEMO_OWNER). */
export const demoAvailable = ["1", "true", "yes"].includes(
  String(import.meta.env.VITE_DEMO ?? "").trim().toLowerCase(),
);
const DEMO_KEY = "nocturne.demo";

export function demoActive(): boolean {
  try {
    return demoAvailable && sessionStorage.getItem(DEMO_KEY) === "1";
  } catch {
    return false;
  }
}

export function setDemo(on: boolean): void {
  try {
    if (on) sessionStorage.setItem(DEMO_KEY, "1");
    else sessionStorage.removeItem(DEMO_KEY);
  } catch {
    // demo only for this page view
  }
  window.dispatchEvent(new Event("nocturne-demo"));
}
