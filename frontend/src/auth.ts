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

/** The current access token (supabase-js refreshes it before it expires). */
export async function accessToken(): Promise<string | null> {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

export type { Session };
