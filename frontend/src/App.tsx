import { useCallback, useEffect, useMemo, useState } from "react";
import { authEnabled, demoActive, setDemo, supabase, type Session } from "./auth";
import { SignIn } from "./components/SignIn";
import { client, loadConnection, saveConnection, type Connection } from "./api";
import { Ctx, plural, store, stored, useHashRoute, useLoad } from "./lib";
import { Settings } from "./components/Settings";
import { Chat } from "./views/Chat";
import { MemoryView } from "./views/Memory";
import { Nights } from "./views/Nights";

const NAV = [
  { route: "chat", label: "Chat", hint: "Ask about anything you've told it" },
  { route: "memory", label: "Memory", hint: "What it knows about you, and how it connects" },
  { route: "nights", label: "Nights", hint: "What each sleep cycle changed" },
] as const;

/** undefined while the stored session is being read, null when signed out. */
function useSession(): Session | null | undefined {
  const [session, setSession] = useState<Session | null | undefined>(authEnabled ? undefined : null);
  useEffect(() => {
    if (!supabase) return;
    void supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data } = supabase.auth.onAuthStateChange((_event, s) => setSession(s));
    return () => data.subscription.unsubscribe();
  }, []);
  return session;
}

function useDemo(): boolean {
  const [demo, setDemoState] = useState(demoActive);
  useEffect(() => {
    const on = () => setDemoState(demoActive());
    window.addEventListener("nocturne-demo", on);
    return () => window.removeEventListener("nocturne-demo", on);
  }, []);
  return demo;
}

export default function App() {
  const session = useSession();
  const demo = useDemo();
  if (authEnabled && session === undefined) return null;
  if (authEnabled && !session) return demo ? <Workspace key="demo" email={null} demo /> : <SignIn />;
  // A fresh workspace per account, so nothing on screen carries over between people.
  return <Workspace key={session?.user.id ?? "local"} email={session?.user.email ?? null} />;
}

async function signOut() {
  try {
    sessionStorage.removeItem("nocturne.chat");
  } catch {
    // nothing stored
  }
  await supabase?.auth.signOut();
}

function Workspace({ email, demo = false }: { email: string | null; demo?: boolean }) {
  const [conn, setConn] = useState<Connection>(loadConnection);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [dev, setDev] = useState(() => stored("nocturne.dev") === "1");
  const [parts, go] = useHashRoute();
  const api = useMemo(() => client(conn, () => void signOut()), [conn]);
  const section = parts[0] ?? "chat";

  const openNote = useCallback((id: string) => go(`memory/${id}`), [go]);
  const bump = useCallback(() => setRefreshKey((k) => k + 1), []);
  const me = useLoad(() => api.me(), [api, refreshKey]);
  const readOnly = demo || (me.data?.demo ?? false);
  const ctx = useMemo(
    () => ({ api, conn, go, openNote, refreshKey, bump, dev, me: me.data, readOnly }),
    [api, conn, go, openNote, refreshKey, bump, dev, me.data, readOnly],
  );

  const stats = useLoad(() => api.stats(), [api, refreshKey]);
  const s = stats.data;

  return (
    <Ctx.Provider value={ctx}>
      <div className="shell">
        <aside className="rail">
          <a className="wordmark" href="#/chat" aria-label="Nocturne home">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M15.2 3.2a8.6 8.6 0 1 0 5.6 13.6A7 7 0 0 1 15.2 3.2z" />
            </svg>
            Nocturne
          </a>
          <nav aria-label="Sections">
            {NAV.map((n) => (
              <a
                key={n.route}
                href={`#/${n.route}`}
                className="nav-item"
                aria-current={section === n.route ? "page" : undefined}
                title={n.hint}
              >
                {n.label}
              </a>
            ))}
          </nav>
          <div className="rail-foot">
            {s ? (
              <dl className="tally">
                <div>
                  <dt>Current facts</dt>
                  <dd>{s.notes.fact.active.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Outdated</dt>
                  <dd>{s.notes.fact.superseded.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Themes</dt>
                  <dd>{s.notes.theme.active.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>Nights slept</dt>
                  <dd>{s.cycles.toLocaleString()}</dd>
                </div>
                {s.pending_episodes > 0 && (
                  <p className="pending">
                    {plural(s.pending_episodes, "turn")} waiting for the next sleep
                  </p>
                )}
              </dl>
            ) : stats.error ? (
              <p className="rail-error">Not connected</p>
            ) : null}
            {email && (
              <p className="account" title={email}>
                {email}
              </p>
            )}
            <div className="rail-links">
              <button className="link-button" onClick={() => setSettingsOpen(true)}>
                Settings
              </button>
              {authEnabled && !demo && (
                <button className="link-button" onClick={() => void signOut()}>
                  Sign out
                </button>
              )}
              {demo && (
                <button className="link-button" onClick={() => setDemo(false)}>
                  Sign in
                </button>
              )}
            </div>
          </div>
        </aside>

        <main className={`stage ${demo ? "has-banner" : ""}`}>
          {demo && (
            <div className="demo-banner" role="note">
              <p>
                You're exploring a demo memory: three months of a student's conversations, tidied up
                over three nights. It's read-only.
              </p>
              <button className="primary" onClick={() => setDemo(false)}>
                Sign in to build your own
              </button>
            </div>
          )}
          <div className="stage-view">
            {section === "memory" ? (
              <MemoryView selected={parts[1] ?? null} />
            ) : section === "nights" ? (
              <Nights selected={parts[1] ?? null} />
            ) : (
              <Chat />
            )}
          </div>
        </main>
      </div>
      {settingsOpen && (
        <Settings
          value={conn}
          dev={dev}
          onClose={() => setSettingsOpen(false)}
          onSave={(c, d) => {
            saveConnection(c);
            setConn(c);
            setDev(d);
            store("nocturne.dev", d ? "1" : "0");
            setSettingsOpen(false);
          }}
        />
      )}
    </Ctx.Provider>
  );
}
