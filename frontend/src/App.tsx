import { useCallback, useMemo, useState } from "react";
import { client, loadConnection, saveConnection, type Connection } from "./api";
import { Ctx, plural, useHashRoute, useLoad } from "./lib";
import { Settings } from "./components/Settings";
import { Chat } from "./views/Chat";
import { MemoryView } from "./views/Memory";
import { Nights } from "./views/Nights";

const NAV = [
  { route: "chat", label: "Chat", hint: "Ask about anything you've told it" },
  { route: "memory", label: "Memory", hint: "Every fact, theme and how they connect" },
  { route: "nights", label: "Nights", hint: "What each sleep cycle changed" },
] as const;

export default function App() {
  const [conn, setConn] = useState<Connection>(loadConnection);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [parts, go] = useHashRoute();
  const api = useMemo(() => client(conn), [conn]);
  const section = parts[0] ?? "chat";

  const openNote = useCallback((id: string) => go(`memory/${id}`), [go]);
  const bump = useCallback(() => setRefreshKey((k) => k + 1), []);
  const ctx = useMemo(
    () => ({ api, conn, go, openNote, refreshKey, bump }),
    [api, conn, go, openNote, refreshKey, bump],
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
            <button className="link-button" onClick={() => setSettingsOpen(true)}>
              Connection settings
            </button>
          </div>
        </aside>

        <main className="stage">
          {section === "memory" ? (
            <MemoryView selected={parts[1] ?? null} />
          ) : section === "nights" ? (
            <Nights selected={parts[1] ?? null} />
          ) : (
            <Chat />
          )}
        </main>
      </div>
      {settingsOpen && (
        <Settings
          value={conn}
          onClose={() => setSettingsOpen(false)}
          onSave={(c) => {
            saveConnection(c);
            setConn(c);
            setSettingsOpen(false);
          }}
        />
      )}
    </Ctx.Provider>
  );
}
