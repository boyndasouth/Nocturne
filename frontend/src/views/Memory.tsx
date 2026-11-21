import { useEffect, useState } from "react";
import { explain } from "../api";
import { day, plural, statusName, tierName, useApp, useLoad } from "../lib";
import { NotePanel } from "../components/NotePanel";
import { Sky } from "../components/Sky";

type Mode = "sky" | "list";
const PAGE = 50;

function readMode(): Mode {
  try {
    return localStorage.getItem("nocturne.memoryMode") === "list" ? "list" : "sky";
  } catch {
    return "sky";
  }
}

export function MemoryView({ selected }: { selected: string | null }) {
  const { api, conn, go, refreshKey } = useApp();
  const [mode, setMode] = useState<Mode>(readMode);
  const [query, setQuery] = useState("");
  const [outdated, setOutdated] = useState(true);
  const [forgotten, setForgotten] = useState(false);
  const [turns, setTurns] = useState(false);
  const [offset, setOffset] = useState(0);

  const tier = turns ? "fact,theme,episode" : "fact,theme";
  const status = ["active", outdated && "superseded", forgotten && "archived"].filter(Boolean).join(",");

  useEffect(() => {
    try {
      localStorage.setItem("nocturne.memoryMode", mode);
    } catch {
      // per-visit only
    }
  }, [mode]);
  useEffect(() => setOffset(0), [query, tier, status]);

  const graph = useLoad(
    () => (mode === "sky" ? api.graph({ tier, status, limit: 600 }) : Promise.resolve(null)),
    [api, mode, tier, status, refreshKey],
  );
  const list = useLoad(
    () =>
      mode === "list"
        ? api.notes({ tier, status, q: query, limit: PAGE, offset })
        : Promise.resolve(null),
    [api, mode, tier, status, query, offset, refreshKey],
  );
  const error = graph.error ?? list.error;
  const select = (id: string) => go(`memory/${id}`);

  return (
    <div className={`memory ${selected ? "with-panel" : ""}`}>
      <div className="memory-main">
        <header className="toolbar">
          <div className="segmented" role="tablist" aria-label="View">
            <button role="tab" aria-selected={mode === "sky"} onClick={() => setMode("sky")}>
              Map
            </button>
            <button role="tab" aria-selected={mode === "list"} onClick={() => setMode("list")}>
              List
            </button>
          </div>
          <input
            type="search"
            className="search"
            placeholder="Find a memory"
            aria-label="Find a memory"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <fieldset className="filters">
            <legend className="sr-only">Include</legend>
            <label>
              <input type="checkbox" checked={outdated} onChange={(e) => setOutdated(e.target.checked)} />
              Outdated
            </label>
            <label>
              <input type="checkbox" checked={forgotten} onChange={(e) => setForgotten(e.target.checked)} />
              Forgotten
            </label>
            <label>
              <input type="checkbox" checked={turns} onChange={(e) => setTurns(e.target.checked)} />
              Conversation turns
            </label>
          </fieldset>
        </header>

        {error ? (
          <p className="error stage-msg" role="alert">
            {explain(error, conn)}
          </p>
        ) : mode === "sky" ? (
          graph.data ? (
            graph.data.nodes.length === 0 ? (
              <EmptyMemory />
            ) : (
              <>
                <Sky graph={graph.data} selected={selected} query={query} onSelect={select} />
                {graph.data.total > graph.data.nodes.length && (
                  <p className="fine sky-note">
                    Showing the newest {graph.data.nodes.length.toLocaleString()} of{" "}
                    {graph.data.total.toLocaleString()}
                  </p>
                )}
              </>
            )
          ) : (
            <p className="muted stage-msg">Loading the map</p>
          )
        ) : list.data ? (
          list.data.total === 0 ? (
            query ? (
              <p className="muted stage-msg">Nothing matches “{query}”.</p>
            ) : (
              <EmptyMemory />
            )
          ) : (
            <div className="note-list">
              <p className="fine">{plural(list.data.total, "memory", "memories")}, newest first</p>
              <ul>
                {list.data.items.map((n) => (
                  <li key={n.id}>
                    <button
                      className={`list-row ${n.status}`}
                      aria-current={selected === n.id ? "true" : undefined}
                      onClick={() => select(n.id)}
                    >
                      <span className={`dot ${n.status === "superseded" ? "outdated" : n.tier}`} aria-hidden="true" />
                      <span className="list-text">{n.content}</span>
                      <span className="list-meta">
                        {tierName[n.tier]}
                        {n.status !== "active" && `, ${statusName[n.status].toLowerCase()}`}
                        <br />
                        {day(n.valid_from)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
              {list.data.total > PAGE && (
                <nav className="pager" aria-label="Pages">
                  <button className="ghost" disabled={offset === 0} onClick={() => setOffset(offset - PAGE)}>
                    Newer
                  </button>
                  <span className="muted">
                    {offset + 1}–{Math.min(offset + PAGE, list.data.total)} of {list.data.total}
                  </span>
                  <button
                    className="ghost"
                    disabled={offset + PAGE >= list.data.total}
                    onClick={() => setOffset(offset + PAGE)}
                  >
                    Older
                  </button>
                </nav>
              )}
            </div>
          )
        ) : (
          <p className="muted stage-msg">Loading memories</p>
        )}
      </div>
      {selected && (
        <aside className="side-panel" aria-label="Selected memory">
          <NotePanel id={selected} onClose={() => go("memory")} onSelect={select} />
        </aside>
      )}
    </div>
  );
}

function EmptyMemory() {
  return (
    <div className="stage-msg">
      <h2>No facts yet</h2>
      <p className="muted">
        Facts appear after a sleep cycle reads your conversations. Chat for a while, then run one from{" "}
        <a href="#/nights">Nights</a>.
      </p>
    </div>
  );
}
