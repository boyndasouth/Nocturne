import { useEffect, useMemo, useState } from "react";
import { explain, type Graph, type Note } from "../api";
import { day, plural, statusName, store, stored, tierName, useApp, useLoad, useText } from "../lib";
import { NotePanel } from "../components/NotePanel";
import { Sky } from "../components/Sky";

type Mode = "about" | "sky" | "list";
const PAGE = 50;
const MODES: { id: Mode; label: string }[] = [
  { id: "about", label: "About you" },
  { id: "sky", label: "Map" },
  { id: "list", label: "List" },
];

function readMode(): Mode {
  const m = stored("nocturne.memoryMode");
  return m === "sky" || m === "list" ? m : "about";
}

export function MemoryView({ selected }: { selected: string | null }) {
  const { api, conn, go, refreshKey } = useApp();
  const [mode, setMode] = useState<Mode>(readMode);
  const [query, setQuery] = useState("");
  const [outdated, setOutdated] = useState(true);
  const [faded, setFaded] = useState(false);
  const [forgotten, setForgotten] = useState(false);
  const [turns, setTurns] = useState(false);
  const [offset, setOffset] = useState(0);
  const text = useText();

  const tier = turns ? "fact,theme,episode" : "fact,theme";
  const status = [
    "active",
    outdated && "superseded",
    faded && "archived",
    forgotten && "forgotten",
  ]
    .filter(Boolean)
    .join(",");

  useEffect(() => store("nocturne.memoryMode", mode), [mode]);
  useEffect(() => setOffset(0), [query, tier, status]);

  const graph = useLoad(
    () =>
      mode === "sky"
        ? api.graph({ tier, status, limit: 600 })
        : mode === "about"
          ? api.graph({ tier: "fact,theme", status: "active", limit: 2000 })
          : Promise.resolve(null),
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
            {MODES.map((m) => (
              <button
                key={m.id}
                role="tab"
                aria-selected={mode === m.id}
                onClick={() => setMode(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <input
            type="search"
            className="search"
            placeholder="Find a memory"
            aria-label="Find a memory"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {mode !== "about" && (
            <fieldset className="filters">
              <legend className="sr-only">Include</legend>
              <Check label="Outdated" on={outdated} set={setOutdated} />
              <Check label="Faded" on={faded} set={setFaded} />
              <Check label="Forgotten" on={forgotten} set={setForgotten} />
              <Check label="Conversation turns" on={turns} set={setTurns} />
            </fieldset>
          )}
        </header>

        {error ? (
          <p className="error stage-msg" role="alert">
            {explain(error, conn)}
          </p>
        ) : mode === "about" ? (
          graph.data ? (
            <AboutYou graph={graph.data} query={query} selected={selected} onSelect={select} />
          ) : (
            <p className="muted stage-msg">Loading what it knows</p>
          )
        ) : mode === "sky" ? (
          graph.data ? (
            graph.data.nodes.length === 0 ? (
              <EmptyMemory />
            ) : (
              <>
                <Sky
                  graph={graph.data}
                  selected={selected}
                  query={query}
                  onSelect={select}
                  label={text}
                />
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
                      <span
                        className={`dot ${n.status === "superseded" ? "outdated" : n.tier}`}
                        aria-hidden="true"
                      />
                      <span className="list-text">{text(n.content)}</span>
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
                  <button
                    className="ghost"
                    disabled={offset === 0}
                    onClick={() => setOffset(offset - PAGE)}
                  >
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

function Check(props: { label: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <label>
      <input type="checkbox" checked={props.on} onChange={(e) => props.set(e.target.checked)} />
      {props.label}
    </label>
  );
}

/** Current facts grouped under the themes that summarize them; the rest at the end. */
function AboutYou(props: {
  graph: Graph;
  query: string;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const text = useText();
  const { groups, rest, facts } = useMemo(() => {
    const byId = new Map(props.graph.nodes.map((n) => [n.id, n]));
    const themed = new Set<string>();
    const groups: { theme: Note; facts: Note[] }[] = [];
    for (const theme of props.graph.nodes.filter((n) => n.tier === "theme")) {
      const members = props.graph.edges
        .filter((e) => e.link_type === "summarizes" && e.src_id === theme.id)
        .map((e) => byId.get(e.dst_id))
        .filter((n): n is Note => !!n && n.tier === "fact");
      members.sort((a, b) => a.valid_from.localeCompare(b.valid_from));
      members.forEach((m) => themed.add(m.id));
      groups.push({ theme, facts: members });
    }
    groups.sort((a, b) => b.facts.length - a.facts.length);
    const rest = props.graph.nodes
      .filter((n) => n.tier === "fact" && !themed.has(n.id))
      .sort((a, b) => b.valid_from.localeCompare(a.valid_from));
    const facts = props.graph.nodes.filter((n) => n.tier === "fact").length;
    return { groups, rest, facts };
  }, [props.graph]);

  const q = props.query.trim().toLowerCase();
  const hit = (n: Note) => !q || n.content.toLowerCase().includes(q);
  const shown = groups
    .map((g) => ({ ...g, facts: hit(g.theme) && q ? g.facts : g.facts.filter(hit) }))
    .filter((g) => g.facts.length > 0 || (q && hit(g.theme)) || !q);
  const restShown = rest.filter(hit);

  if (facts === 0) return <EmptyMemory />;

  const row = (n: Note) => (
    <li key={n.id}>
      <button
        className="row-button"
        aria-current={props.selected === n.id ? "true" : undefined}
        onClick={() => props.onSelect(n.id)}
      >
        <span className="dot fact" aria-hidden="true" />
        {text(n.content)}
      </button>
    </li>
  );

  return (
    <div className="about">
      <header className="about-head">
        <h1>What it knows about you</h1>
        <p className="muted">
          {plural(facts, "thing")} it currently believes, gathered from your conversations while it
          slept. Open one to see where it came from, correct it, or make it forget.
        </p>
      </header>
      {shown.map(({ theme, facts }) => {
        const [title, ...summary] = text(theme.content).split(": ");
        return (
          <section key={theme.id} className="about-group">
            <h2>
              <button className="heading-button" onClick={() => props.onSelect(theme.id)}>
                {summary.length ? title : "Theme"}
              </button>
            </h2>
            <p className="about-summary">{summary.length ? capital(summary.join(": ")) : title}</p>
            {facts.length > 0 && <ul>{facts.map(row)}</ul>}
          </section>
        );
      })}
      {restShown.length > 0 && (
        <section className="about-group">
          <h2>{groups.length ? "Everything else" : "Things you've mentioned"}</h2>
          <ul>{restShown.map(row)}</ul>
        </section>
      )}
      {q && shown.every((g) => g.facts.length === 0) && restShown.length === 0 && (
        <p className="muted">Nothing current matches “{props.query}”. Try the List view to include outdated ones.</p>
      )}
    </div>
  );
}

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function EmptyMemory() {
  return (
    <div className="stage-msg">
      <h2>Nothing learned yet</h2>
      <p className="muted">
        Memories appear after a sleep cycle reads your conversations. Chat for a while, then start
        one from <a href="#/nights">Nights</a>.
      </p>
    </div>
  );
}
