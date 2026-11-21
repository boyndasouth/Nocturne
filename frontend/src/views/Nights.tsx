import { useEffect, useState } from "react";
import { explain, type Cycle, type CycleChanges } from "../api";
import { day, dayTime, episodeQuote, linkPhrase, plural, useApp, useLoad } from "../lib";

function summary(c: Cycle): string {
  if (c.status === "running") return "Sleeping now";
  if (c.status === "failed") return "Didn't finish";
  const bits = [`read ${plural(c.episodes_processed, "turn")}`];
  if (c.facts_created) bits.push(`learned ${plural(c.facts_created, "fact")}`);
  if (c.supersessions) bits.push(`retired ${c.supersessions} outdated`);
  const s = bits.join(", ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function Nights({ selected }: { selected: string | null }) {
  const { api, conn, go, refreshKey, bump } = useApp();
  const cycles = useLoad(() => api.cycles(100), [api, refreshKey]);
  const [notice, setNotice] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const list = [...(cycles.data ?? [])].sort((a, b) => b.clock.localeCompare(a.clock));
  const current = selected ?? list[0]?.cycle_id ?? null;
  const running = list.some((c) => c.status === "running");

  // While a cycle runs, check on it every few seconds.
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(bump, 4000);
    return () => window.clearInterval(t);
  }, [running, bump]);

  async function sleepNow() {
    setStarting(true);
    setNotice(null);
    try {
      const r = await api.sleep();
      if (r.status === "nothing_to_do") setNotice("Nothing new to consolidate. Chat first, then sleep.");
      else if (r.cycle) go(`nights/${r.cycle.cycle_id}`);
      bump();
    } catch (e) {
      setNotice(explain(e, conn));
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="nights">
      <aside className="night-list" aria-label="Sleep cycles">
        <button className="primary wide" onClick={() => void sleepNow()} disabled={starting || running}>
          {running ? "Sleeping" : starting ? "Starting" : "Sleep now"}
        </button>
        {notice && (
          <p className="notice" role="status">
            {notice}
          </p>
        )}
        {cycles.error ? (
          <p className="error" role="alert">
            {explain(cycles.error, conn)}
          </p>
        ) : list.length === 0 && cycles.data ? (
          <p className="muted">No nights yet. A cycle runs after a quiet spell, or when you start one.</p>
        ) : (
          <ol>
            {list.map((c) => (
              <li key={c.cycle_id}>
                <a
                  href={`#/nights/${c.cycle_id}`}
                  className={`night-item ${c.status}`}
                  aria-current={c.cycle_id === current ? "page" : undefined}
                >
                  <span className="night-date">{day(c.clock)}</span>
                  <span className="night-sum">{summary(c)}</span>
                </a>
              </li>
            ))}
          </ol>
        )}
      </aside>
      <section className="night-detail">
        {current && <NightDetail key={current} id={current} cycle={list.find((c) => c.cycle_id === current)} />}
      </section>
    </div>
  );
}

function NightDetail({ id, cycle }: { id: string; cycle?: Cycle }) {
  const { api, conn, go, refreshKey } = useApp();
  const ch = useLoad(() => api.changes(id), [api, id, refreshKey]);
  if (ch.error) return <p className="error" role="alert">{explain(ch.error, conn)}</p>;
  if (!ch.data || !cycle) return <p className="muted">Loading the night</p>;
  const d = ch.data;
  const open = (nid: string) => go(`memory/${nid}`);
  const minutes = cycle.finished_at
    ? Math.max(1, Math.round((Date.parse(cycle.finished_at) - Date.parse(cycle.started_at)) / 60000))
    : null;

  return (
    <article className="night">
      <header>
        <h1>The night of {day(d.clock)}</h1>
        <p className="lede">{prose(cycle, d)}</p>
        <p className="fine">
          {cycle.status === "running"
            ? "Still running"
            : `Took ${minutes ? plural(minutes, "minute") : "under a minute"}, ${plural(cycle.llm_calls, "model call")} and ${cycle.tokens_used.toLocaleString()} tokens.`}{" "}
          Clock set to {dayTime(d.clock)}.
        </p>
        {cycle.error && (
          <p className="error" role="alert">
            {cycle.error}
          </p>
        )}
      </header>

      <Changes d={d} open={open} />
    </article>
  );
}

function prose(c: Cycle, d: CycleChanges): string {
  if (c.status === "running") return "It's still reading. Changes show up here when it commits.";
  if (c.status === "failed")
    return "This cycle stopped before committing, so nothing changed. The next sleep resumes it.";
  const parts: string[] = [];
  parts.push(`It read ${plural(c.episodes_processed, "conversation turn")}`);
  const facts = d.created.filter((n) => n.tier === "fact").length;
  if (facts) parts.push(`learned ${plural(facts, "new fact")}`);
  if (c.supersessions) parts.push(`retired ${plural(c.supersessions, "fact")} that stopped being true`);
  if (c.merges) parts.push(`confirmed ${plural(c.merges, "fact")} it already knew`);
  const themes = c.themes_created + c.themes_updated;
  if (themes) parts.push(`wrote or revised ${plural(themes, "theme")}`);
  const forgot = d.episodes_forgotten + d.edits.filter((e) => e.kind === "archived").length;
  if (forgot) parts.push(`let ${forgot} faded ${forgot === 1 ? "memory" : "memories"} go`);
  if (parts.length === 1) return `${parts[0]} and found nothing new to keep.`;
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}.`;
}

function Changes({ d, open }: { d: CycleChanges; open: (id: string) => void }) {
  const facts = d.created.filter((n) => n.tier === "fact");
  const themes = d.created.filter((n) => n.tier === "theme");
  const retired = d.edits.filter((e) => e.kind === "superseded");
  const merged = d.edits.filter((e) => e.kind === "merged");
  const themeEdits = d.edits.filter((e) => e.kind === "theme_updated");
  const archived = d.edits.filter((e) => e.kind === "archived");
  const replacedBy = new Map(
    d.links.filter((l) => l.link_type === "supersedes").map((l) => [l.dst_id, l]),
  );
  const connections = d.links.filter((l) => l.link_type !== "supersedes");
  const verdicts = d.audit?.verdicts ?? [];

  return (
    <>
      {retired.length > 0 && (
        <section className="change replaced">
          <h2>No longer true</h2>
          <ul>
            {retired.map((e) => {
              const by = replacedBy.get(e.note_id);
              return (
                <li key={e.note_id}>
                  <button className="row-button" onClick={() => open(e.note_id)}>
                    <s>{e.before}</s>
                  </button>
                  {by && (
                    <button className="row-button now" onClick={() => open(by.src_id)}>
                      {by.src_content}
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {facts.length > 0 && (
        <section className="change">
          <h2>Learned</h2>
          <ul>
            {facts.map((n) => (
              <li key={n.id}>
                <button className={`row-button ${n.status}`} onClick={() => open(n.id)}>
                  <span className="dot fact" aria-hidden="true" />
                  {n.content}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {merged.length > 0 && (
        <section className="change">
          <h2>Heard again</h2>
          <ul>
            {merged.map((e) => (
              <li key={`${e.note_id}-${e.version}`}>
                <button className="row-button" onClick={() => open(e.note_id)}>
                  <span className="dot fact" aria-hidden="true" />
                  {e.after}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {(themes.length > 0 || themeEdits.length > 0) && (
        <section className="change">
          <h2>Themes</h2>
          <ul>
            {themes.map((n) => (
              <li key={n.id}>
                <button className="row-button theme-text" onClick={() => open(n.id)}>
                  <span className="dot theme" aria-hidden="true" />
                  {n.content}
                </button>
                <span className="fine">New</span>
              </li>
            ))}
            {themeEdits.map((e) => (
              <li key={`${e.note_id}-${e.version}`}>
                <button className="row-button theme-text" onClick={() => open(e.note_id)}>
                  <span className="dot theme" aria-hidden="true" />
                  {e.after}
                </button>
                <span className="fine">Revised. It used to say: {e.before}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {connections.length > 0 && (
        <section className="change">
          <h2>Connected</h2>
          <ul className="connections">
            {connections.map((l) => (
              <li key={`${l.src_id}-${l.dst_id}-${l.link_type}`}>
                <button className="row-button" onClick={() => open(l.src_id)}>
                  {l.src_content}
                </button>
                <span className="relation">{linkPhrase(l.link_type, "out").toLowerCase()}</span>
                <button className="row-button" onClick={() => open(l.dst_id)}>
                  {l.dst_content}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {(archived.length > 0 || d.episodes_forgotten > 0) && (
        <section className="change">
          <h2>Let go</h2>
          {d.episodes_forgotten > 0 && (
            <p className="muted">
              {plural(d.episodes_forgotten, "old conversation turn")} faded after their facts were
              kept.
            </p>
          )}
          <ul>
            {archived.map((e) => (
              <li key={`${e.note_id}-${e.version}`}>
                <button className="row-button archived" onClick={() => open(e.note_id)}>
                  {e.before}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {verdicts.length > 0 && (
        <section className="change">
          <h2>Spot check</h2>
          <p className="muted">
            A second model reviewed {plural(verdicts.length, "decision")} from this night.
          </p>
          <ul className="verdicts">
            {verdicts.map((v) => (
              <li key={v.item} className={v.verdict}>
                <span className="verdict">{verdictName[v.verdict] ?? v.verdict}</span>
                <span>
                  {kindName[v.kind] ?? v.kind}. {v.note}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {d.audit?.error && <p className="fine">The spot check didn't run: {d.audit.error}</p>}

      {d.episodes.length > 0 && (
        <details className="change read">
          <summary>
            <h2>What it read</h2>
            <span className="muted">{plural(d.episodes.length, "turn")}</span>
          </summary>
          <ul className="quotes">
            {d.episodes.map((e) => (
              <li key={e.id}>
                <blockquote>{episodeQuote(e.content)}</blockquote>
                <span className="muted">{dayTime(e.valid_from)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}

const verdictName: Record<string, string> = {
  good: "Sound",
  questionable: "Doubtful",
  wrong: "Wrong",
};
const kindName: Record<string, string> = {
  merge: "Merge",
  supersede: "Replacement",
  link: "Connection",
  theme: "Theme",
  forget: "Forgetting",
};
