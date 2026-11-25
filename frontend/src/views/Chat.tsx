import { useEffect, useRef, useState } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { explain, type AnswerTrace, type ChatResponse } from "../api";
import { day, episodeQuote, plural, store, stored, tierName, useApp, useLoad, useText } from "../lib";
import { NotePanel } from "../components/NotePanel";

type Msg =
  | { role: "user"; text: string }
  | { role: "assistant"; res: ChatResponse }
  | { role: "error"; text: string };

const STORE = "nocturne.chat";

function loadMsgs(): Msg[] {
  try {
    return JSON.parse(sessionStorage.getItem(STORE) ?? "[]") as Msg[];
  } catch {
    return [];
  }
}

const SUGGESTIONS = ["Where do I live?", "What am I training for?", "What did I get on my midterm?"];

export function Chat() {
  const { api, conn, bump } = useApp();
  const [msgs, setMsgs] = useState<Msg[]>(loadMsgs);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      sessionStorage.setItem(STORE, JSON.stringify(msgs.slice(-60)));
    } catch {
      // not kept across reloads
    }
    end.current?.scrollIntoView({ block: "end" });
  }, [msgs]);

  async function send(text: string) {
    const t = text.trim();
    if (!t || busy) return;
    setDraft("");
    setMsgs((m) => [...m, { role: "user", text: t }]);
    setBusy(true);
    try {
      const res = await api.chat(t);
      setMsgs((m) => [...m, { role: "assistant", res }]);
      bump(); // a new turn is waiting for sleep
    } catch (e) {
      setMsgs((m) => [...m, { role: "error", text: explain(e, conn) }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={`chat ${open ? "with-drawer" : ""}`}>
      <section className="thread" aria-live="polite">
        <SinceLastTime />
        {msgs.length === 0 ? (
          <div className="empty-chat">
            <h1>Tell it about your life. Ask it later.</h1>
            <p>
              Each message is remembered as it happens. While you're away, a sleep cycle turns those
              turns into facts, retires the ones that stopped being true, and links what belongs
              together.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="ghost" onClick={() => void send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <ol className="messages">
            {msgs.map((m, i) => (
              <li key={i} className={`msg ${m.role}`}>
                {m.role === "user" ? (
                  <p>{m.text}</p>
                ) : m.role === "error" ? (
                  <p role="alert">{m.text}</p>
                ) : (
                  <Answer res={m.res} onCite={setOpen} active={open} />
                )}
              </li>
            ))}
            {busy && (
              <li className="msg assistant thinking" aria-label="Thinking">
                <span />
                <span />
                <span />
              </li>
            )}
          </ol>
        )}
        <div ref={end} />
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            void send(draft);
          }}
        >
          <textarea
            value={draft}
            rows={1}
            placeholder="Write a message"
            aria-label="Message"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(draft);
              }
            }}
          />
          <button type="submit" className="primary" disabled={busy || !draft.trim()}>
            Send
          </button>
        </form>
      </section>
      {open && (
        <aside className="drawer" aria-label="Cited memory">
          <NotePanel id={open} onClose={() => setOpen(null)} onSelect={setOpen} />
        </aside>
      )}
    </div>
  );
}

function Answer({
  res,
  onCite,
  active,
}: {
  res: ChatResponse;
  onCite: (id: string) => void;
  active: string | null;
}) {
  const text = useText();
  const { dev } = useApp();
  const byRef = new Map(res.citations.map((c) => [c.ref, c.note_id]));
  const order = [...byRef.keys()];
  const source = res.answer
    // Citations become links the renderer turns back into buttons.
    .replace(/\[(M\d+)\]/g, (whole, ref: string) =>
      byRef.has(ref) ? `[${order.indexOf(ref) + 1}](#cite-${ref})` : whole,
    )
    // remark-math only knows $…$ and $$…$$; models often write \(…\) and \[…\].
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m: string) => `\n$$\n${m.trim()}\n$$\n`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, m: string) => `$${m.trim()}$`);

  const components: Components = {
    a({ href, children }) {
      const ref = href?.startsWith("#cite-") ? href.slice(6) : null;
      const id = ref ? byRef.get(ref) : undefined;
      if (id) {
        return (
          <button
            className="cite"
            aria-pressed={active === id}
            onClick={() => onCite(id)}
            title="Show this memory"
          >
            {children}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    },
    table({ children }) {
      return (
        <div className="table-wrap">
          <table>{children}</table>
        </div>
      );
    },
  };

  return (
    <>
      <div className="answer">
        <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
          {source}
        </Markdown>
      </div>
      {dev && res.trace ? (
        <Trace trace={res.trace} onOpen={onCite} />
      ) : res.retrieved.length > 0 && (
        <details className="used">
          <summary>Looked at {res.retrieved.length} memories</summary>
          <ul>
            {res.retrieved.map((r) => (
              <li key={r.id}>
                <button className="row-button" onClick={() => onCite(r.id)}>
                  <span className={`dot ${r.tier}`} aria-hidden="true" />
                  <span className="sr-only">{tierName[r.tier]}: </span>
                  {r.tier === "episode" ? episodeQuote(r.preview) : text(r.preview)}
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}

/** "While you were away": the newest finished night the reader hasn't seen yet. */
function SinceLastTime() {
  const { api, go, refreshKey } = useApp();
  const cycles = useLoad(() => api.cycles(20), [api, refreshKey]);
  const [seen, setSeen] = useState(() => stored("nocturne.seenNight"));
  const latest = (cycles.data ?? [])
    .filter((c) => c.status === "succeeded")
    .sort((a, b) => b.clock.localeCompare(a.clock))[0];
  if (!latest || latest.cycle_id === seen) return null;
  const changed = latest.supersessions + latest.merges + latest.themes_created + latest.themes_updated;
  const dismiss = () => {
    store("nocturne.seenNight", latest.cycle_id);
    setSeen(latest.cycle_id);
  };
  return (
    <aside className="since" aria-label="Since you last looked">
      <p>
        While you were away it slept on {day(latest.clock)}: it learned{" "}
        {plural(latest.facts_created, "new thing")}
        {changed > 0 && ` and updated ${plural(changed, "memory", "memories")}`}.
      </p>
      <div className="action-row">
        <button
          className="ghost"
          onClick={() => {
            dismiss();
            go(`nights/${latest.cycle_id}`);
          }}
        >
          See what changed
        </button>
        <button className="link-button" onClick={dismiss}>
          Dismiss
        </button>
      </div>
    </aside>
  );
}

const VIA: Record<string, string> = {
  search: "Matched the question",
  theme: "Member of a matching theme",
  link: "Linked to a matching fact",
  fallback: "Faded memory, used as a fallback",
};

/** Developer view: why the assistant answered as it did. */
function Trace({ trace, onOpen }: { trace: AnswerTrace; onOpen: (id: string) => void }) {
  const top = Math.max(...trace.memories.map((m) => m.score), 0.0001);
  const inPrompt = trace.memories.filter((m) => m.ref).length;
  return (
    <details className="trace">
      <summary>
        Answer trace: {plural(trace.memories.length, "memory", "memories")} retrieved, {inPrompt} in
        the prompt, {trace.memories.filter((m) => m.cited).length} cited
      </summary>
      <p className="trace-meta">
        {trace.model}, {trace.input_tokens.toLocaleString()} tokens in,{" "}
        {trace.output_tokens.toLocaleString()} out, {Math.round(trace.latency_ms).toLocaleString()} ms.
        The prompt also had {plural(trace.recent_turns, "recent turn")} and{" "}
        {plural(trace.history_notes, "earlier version")} of current facts.
      </p>
      <table className="trace-table">
        <thead>
          <tr>
            <th scope="col">Ref</th>
            <th scope="col">Memory</th>
            <th scope="col">Score</th>
            <th scope="col" title="semantic / keyword / recency">Parts</th>
          </tr>
        </thead>
        <tbody>
          {trace.memories.map((m) => (
            <tr key={m.id} className={m.ref ? "" : "dropped"}>
              <td className="mono">
                {m.ref ?? "—"}
                {m.cited && <span className="cited" title="Cited in the answer"> ✓</span>}
              </td>
              <td>
                <button className="row-button" onClick={() => onOpen(m.id)}>
                  <span className={`dot ${m.status === "superseded" ? "outdated" : m.tier}`} aria-hidden="true" />
                  <span>
                    {m.tier === "episode" ? episodeQuote(m.content) : m.content}
                    <span className="trace-via">
                      {VIA[m.via] ?? m.via}
                      {m.status !== "active" && `, ${m.status}`}
                      {!m.ref && ", left out of the prompt"}
                    </span>
                  </span>
                </button>
              </td>
              <td>
                <span className="bar" style={{ width: `${(m.score / top) * 100}%` }} aria-hidden="true" />
                {m.score.toFixed(3)}
              </td>
              <td className="mono small">
                {m.semantic.toFixed(2)} / {m.keyword.toFixed(2)} / {m.recency.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {trace.messages.map((msg, i) => (
        <details key={i} className="prompt">
          <summary>
            {msg.role === "system" ? "System prompt" : "Message sent with memories"} (
            {msg.content.length.toLocaleString()} characters)
          </summary>
          <pre>{msg.content}</pre>
        </details>
      ))}
    </details>
  );
}
