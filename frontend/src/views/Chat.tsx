import { useEffect, useRef, useState, type ReactNode } from "react";
import { explain, type ChatResponse } from "../api";
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
  const byRef = new Map(res.citations.map((c) => [c.ref, c.note_id]));
  const order = [...byRef.keys()];
  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of res.answer.matchAll(/\[(M\d+)\]/g)) {
    const id = byRef.get(m[1]);
    parts.push(res.answer.slice(last, m.index));
    if (id) {
      parts.push(
        <button
          key={m.index}
          className="cite"
          aria-pressed={active === id}
          onClick={() => onCite(id)}
          title="Show this memory"
        >
          {order.indexOf(m[1]) + 1}
        </button>,
      );
    } else parts.push(m[0]);
    last = (m.index ?? 0) + m[0].length;
  }
  parts.push(res.answer.slice(last));

  return (
    <>
      <p className="answer">{parts}</p>
      {res.retrieved.length > 0 && (
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
