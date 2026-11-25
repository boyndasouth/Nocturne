import { useEffect, useState } from "react";
import { explain, type ForgetResult, type HistoryEntry, type NoteDetail, type NoteLink } from "../api";
import {
  day,
  episodeQuote,
  linkPhrase,
  plural,
  statusName,
  tierName,
  useApp,
  useLoad,
  useText,
  validRange,
} from "../lib";
import { toStored, toYou } from "../voice";

export function NotePanel(props: {
  id: string;
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  const { api, conn, go, refreshKey, dev } = useApp();
  const text = useText();
  const { data, error, loading, reload } = useLoad(
    () => api.note(props.id),
    [api, props.id, refreshKey],
  );
  const [forgot, setForgot] = useState<ForgetResult | null>(null);
  useEffect(() => setForgot(null), [props.id]);

  return (
    <div className="note-panel">
      <div className="panel-bar">
        {data && (
          <span className={`badge ${data.note.tier} ${data.note.status}`}>
            {tierName[data.note.tier]}
            {data.note.status !== "active" && <>, {statusName[data.note.status].toLowerCase()}</>}
          </span>
        )}
        <button className="icon-button" onClick={props.onClose} aria-label="Close">
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M4 4l8 8M12 4l-8 8" />
          </svg>
        </button>
      </div>

      {error ? (
        <p className="error" role="alert">
          {explain(error, conn)}
        </p>
      ) : !data ? (
        <p className="muted">{loading ? "Loading memory" : ""}</p>
      ) : (
        <>
          <p className={`note-text ${data.note.status}`}>
            {data.note.tier === "episode" ? episodeQuote(data.note.content) : text(data.note.content)}
          </p>
          {forgot && <Forgotten result={forgot} onSelect={props.onSelect} />}
          <ul className="facts-about">
            <li>{validRange(data.note)}</li>
            {data.created_by_cycle && (
              <li>
                {data.note.tier === "episode" ? (
                  <button className="inline-link" onClick={() => go(`nights/${data.created_by_cycle}`)}>
                    See the night it was consolidated
                  </button>
                ) : (
                  <>
                    Learned on the{" "}
                    <button
                      className="inline-link"
                      onClick={() => go(`nights/${data.created_by_cycle}`)}
                    >
                      night of {day(data.note.recorded_at)}
                    </button>
                  </>
                )}
              </li>
            )}
            {data.note.tier !== "episode" && data.note.source_count > 0 && (
              <li>From {plural(data.note.source_count, "conversation turn")}</li>
            )}
            {data.note.access_count > 0 && (
              <li>Used in {plural(data.note.access_count, "answer")}</li>
            )}
            {data.note.status === "archived" && (
              <li>Faded from disuse. It can still come back if nothing current answers a question.</li>
            )}
            {data.note.status === "forgotten" && (
              <li>Won't be used again. It stays in the history so every change can be traced.</li>
            )}
          </ul>

          {!forgot && (
            <Actions
              detail={data}
              onChanged={reload}
              onForgot={(r) => {
                setForgot(r);
                reload();
              }}
            />
          )}

          {dev && (
            <dl className="dev-facts">
              <div>
                <dt>Id</dt>
                <dd>{data.note.id}</dd>
              </div>
              <div>
                <dt>Stored as</dt>
                <dd>{data.note.content}</dd>
              </div>
              <div>
                <dt>Strength</dt>
                <dd>{data.note.strength.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Recorded</dt>
                <dd>{new Date(data.note.recorded_at).toISOString().replace(".000Z", "Z")}</dd>
              </div>
              {data.note.keywords.length > 0 && (
                <div>
                  <dt>Keywords</dt>
                  <dd>{data.note.keywords.join(", ")}</dd>
                </div>
              )}
            </dl>
          )}

          {data.history.length > 1 && (
            <section className="panel-section">
              <h3>History</h3>
              <History entries={data.history} />
            </section>
          )}

          {data.links.length > 0 && (
            <section className="panel-section">
              <h3>Connections</h3>
              <Links links={data.links} onSelect={props.onSelect} />
            </section>
          )}

          {data.sources.length > 0 && (
            <section className="panel-section">
              <h3>What you said</h3>
              <ul className="quotes">
                {data.sources.map((s) => (
                  <li key={s.id} className={s.status}>
                    <blockquote>{episodeQuote(s.content)}</blockquote>
                    <span className="muted">
                      {day(s.valid_from)}
                      {s.status === "forgotten" && ", forgotten"}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}

/** Correct and Forget. Corrections are typed the way the person reads the memory and saved
 * in the stored form; the preview shows exactly what will be saved. */
function Actions(props: {
  detail: NoteDetail;
  onChanged: () => void;
  onForgot: (r: ForgetResult) => void;
}) {
  const { api, conn, bump, dev, readOnly } = useApp();
  const n = props.detail.note;
  const [mode, setMode] = useState<"idle" | "correct" | "forget">("idle");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setMode("idle");
    setErr(null);
  }, [n.id]);

  const canCorrect = !readOnly && n.status === "active" && n.tier !== "episode";
  const canForget = !readOnly && n.status !== "forgotten";
  if (!canCorrect && !canForget) return null;
  const saved = dev ? draft.trim() : toStored(draft);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setErr(null);
    try {
      await action();
      setMode("idle");
      bump();
    } catch (e) {
      setErr(explain(e, conn));
    } finally {
      setBusy(false);
    }
  }

  if (mode === "correct") {
    return (
      <form
        className="action-box"
        onSubmit={(e) => {
          e.preventDefault();
          void run(async () => {
            await api.correct(n.id, saved);
            props.onChanged();
          });
        }}
      >
        <label htmlFor="correction">What's true instead?</label>
        <textarea
          id="correction"
          rows={3}
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
        />
        {!dev && draft.trim() && <p className="fine">Saved as: {saved}</p>}
        {err && (
          <p className="error" role="alert">
            {err}
          </p>
        )}
        <div className="action-row">
          <button type="submit" className="primary" disabled={busy || !draft.trim()}>
            Save correction
          </button>
          <button type="button" className="ghost" onClick={() => setMode("idle")}>
            Cancel
          </button>
        </div>
      </form>
    );
  }

  if (mode === "forget") {
    const turns = n.tier === "fact" ? " and the conversation turns it came from" : "";
    return (
      <div className="action-box warn" role="group" aria-label="Forget this memory">
        <p>
          Forget this{turns}? It won't be used in answers again. The history keeps a record that it
          was forgotten.
        </p>
        {err && (
          <p className="error" role="alert">
            {err}
          </p>
        )}
        <div className="action-row">
          <button
            className="danger"
            disabled={busy}
            onClick={() => void run(async () => props.onForgot(await api.forget(n.id)))}
          >
            Forget it
          </button>
          <button className="ghost" onClick={() => setMode("idle")}>
            Keep it
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="action-row">
      {canCorrect && (
        <button
          className="ghost"
          onClick={() => {
            setDraft(dev ? n.content : toYou(n.content));
            setMode("correct");
          }}
        >
          Correct
        </button>
      )}
      {canForget && (
        <button className="ghost" onClick={() => setMode("forget")}>
          Forget
        </button>
      )}
    </div>
  );
}

function Forgotten({ result, onSelect }: { result: ForgetResult; onSelect: (id: string) => void }) {
  const text = useText();
  return (
    <div className="action-box done" role="status">
      <p>
        Forgotten
        {result.turns_forgotten > 0 && `, along with ${plural(result.turns_forgotten, "conversation turn")}`}
        .
      </p>
      {result.themes.length > 0 && (
        <>
          <p className="small">
            {result.themes.length === 1 ? "This theme still mentions it" : "These themes still mention it"}.
            Correct or forget {result.themes.length === 1 ? "it" : "them"} too:
          </p>
          <ul>
            {result.themes.map((t) => (
              <li key={t.id}>
                <button className="row-button theme-text" onClick={() => onSelect(t.id)}>
                  <span className="dot theme" aria-hidden="true" />
                  {text(t.content)}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function History({ entries }: { entries: HistoryEntry[] }) {
  const { go } = useApp();
  const text = useText();
  return (
    <ol className="history">
      {entries.map((h, i) => {
        const prev = entries[i - 1];
        const changedText = !prev || prev.content !== h.content;
        return (
          <li key={h.version} className={h.status}>
            <div className="history-head">
              <span>Version {h.version}</span>
              <span className="muted">
                {i === entries.length - 1
                  ? statusName[h.status]
                  : h.status === "active"
                    ? "Believed true"
                    : statusName[h.status]}
              </span>
            </div>
            {changedText ? (
              <p className="history-text">{text(h.content)}</p>
            ) : (
              <p className="muted small">Same wording</p>
            )}
            <p className="small">{validRange(h)}</p>
            {h.source_count !== null && prev && prev.source_count !== h.source_count && (
              <p className="small">Now backed by {plural(h.source_count, "turn")}</p>
            )}
            {h.ended && (
              <p className="history-end small">
                {sentence(h.ended.reason)}
                {h.ended.cycle_id ? (
                  <>
                    {" "}on the{" "}
                    <button className="inline-link" onClick={() => go(`nights/${h.ended!.cycle_id}`)}>
                      night of {day(h.ended.cycle_clock ?? h.ended.at)}
                    </button>
                    .
                  </>
                ) : (
                  <> on {day(h.ended.at)}.</>
                )}
              </p>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** Reasons are written for logs ("superseded by <uuid>: moved"); say what happened. */
function sentence(reason: string | null): string {
  if (!reason) return "Changed";
  const why = reason.includes(": ") ? ` (${reason.slice(reason.indexOf(": ") + 2)})` : "";
  if (reason.startsWith("superseded")) return `Replaced by a newer fact${why}`;
  if (reason.startsWith("merged")) return "Heard again in another conversation";
  if (reason.startsWith("theme re-summarised")) return "Rewritten as the theme grew";
  if (reason.startsWith("theme retired")) return "Retired once most of its facts were gone";
  if (reason.startsWith("forgotten at your request")) return "You asked to forget it";
  if (reason.startsWith("forgotten")) return "Faded from disuse";
  if (reason.startsWith("corrected by you")) return "You corrected it";
  return reason.charAt(0).toUpperCase() + reason.slice(1);
}

function Links({ links, onSelect }: { links: NoteLink[]; onSelect: (id: string) => void }) {
  const text = useText();
  const groups = new Map<string, NoteLink[]>();
  for (const l of links) {
    const k = linkPhrase(l.link_type, l.direction);
    groups.set(k, [...(groups.get(k) ?? []), l]);
  }
  return (
    <dl className="links">
      {[...groups].map(([phrase, ls]) => (
        <div key={phrase}>
          <dt>{phrase}</dt>
          {ls.map((l) => (
            <dd key={l.other.id + l.link_type}>
              <button className={`row-button ${l.other.status}`} onClick={() => onSelect(l.other.id)}>
                <span
                  className={`dot ${l.other.status === "superseded" ? "outdated" : l.other.tier}`}
                  aria-hidden="true"
                />
                {l.other.tier === "episode" ? episodeQuote(l.other.content) : text(l.other.content)}
              </button>
            </dd>
          ))}
        </div>
      ))}
    </dl>
  );
}
