import { explain, type HistoryEntry, type NoteLink } from "../api";
import {
  day,
  episodeQuote,
  linkPhrase,
  plural,
  statusName,
  tierName,
  useApp,
  useLoad,
  validRange,
} from "../lib";

export function NotePanel(props: {
  id: string;
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  const { api, conn, go, refreshKey } = useApp();
  const { data, error, loading } = useLoad(() => api.note(props.id), [api, props.id, refreshKey]);

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
            {data.note.tier === "episode" ? episodeQuote(data.note.content) : data.note.content}
          </p>
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
              <li>Recalled in {plural(data.note.access_count, "answer")}</li>
            )}
          </ul>

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
                  <li key={s.id}>
                    <blockquote>{episodeQuote(s.content)}</blockquote>
                    <span className="muted">{day(s.valid_from)}</span>
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

function History({ entries }: { entries: HistoryEntry[] }) {
  const { go } = useApp();
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
              <p className="history-text">{h.content}</p>
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
                  "."
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
  if (reason.startsWith("forgotten")) return "Faded from disuse";
  return reason.charAt(0).toUpperCase() + reason.slice(1);
}

function Links({ links, onSelect }: { links: NoteLink[]; onSelect: (id: string) => void }) {
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
                {l.other.content}
              </button>
            </dd>
          ))}
        </div>
      ))}
    </dl>
  );
}
