import { useEffect, useRef, useState } from "react";
import type { Connection } from "../api";
import { authEnabled } from "../auth";

export function Settings(props: {
  value: Connection;
  dev: boolean;
  onSave: (c: Connection, dev: boolean) => void;
  onClose: () => void;
}) {
  const [url, setUrl] = useState(props.value.url);
  const [apiKey, setApiKey] = useState(props.value.apiKey);
  const [dev, setDev] = useState(props.dev);
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal();
  }, []);

  return (
    <dialog ref={ref} className="dialog" onClose={props.onClose}>
      <form
        method="dialog"
        onSubmit={(e) => {
          e.preventDefault();
          props.onSave({ url: url.trim(), apiKey: apiKey.trim() }, dev);
        }}
      >
        <h2>Settings</h2>
        <label>
          Backend URL
          <input
            type="url"
            required
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="http://localhost:8000"
          />
        </label>
{!authEnabled && (
        <label>
          API key
          <input
            type="password"
            value={apiKey}
            autoComplete="off"
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Leave empty if the backend has no API_KEY"
          />
        </label>
        )}
        <label className="check">
          <input type="checkbox" checked={dev} onChange={(e) => setDev(e.target.checked)} />
          <span>
            Show developer details
            <span className="fine">
              Memories as stored for the model, ids, scores, model calls and tokens.
            </span>
          </span>
        </label>
        <p className="fine">Saved in this browser only.</p>
        <div className="dialog-actions">
          <button type="button" className="ghost" onClick={props.onClose}>
            Cancel
          </button>
          <button type="submit" className="primary">
            Save settings
          </button>
        </div>
      </form>
    </dialog>
  );
}
