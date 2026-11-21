import { useEffect, useRef, useState } from "react";
import type { Connection } from "../api";

export function Settings(props: {
  value: Connection;
  onSave: (c: Connection) => void;
  onClose: () => void;
}) {
  const [url, setUrl] = useState(props.value.url);
  const [apiKey, setApiKey] = useState(props.value.apiKey);
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
          props.onSave({ url: url.trim(), apiKey: apiKey.trim() });
        }}
      >
        <h2>Connection</h2>
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
        <p className="fine">Saved in this browser only.</p>
        <div className="dialog-actions">
          <button type="button" className="ghost" onClick={props.onClose}>
            Cancel
          </button>
          <button type="submit" className="primary">
            Save connection
          </button>
        </div>
      </form>
    </dialog>
  );
}
