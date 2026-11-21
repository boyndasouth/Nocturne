import { useState } from "react";
import { providers, supabase, type Provider } from "../auth";

type Mode = "link" | "password" | "create";
const PROVIDER_NAME: Record<Provider, string> = { github: "GitHub", google: "Google" };

export function SignIn() {
  const [mode, setMode] = useState<Mode>("link");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const redirectTo = window.location.origin + window.location.pathname;

  async function submit() {
    if (!supabase) return;
    setBusy(true);
    setError(null);
    try {
      if (mode === "link") {
        const { error } = await supabase.auth.signInWithOtp({
          email,
          options: { emailRedirectTo: redirectTo },
        });
        if (error) throw error;
        setSent(`We sent a sign-in link to ${email}. Open it on this device.`);
      } else if (mode === "password") {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
      } else {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: { emailRedirectTo: redirectTo },
        });
        if (error) throw error;
        if (!data.session) setSent(`Confirm your address from the email we sent to ${email}.`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sign-in failed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function oauth(p: Provider) {
    if (!supabase) return;
    setError(null);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: p,
      options: { redirectTo },
    });
    if (error) setError(error.message);
  }

  return (
    <main className="signin">
      <Example />
      <div className="signin-inner">
        <p className="wordmark static">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M15.2 3.2a8.6 8.6 0 1 0 5.6 13.6A7 7 0 0 1 15.2 3.2z" />
          </svg>
          Nocturne
        </p>
        <h1>An assistant that remembers, and shows its work.</h1>
        <p className="signin-lede">
          Tell it about your life. While you're away it sleeps on what you said: it keeps the
          facts, retires what stopped being true and connects the rest. You can read, correct or
          erase every memory it has of you.
        </p>

        {sent ? (
          <div className="action-box done" role="status">
            <p>{sent}</p>
            <button className="link-button" onClick={() => setSent(null)}>
              Use a different address
            </button>
          </div>
        ) : (
          <form
            className="signin-form"
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            <label>
              Email
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>
            {mode !== "link" && (
              <label>
                Password
                <input
                  type="password"
                  required
                  minLength={8}
                  autoComplete={mode === "create" ? "new-password" : "current-password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </label>
            )}
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
            <button type="submit" className="primary wide" disabled={busy}>
              {mode === "link" ? "Email me a sign-in link" : mode === "password" ? "Sign in" : "Create account"}
            </button>
            {providers.map((p) => (
              <button key={p} type="button" className="ghost wide" onClick={() => void oauth(p)}>
                Continue with {PROVIDER_NAME[p]}
              </button>
            ))}
            <p className="signin-switch">
              {mode === "link" ? (
                <button type="button" className="link-button" onClick={() => setMode("password")}>
                  Use a password instead
                </button>
              ) : mode === "password" ? (
                <>
                  <button type="button" className="link-button" onClick={() => setMode("create")}>
                    Create an account
                  </button>
                  <button type="button" className="link-button" onClick={() => setMode("link")}>
                    Email me a link instead
                  </button>
                </>
              ) : (
                <button type="button" className="link-button" onClick={() => setMode("password")}>
                  I already have an account
                </button>
              )}
            </p>
          </form>
        )}
        <p className="fine signin-foot">
          Your memory is private to your account. Building an assistant? Turn on developer details
          in Settings to see why it answered the way it did.
        </p>
      </div>
    </main>
  );
}

/** What a night looks like, beside the form on wide screens. */
function Example() {
  return (
    <aside className="signin-example" aria-label="Example of one night's changes">
      <p className="example-tag">An example night</p>
      <h2>The night of March 3</h2>
      <p className="example-lede">
        It read 4 conversation turns, learned 4 new facts and retired 2 that stopped being true.
      </p>
      <h3>No longer true</h3>
      <p>
        <s>You live in Austin, near the UT campus.</s>
      </p>
      <p className="example-now">You live in Dallas.</p>
      <h3>Learned</h3>
      <p>
        <span className="dot fact" aria-hidden="true" /> You're doing a spring co-op at a fintech
        company in Dallas.
      </p>
      <p>
        <span className="dot fact" aria-hidden="true" /> You're considering the Dallas marathon in
        December.
      </p>
    </aside>
  );
}
