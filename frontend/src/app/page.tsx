"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { Button, Empty, PageHead, Panel, Stat, Toggle } from "@/components/ui";
import { GenMsg, Metrics, stream, TokenMsg } from "@/lib/api";

export default function InferenceLab() {
  const [prompt, setPrompt] = useState("The key to understanding transformers is");
  const [maxTokens, setMaxTokens] = useState(40);
  const [useCache, setUseCache] = useState(true);
  const [paged, setPaged] = useState(true);
  const [running, setRunning] = useState(false);

  const [promptTokens, setPromptTokens] = useState<{ id: number; text: string }[]>([]);
  const [tokens, setTokens] = useState<TokenMsg[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<(() => void) | null>(null);

  const run = useCallback(() => {
    setTokens([]);
    setMetrics(null);
    setError(null);
    setPromptTokens([]);
    setRunning(true);

    closeRef.current = stream<GenMsg>(
      "/api/generate",
      {
        prompt,
        max_new_tokens: maxTokens,
        use_cache: useCache,
        paged: useCache && paged,
        temperature: 0.8,
      },
      (m) => {
        if (m.type === "start") setPromptTokens(m.prompt_tokens);
        else if (m.type === "token") setTokens((t) => [...t, m]);
        else if (m.type === "done") {
          setMetrics(m.metrics);
          setRunning(false);
        } else if (m.type === "error") {
          setError(m.message);
          setRunning(false);
        }
      },
      () => setRunning(false)
    );
  }, [prompt, maxTokens, useCache, paged]);

  const stop = () => {
    closeRef.current?.();
    setRunning(false);
  };

  const decodes = useMemo(() => tokens.filter((t) => t.kind === "decode"), [tokens]);
  const maxLat = useMemo(() => Math.max(1, ...tokens.map((t) => t.latency_ms)), [tokens]);
  const maxCache = useMemo(
    () => Math.max(1, ...tokens.map((t) => t.cache_bytes)),
    [tokens]
  );
  const text = useMemo(() => tokens.map((t) => t.token).join(""), [tokens]);

  const last = tokens.length ? tokens[tokens.length - 1] : null;
  // the final token frees its cache, so show the last frame that still had blocks
  const blocks = useMemo(() => {
    for (let i = tokens.length - 1; i >= 0; i--) {
      if (tokens[i].blocks && tokens[i].blocks!.length) return tokens[i].blocks!;
    }
    return null;
  }, [tokens]);
  const peakCacheMB = maxCache / 1024 / 1024;

  return (
    <div className="space-y-5">
      <PageHead kicker="01 — Inference Lab" title="Watch it think.">
        Every token below comes from a GPT-2 built from scratch — hand-written
        attention, KV cache, and sampling. The bars are the engine&apos;s real
        per-token latency, not an animation. Switch the KV cache off and watch what it
        costs to recompute the entire sequence on every single step.
      </PageHead>

      <Panel tone="amber">
        <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
          <div className="space-y-3">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={2}
              disabled={running}
              className="mono w-full resize-none rounded-lg border border-[var(--line)] bg-[var(--panel-2)] px-3 py-2.5 text-sm outline-none focus:border-[var(--accent)] disabled:opacity-60"
              placeholder="Enter a prompt…"
            />
            <div className="flex flex-wrap items-center gap-5">
              <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
                <span>tokens</span>
                <input
                  type="range"
                  min={8}
                  max={120}
                  value={maxTokens}
                  disabled={running}
                  onChange={(e) => setMaxTokens(+e.target.value)}
                  className="w-28 accent-[var(--accent)]"
                />
                <span className="mono w-7 tabular-nums text-[var(--text)]">
                  {maxTokens}
                </span>
              </label>
              <Toggle
                label="KV Cache"
                checked={useCache}
                onChange={setUseCache}
                hint={useCache ? "reuse past K/V" : "recompute everything"}
              />
              <Toggle
                label="PagedAttention"
                checked={paged}
                onChange={setPaged}
                disabled={!useCache}
                hint="16-token blocks"
              />
            </div>
          </div>
          <div className="flex items-start gap-2">
            {running ? (
              <Button variant="ghost" onClick={stop}>
                Stop
              </Button>
            ) : (
              <Button onClick={run}>Generate</Button>
            )}
          </div>
        </div>
      </Panel>

      {error && (
        <div className="rounded-lg border border-[var(--bad)]/40 bg-[var(--bad)]/10 px-4 py-3 text-xs text-[var(--bad)]">
          {error} — is the backend running?
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat
          label="Tokens"
          value={tokens.length}
          hint={running ? "generating…" : metrics ? "complete" : "idle"}
        />
        <Stat
          label="Avg decode"
          value={
            decodes.length
              ? (decodes.reduce((s, t) => s + t.latency_ms, 0) / decodes.length).toFixed(0)
              : "—"
          }
          unit="ms"
          tone={useCache ? "accent" : "bad"}
          hint={useCache ? "with KV cache" : "no cache — recomputing"}
        />
        <Stat
          label="Peak KV cache"
          value={tokens.length ? peakCacheMB.toFixed(2) : "—"}
          unit="MB"
          tone="default"
          hint={last ? `${last.cache_tokens} tokens cached` : ""}
        />
        <Stat
          label="Throughput"
          value={metrics ? metrics.throughput_tok_s.toFixed(1) : "—"}
          unit="tok/s"
          tone="good"
          hint={metrics ? `TTFT ${metrics.avg_ttft_s.toFixed(2)}s` : ""}
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
        <Panel
          title="Token Timeline"
          subtitle="one bar per token — height is the real latency of that forward pass"
        >
          {tokens.length === 0 ? (
            <Empty shape="bars">Press Generate to watch tokens arrive, one forward pass at a time.</Empty>
          ) : (
            <div className="space-y-4">
              <div className="flex h-[140px] items-end gap-[3px] overflow-x-auto pb-1">
                {tokens.map((t, i) => (
                  <div
                    key={i}
                    className="pop group relative flex shrink-0 flex-col items-center"
                    title={`${JSON.stringify(t.token)} — ${t.latency_ms.toFixed(1)}ms`}
                  >
                    <div
                      className="w-[10px] rounded-t"
                      style={{
                        height: `${Math.max(3, (t.latency_ms / maxLat) * 120)}px`,
                        background:
                          t.kind === "prefill"
                            ? "var(--warn-fill)"
                            : "var(--accent-fill)",
                      }}
                    />
                  </div>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-4 text-[10px] text-[var(--dim)]">
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-sm bg-[var(--warn)]" /> prefill — whole
                  prompt in one pass
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-sm bg-[var(--accent)]" /> decode — one
                  token per pass
                </span>
                <span className="mono ml-auto">peak {maxLat.toFixed(0)}ms</span>
              </div>
              {!useCache && tokens.length > 4 && (
                <p className="rounded border border-[var(--bad)]/30 bg-[var(--bad)]/5 px-2.5 py-1.5 text-[10px] leading-snug text-[var(--bad)]">
                  No cache: every step re-processes the whole sequence, so the bars keep
                  climbing. Generating N tokens costs O(N²) work.
                </p>
              )}
            </div>
          )}
        </Panel>

        <Panel title="Output" subtitle="prompt in grey, generated tokens in white">
          {promptTokens.length === 0 && tokens.length === 0 ? (
            <Empty shape="rows">No output yet.</Empty>
          ) : (
            <div className="mono max-h-[190px] overflow-y-auto text-[13px] leading-relaxed">
              {promptTokens.map((t, i) => (
                <span key={`p${i}`} className="text-[var(--dim)]">
                  {t.text}
                </span>
              ))}
              <span>{text}</span>
              {running && (
                <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-[var(--accent)] align-middle" />
              )}
            </div>
          )}
        </Panel>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel
          title="KV Cache Growth"
          subtitle="the price of caching: memory climbs with every token"
        >
          {tokens.length === 0 || !useCache ? (
            <Empty>{useCache ? "—" : "KV cache is off — nothing is stored."}</Empty>
          ) : (
            <>
              <div className="flex h-[90px] items-end gap-[2px]">
                {tokens.map((t, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t bg-[var(--violet-fill)]/85"
                    style={{
                      height: `${Math.max(2, (t.cache_bytes / maxCache) * 80)}px`,
                    }}
                    title={`${t.cache_tokens} tokens · ${(t.cache_bytes / 1024 / 1024).toFixed(2)} MB`}
                  />
                ))}
              </div>
              <p className="mt-3 text-[10px] leading-snug text-[var(--dim)]">
                Every token permanently adds its Keys and Values to the cache. This is why{" "}
                <span className="text-[var(--muted)]">memory, not compute</span>, is what
                limits how many users a GPU can serve at once.
              </p>
            </>
          )}
        </Panel>

        <Panel
          title="Paged Blocks"
          subtitle="physical 16-token blocks, allocated on demand"
        >
          {!paged || !useCache ? (
            <Empty>PagedAttention is off.</Empty>
          ) : !blocks ? (
            <Empty>Run a generation to allocate blocks.</Empty>
          ) : (
            <>
              <div className="flex flex-wrap gap-1.5">
                {blocks.map((b) => (
                  <div
                    key={b}
                    className="mono pop grid h-9 w-14 place-items-center rounded border border-[var(--good)]/40 bg-[var(--good)]/10 text-[10px] text-[var(--good)]"
                    title={`physical block #${b}`}
                  >
                    #{b}
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[10px] leading-snug text-[var(--dim)]">
                {blocks.length} block{blocks.length === 1 ? "" : "s"} × 16 tokens. Blocks
                are handed out only when actually needed, and returned to the pool the
                instant the request finishes — so nothing is reserved for a length the
                request never reaches.
              </p>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
