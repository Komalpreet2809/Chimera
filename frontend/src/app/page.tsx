"use client";

import { useCallback, useRef, useState } from "react";
import { Button, PageHead, Panel, Toggle } from "@/components/ui";
import { EngineLog, ReplayControls, TokenInspector, TokenRail } from "@/components/runtime-debugger";
import Link from "next/link";
import { RuntimeToken, useRuntime } from "@/components/runtime";
import { API_BASE, GenMsg, Metrics, stream } from "@/lib/api";

// Decoding settings, in one place, so the request and the label describing it
// can never drift apart. top_k is fixed server-side at 40.
const TEMPERATURE = 0.8;
const TOP_K = 40;

export default function InferenceLab() {
  const runtime = useRuntime();
  const [prompt, setPrompt] = useState("The key to understanding transformers is");
  const [maxTokens, setMaxTokens] = useState(40);
  const [useCache, setUseCache] = useState(true);
  const [paged, setPaged] = useState(true);
  const [compare, setCompare] = useState(false);
  const [running, setRunning] = useState(false);
  const [promptTokens, setPromptTokens] = useState<{ id: number; text: string }[]>([]);
  const [uncached, setUncached] = useState<RuntimeToken[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [uncachedMetrics, setUncachedMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closes = useRef<(() => void)[]>([]);
  const startedAt = useRef(0);

  const normalize = useCallback((message: Extract<GenMsg, { type: "token" }>): RuntimeToken => ({
    ...message,
    probability: message.probability ?? 0,
    elapsed_ms: performance.now() - startedAt.current,
  }), []);

  const run = useCallback(() => {
    closes.current.forEach((close) => close());
    runtime.clear();
    setUncached([]);
    setPromptTokens([]);
    setMetrics(null);
    setUncachedMetrics(null);
    setError(null);
    setRunning(true);
    startedAt.current = performance.now();

    const cachedClose = stream<GenMsg>(
      "/api/generate",
      {
        prompt,
        max_new_tokens: maxTokens,
        use_cache: useCache,
        paged: useCache && paged,
        temperature: TEMPERATURE,
        // Comparison only means anything if both lanes are deterministic,
        // so Compare forces greedy. The Sampling readout says so.
        greedy: compare,
      },
      (message) => {
        if (message.type === "start") setPromptTokens(message.prompt_tokens);
        if (message.type === "token") runtime.appendToken(prompt, normalize(message));
        if (message.type === "done") {
          setMetrics(message.metrics);
          if (!compare) setRunning(false);
        }
        if (message.type === "error") {
          setError(message.message);
          setRunning(false);
        }
      },
      () => setRunning(false)
    );
    closes.current = [cachedClose];

    if (compare) {
      const comparisonStart = performance.now();
      const compareClose = stream<GenMsg>(
        "/api/generate",
        {
          prompt,
          max_new_tokens: maxTokens,
          use_cache: false,
          paged: false,
          temperature: TEMPERATURE,
          greedy: true,
        },
        (message) => {
          if (message.type === "token") {
            setUncached((current) => [...current, {
              ...message,
              probability: message.probability ?? 0,
              elapsed_ms: performance.now() - comparisonStart,
            }]);
          }
          if (message.type === "done") {
            setUncachedMetrics(message.metrics);
            setRunning(false);
          }
          if (message.type === "error") {
            setError(message.message);
            setRunning(false);
          }
        },
        () => setRunning(false)
      );
      closes.current.push(compareClose);
    }
  }, [compare, maxTokens, normalize, paged, prompt, runtime, useCache]);

  const stop = () => {
    closes.current.forEach((close) => close());
    setRunning(false);
  };

  const token = runtime.tokens[runtime.selected];
  const activeBlock = token?.blocks?.at(-1);
  const sampling = compare
    ? "greedy (argmax)"
    : `temperature ${TEMPERATURE} · top-k ${TOP_K}`;
  return (
    <div className="space-y-5">
      <PageHead kicker="01 — Runtime Debugger" title="Inspect every token.">
        Generation is a causal sequence, not a scorecard. Select any output token and
        the timeline, cache, memory, scheduler, sampling and engine log snap to the exact
        event that produced it.
      </PageHead>

      <Panel tone="amber">
        <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
          <div className="space-y-3">
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={2} disabled={running}
              className="mono w-full resize-none rounded-lg border border-[var(--line)] bg-[var(--panel-2)] px-3 py-2.5 text-sm outline-none focus:border-[var(--accent)] disabled:opacity-60" />
            <div className="flex flex-wrap items-center gap-5">
              <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
                <span>tokens</span><input type="range" min={8} max={80} value={maxTokens} disabled={running}
                  onChange={(event) => setMaxTokens(+event.target.value)} className="w-28 accent-[var(--accent)]" />
                <span className="mono w-7 text-[var(--text)]">{maxTokens}</span>
              </label>
              <Toggle label="KV Cache" checked={useCache} onChange={setUseCache} hint={useCache ? "reuse past K/V" : "recompute everything"} />
              <Toggle label="PagedAttention" checked={paged} onChange={setPaged} disabled={!useCache} hint="16-token blocks" />
              <Toggle label="Compare" checked={compare} onChange={setCompare} hint="cached vs uncached replay" />
            </div>
          </div>
          <div>{running ? <Button variant="ghost" onClick={stop}>Stop</Button> : <Button onClick={run}>Generate trace</Button>}</div>
        </div>
      </Panel>

      {error ? <div className="error-banner">{error} — is the backend running at {API_BASE}?</div> : null}

      <section className="debugger-shell">
        <div className="debugger-main">
          <div className="debugger-heading">
            <div><span className="section-label text-[var(--dim)]">Token timeline</span><h3>One token, one chain of events.</h3></div>
            <span className="mono debugger-status">{running ? "● RECORDING" : runtime.tokens.length ? `${runtime.tokens.length} EVENTS` : "READY"}</span>
          </div>

          {runtime.tokens.length ? (
            <TokenRail tokens={runtime.tokens} selected={runtime.selected} onSelect={runtime.select} />
          ) : (
            <button className="debugger-primer" onClick={run}>
              <span className="primer-play">▶</span>
              <span><strong>Record your first inference trace</strong><small>Watch prefill → attention → cache update → token commit happen in order.</small></span>
            </button>
          )}

          <div className="causal-strip" aria-label="Selected token causal path">
            {[
              ["01", token?.kind === "prefill" ? "Prompt prefilled" : "Attention computed", token ? `forward pass · ${token.latency_ms.toFixed(1)} ms` : "model reads context"],
              // Not "hit/miss": with the cache on every decode step reuses it
              // by construction, so the honest axis is reuse vs recompute.
              ["02", token && token.cache_tokens > 0 ? "KV cache reused" : "Full recompute", token ? (token.cache_tokens > 0 ? `${token.cache_tokens} tokens reused` : "no cache — whole sequence again") : "reuse or recompute"],
              ["03", activeBlock === undefined ? "Contiguous memory" : `Block #${activeBlock} active`, token ? `${(token.cache_bytes / 1024 / 1024).toFixed(2)} MB resident` : "memory is allocated"],
              ["04", token ? `Token ${JSON.stringify(token.token)}` : "Token committed", token ? `${(token.probability * 100).toFixed(2)}% sampled probability` : "output becomes visible"],
            ].map(([step, title, detail]) => (
              <div key={step} className="causal-event"><span>{step}</span><strong>{title}</strong><small>{detail}</small></div>
            ))}
          </div>

          <div className="debugger-grid">
            <div className="runtime-output">
              <div className="debugger-subhead"><b>Model output</b><span>click a token to inspect it</span></div>
              <div className="mono output-trace">
                <span className="prompt-trace">{promptTokens.map((item) => item.text).join("")}</span>
                {runtime.tokens.map((item, index) => (
                  <button key={item.index} className={index === runtime.selected ? "is-selected" : ""} onClick={() => runtime.select(index)}>{item.token}</button>
                ))}
                {running ? <i className="trace-cursor" /> : null}
              </div>
            </div>
            <div className="runtime-snapshot">
              <div className="debugger-subhead"><b>Linked runtime state</b><span>same event, four views</span></div>
              <div className="snapshot-grid">
                {/* A single generate() request never enters the scheduler, so
                    there is no seat to report — say so, and point at the page
                    where batching actually happens. */}
                <LinkCard href="/scheduler" label="Scheduler" value="Not batched" detail="single request — see batching" active={Boolean(token)} />
                <LinkCard href="/memory" label="KV memory" value={activeBlock === undefined ? "Contiguous" : `Block #${activeBlock}`} detail={`${token?.cache_tokens ?? 0} tokens resident`} active={Boolean(token)} />
                <LinkCard href="/attention" label="Attention" value="Current token" detail="open causal relationships" active={Boolean(token)} />
                <div className={`snapshot-card ${token ? "is-active" : ""}`}><span>Sampling</span><strong>{token ? `${(token.probability * 100).toFixed(2)}%` : "—"}</strong><small>{sampling}</small></div>
              </div>
            </div>
          </div>

          <div className="debugger-subhead log-head"><b>Engine Log</b><span>chronological · synchronized to token #{token?.index ?? 0}</span></div>
          <EngineLog tokens={runtime.tokens} selected={runtime.selected} />
          <div className="guided-replay"><span className="section-label">Guided Replay</span><ReplayControls count={runtime.tokens.length} selected={runtime.selected} onSelect={runtime.select} /></div>
        </div>
        <TokenInspector token={token} sampling={sampling} />
      </section>

      {compare ? (
        <Panel title="Comparison replay" subtitle="the same prompt, cached and uncached, aligned by output position">
          <div className="compare-grid">
            <CompareLane title="With KV Cache" tokens={runtime.tokens} metrics={metrics} tone="good" selected={runtime.selected} />
            <CompareLane title="Without KV Cache" tokens={uncached} metrics={uncachedMetrics} tone="bad" selected={runtime.selected} />
          </div>
        </Panel>
      ) : null}
    </div>
  );
}

function LinkCard({ href, label, value, detail, active }: { href: string; label: string; value: string; detail: string; active: boolean }) {
  return <Link href={href} className={`snapshot-card ${active ? "is-active" : ""}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></Link>;
}

function CompareLane({ title, tokens, metrics, tone, selected }: { title: string; tokens: RuntimeToken[]; metrics: Metrics | null; tone: "good" | "bad"; selected: number }) {
  const token = tokens[selected];
  return (
    <div className={`compare-lane ${tone}`}>
      <div><strong>{title}</strong><span className="mono">{token ? `${token.latency_ms.toFixed(1)} ms` : "waiting…"}</span></div>
      <div className="compare-bars">{tokens.map((item, index) => <i key={item.index} className={index === selected ? "is-selected" : ""} style={{ height: `${Math.max(4, Math.min(52, item.latency_ms / 2))}px` }} />)}</div>
      <p>{metrics ? `${metrics.throughput_tok_s.toFixed(1)} tok/s · TTFT ${metrics.avg_ttft_s.toFixed(2)}s` : `${tokens.length} tokens recorded`}</p>
    </div>
  );
}
