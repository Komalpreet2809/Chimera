"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";
import { RuntimeToken, useRuntime } from "@/components/runtime";

/** Pages that actually read the selected token. Elsewhere the dock is noise. */
const SYNCED_ROUTES = ["/", "/scheduler", "/memory", "/attention"];

export function RuntimeDock() {
  const runtime = useRuntime();
  const pathname = usePathname();
  const token = runtime.tokens[runtime.selected];

  if (!token || !SYNCED_ROUTES.includes(pathname)) return null;

  return (
    <div className="runtime-dock">
      <div>
        <span className="section-label">Selected token #{token.index}</span>
        <strong className="mono">{token.token.trim() || "space"}</strong>
      </div>
      <button
        onClick={() => runtime.select(runtime.selected - 1)}
        disabled={runtime.selected === 0}
        aria-label="Previous token"
      >
        ‹
      </button>
      <button
        onClick={() => runtime.select(runtime.selected + 1)}
        disabled={runtime.selected >= runtime.tokens.length - 1}
        aria-label="Next token"
      >
        ›
      </button>
      <span className="mono dock-readout">
        {token.latency_ms.toFixed(1)}ms · {(token.probability * 100).toFixed(2)}%
      </span>
      <span className="dock-hint">← → to step</span>
    </div>
  );
}

export function TokenRail({
  tokens,
  selected,
  onSelect,
}: {
  tokens: RuntimeToken[];
  selected: number;
  onSelect: (index: number) => void;
}) {
  const railRef = useRef<HTMLDivElement>(null);

  // Keep the selection visible: the rail overflows well before 40 tokens, so
  // stepping with the keyboard or replay controls would otherwise walk the
  // selected token off-screen.
  useEffect(() => {
    const el = railRef.current?.querySelector<HTMLElement>('[aria-selected="true"]');
    el?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }, [selected, tokens.length]);

  return (
    <div
      ref={railRef}
      className="runtime-rail"
      role="listbox"
      aria-label="Generated token timeline"
    >
      {tokens.map((token, index) => (
        <button
          key={`${token.index}-${token.token_id}`}
          role="option"
          aria-selected={index === selected}
          onClick={() => onSelect(index)}
          className={`runtime-token ${index === selected ? "is-selected" : ""}`}
          title={`Token ${token.index} · ${token.latency_ms.toFixed(1)}ms · ${(
            token.probability * 100
          ).toFixed(1)}% probability`}
        >
          <span className="runtime-token-index">{token.index}</span>
          <span className="runtime-token-text">{token.token.trim() || "␣"}</span>
          <span className="runtime-token-time">+{token.elapsed_ms.toFixed(0)}ms</span>
        </button>
      ))}
    </div>
  );
}

export function TokenInspector({
  token,
  sampling,
}: {
  token?: RuntimeToken;
  /** How this trace was actually decoded — never assume, the caller knows. */
  sampling: string;
}) {
  if (!token) {
    return (
      <div className="inspector-empty">
        <span className="mono">TOKEN INSPECTOR</span>
        <strong>Generate, then click any token.</strong>
        <p>Latency, memory, sampling and causal events will meet here.</p>
      </div>
    );
  }

  const block = token.blocks?.at(-1);
  const cached = token.cache_tokens > 0;
  const rows: [string, string | number][] = [
    ["Token ID", token.token_id],
    ["Probability", `${(token.probability * 100).toFixed(2)}%`],
    ["Sampling", sampling],
    ["Forward pass", token.kind === "prefill" ? "Prefill (whole prompt)" : "Decode (1 token)"],
    ["Latency", `${token.latency_ms.toFixed(1)} ms`],
    ["Elapsed", `${token.elapsed_ms.toFixed(0)} ms`],
    // "Hit/miss" would be a fiction: with the cache on, every decode step
    // reuses it by construction — there is no miss to report.
    ["KV cache", cached ? `Reusing ${token.cache_tokens} tokens` : "Disabled"],
    ["Memory", block === undefined ? "Contiguous" : `Paged · block #${block}`],
  ];

  return (
    <aside className="token-inspector">
      <div className="section-label text-[var(--dim)]">Token Inspector · #{token.index}</div>
      <div className="mono inspector-token">{token.token.trim() || "space"}</div>
      <dl>
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <Link className="inspector-link" href="/attention">
        Inspect this token&rsquo;s attention →
      </Link>
    </aside>
  );
}

export function EngineLog({
  tokens,
  selected,
}: {
  tokens: RuntimeToken[];
  selected: number;
}) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = logRef.current?.querySelector<HTMLElement>(".is-current");
    el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selected, tokens.length]);

  // Every line below is derived from something the engine actually reported.
  const rows = tokens.flatMap((token, index) => {
    const at = formatElapsed(token.elapsed_ms);
    const block = token.blocks?.at(-1);
    const prefill = token.kind === "prefill";
    return [
      {
        index,
        at,
        event: prefill ? "PREFILL" : "DECODE",
        detail: prefill
          ? `prompt processed in one pass · ${token.latency_ms.toFixed(1)}ms`
          : `1 token in, attending over ${token.cache_tokens - 1} cached · ${token.latency_ms.toFixed(1)}ms`,
      },
      {
        index,
        at,
        event: token.cache_tokens > 0 ? "KV_CACHE_APPEND" : "KV_CACHE_OFF",
        detail:
          token.cache_tokens > 0
            ? `cache now holds ${token.cache_tokens} tokens`
            : "no cache — whole sequence recomputed",
      },
      ...(block === undefined
        ? []
        : [{ index, at, event: "BLOCK_ACTIVE", detail: `physical block #${block}` }]),
      {
        index,
        at,
        event: "TOKEN_COMMIT",
        detail: `id=${token.token_id} · text=${JSON.stringify(token.token)} · p=${(
          token.probability * 100
        ).toFixed(2)}%`,
      },
    ];
  });

  return (
    <div ref={logRef} className="engine-log mono" role="log" aria-label="Chronological engine log">
      {rows.length === 0 ? (
        <div className="engine-log-seed">
          <span>00:00.000</span>
          <b>READY</b>
          <span>Waiting for a generation event…</span>
        </div>
      ) : (
        rows.map((row, i) => (
          <div
            key={`${row.index}-${row.event}-${i}`}
            className={row.index === selected ? "is-current" : ""}
          >
            <span>{row.at}</span>
            <b>{row.event}</b>
            <span>{row.detail}</span>
          </div>
        ))
      )}
    </div>
  );
}

function formatElapsed(milliseconds: number) {
  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = Math.floor(milliseconds / 1_000) % 60;
  const millis = Math.floor(milliseconds % 1_000);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(
    millis
  ).padStart(3, "0")}`;
}

export function ReplayControls({
  count,
  selected,
  onSelect,
}: {
  count: number;
  selected: number;
  onSelect: (index: number) => void;
}) {
  const disabled = count === 0;
  return (
    <div className="replay-controls">
      <button disabled={disabled || selected === 0} onClick={() => onSelect(0)} aria-label="First token">
        |‹
      </button>
      <button
        disabled={disabled || selected === 0}
        onClick={() => onSelect(selected - 1)}
        aria-label="Previous token"
      >
        ‹
      </button>
      <div>
        <strong className="mono">
          Step {disabled ? 0 : selected + 1} / {count}
        </strong>
        <span>
          Replay “what just happened?” one event at a time — or use ← →.
        </span>
      </div>
      <button
        disabled={disabled || selected >= count - 1}
        onClick={() => onSelect(selected + 1)}
        aria-label="Next token"
      >
        ›
      </button>
      <button
        disabled={disabled || selected >= count - 1}
        onClick={() => onSelect(count - 1)}
        aria-label="Last token"
      >
        ›|
      </button>
    </div>
  );
}
