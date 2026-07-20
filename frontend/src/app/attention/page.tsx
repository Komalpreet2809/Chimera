"use client";

import { useCallback, useEffect, useState } from "react";
import { Button, Empty, PageHead, Panel } from "@/components/ui";
import { AttentionResp, post } from "@/lib/api";

export default function AttentionPage() {
  const [text, setText] = useState(
    "The animal didn't cross the street because it was too tired"
  );
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [data, setData] = useState<AttentionResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await post<AttentionResp>("/api/attention", { text, layer, head }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [text, layer, head]);

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layer, head]);

  const toks = data?.tokens.map((t) => t.text.trim() || "␣") ?? [];

  return (
    <div className="space-y-5">
      <PageHead kicker="04 — Attention" title="Who looks at whom?">
          Attention is the only place tokens look at each other. Each row is a token
          asking <em>&ldquo;who is relevant to me?&rdquo;</em>; each column is a token
          being looked at. The grid is strictly lower-triangular because a token may only
          see the <span className="text-[var(--text)]">past</span> — that causal rule is
          what makes left-to-right generation possible. These are real weights pulled from
          the running model — all {data?.n_layer ?? 12} layers ×{" "}
          {data?.n_head ?? 12} heads are browsable.
      </PageHead>

      <Panel>
        <div className="space-y-3">
          <div className="flex gap-2">
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="mono flex-1 rounded-lg border border-[var(--line)] bg-[var(--panel-2)] px-3 py-2 text-sm outline-none focus:border-[var(--accent)]"
            />
            <Button onClick={run} disabled={loading}>
              {loading ? "…" : "Analyze"}
            </Button>
          </div>
          <div className="flex flex-wrap items-center gap-6">
            <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
              <span>layer</span>
              <input
                type="range"
                min={0}
                max={(data?.n_layer ?? 12) - 1}
                value={layer}
                onChange={(e) => setLayer(+e.target.value)}
                className="w-36 accent-[var(--accent)]"
              />
              <span className="mono w-8 tabular-nums text-[var(--text)]">{layer}</span>
            </label>
            <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
              <span>head</span>
              <input
                type="range"
                min={0}
                max={(data?.n_head ?? 12) - 1}
                value={head}
                onChange={(e) => setHead(+e.target.value)}
                className="w-36 accent-[var(--accent)]"
              />
              <span className="mono w-8 tabular-nums text-[var(--text)]">{head}</span>
            </label>
            <span className="mono text-[10px] text-[var(--dim)]">
              drag to browse all {(data?.n_layer ?? 12) * (data?.n_head ?? 12)} heads —
              each learns a different relationship
            </span>
          </div>
        </div>
      </Panel>

      {error && (
        <div className="rounded-lg border border-[var(--bad)]/40 bg-[var(--bad)]/10 px-4 py-3 text-xs text-[var(--bad)]">
          {error} — is the backend running on port 8000?
        </div>
      )}

      <Panel
        title={`Attention weights — layer ${layer}, head ${head}`}
        subtitle="brighter = more attention. Row = who's asking, column = who's being looked at."
      >
        {!data ? (
          <Empty>Loading…</Empty>
        ) : (
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              {/* column headers */}
              <div className="flex">
                <div className="w-24 shrink-0" />
                {toks.map((t, i) => (
                  <div
                    key={i}
                    className="mono w-[34px] shrink-0 text-center text-[9px] text-[var(--dim)]"
                    style={{
                      color: hover?.c === i ? "var(--accent)" : undefined,
                    }}
                  >
                    <span className="inline-block max-w-[32px] truncate">{t}</span>
                  </div>
                ))}
              </div>
              {/* rows */}
              {data.attention.map((row, r) => (
                <div key={r} className="flex items-center">
                  <div
                    className="mono w-24 shrink-0 truncate pr-2 text-right text-[10px]"
                    style={{
                      color: hover?.r === r ? "var(--accent)" : "var(--muted)",
                    }}
                  >
                    {toks[r]}
                  </div>
                  {row.map((v, c) => (
                    <div
                      key={c}
                      onMouseEnter={() => setHover({ r, c })}
                      onMouseLeave={() => setHover(null)}
                      className="mono grid h-[30px] w-[34px] shrink-0 place-items-center border border-[var(--bg)] text-[9px] transition-transform hover:scale-110 hover:border-[var(--accent)]"
                      style={{
                        background:
                          c > r
                            ? "var(--panel-2)"
                            : `rgba(77,163,255,${Math.max(0.04, Math.pow(v, 0.55))})`,
                        color: v > 0.5 ? "#06121f" : "var(--muted)",
                      }}
                      title={
                        c > r
                          ? "masked — cannot see the future"
                          : `${toks[r]} → ${toks[c]}: ${(v * 100).toFixed(1)}%`
                      }
                    >
                      {c <= r && v >= 0.08 ? (v * 100).toFixed(0) : ""}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="mt-4 flex flex-wrap items-center gap-4 text-[10px] text-[var(--dim)]">
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-[var(--panel-2)]" /> masked (the
            future — a token can never look ahead)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-sm bg-[var(--accent)]" /> strong
            attention
          </span>
          <span className="ml-auto">every row sums to 100%</span>
        </div>
      </Panel>

      <Panel title="What am I looking at?" tone="dark">
        <p className="text-xs leading-relaxed text-[var(--muted)]">
          Each of the {(data?.n_layer ?? 12) * (data?.n_head ?? 12)} heads specialises.
          Some track syntax (which verb belongs to which subject), some resolve pronouns,
          some just look at the previous token. Many heads — especially in early layers —
          dump most of their weight on the very first token: that&apos;s a real, documented
          phenomenon called an <span className="text-[var(--text)]">attention sink</span>,
          the model&apos;s way of saying &ldquo;nothing here is relevant to me,&rdquo;
          parking its attention somewhere harmless rather than being forced to pull in
          noise. Browse the layers and you can watch heads go from mechanical
          (position-based) to semantic (meaning-based).
        </p>
      </Panel>
    </div>
  );
}
