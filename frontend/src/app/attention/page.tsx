"use client";

import { useCallback, useEffect, useState } from "react";
import { Button, Empty, PageHead, Panel } from "@/components/ui";
import { API_BASE, AttentionResp, post } from "@/lib/api";

/**
 * Sequential ramp for the heatmap, built from the brand ramps.
 *
 * A single hue varied by opacity only travels cream -> amber, so every cell
 * lands in the same pale band and the map reads flat. Real heat ramps move
 * through lightness AND hue, so this walks paper -> amber -> rust: light and
 * warm at the bottom, dark and saturated at the top. Same palette, far more
 * dynamic range.
 */
const HEAT: [number, [number, number, number]][] = [
  [0.0, [250, 244, 233]], // paper
  [0.12, [246, 231, 203]], // amber wash
  [0.35, [242, 200, 119]], // amber-6
  [0.6, [224, 160, 50]], // amber-5
  [0.78, [196, 131, 26]], // amber-4
  [0.92, [140, 58, 36]], // rust-2
  [1.0, [109, 46, 28]], // rust-1
];

function heat(v: number): string {
  const t = Math.min(1, Math.max(0, v));
  for (let i = 1; i < HEAT.length; i++) {
    const [p1, c1] = HEAT[i];
    if (t <= p1) {
      const [p0, c0] = HEAT[i - 1];
      const k = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
      const m = c0.map((c, j) => Math.round(c + (c1[j] - c) * k));
      return `rgb(${m[0]},${m[1]},${m[2]})`;
    }
  }
  const last = HEAT[HEAT.length - 1][1];
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

/** Relative luminance, to decide whether a cell needs light or dark type. */
function isDark(v: number): boolean {
  return v > 0.55;
}

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
          {error} — is the backend running at {API_BASE}?
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
                      className={`mono grid h-[30px] w-[34px] shrink-0 place-items-center rounded-[2px] text-[9px] font-medium transition-transform hover:z-10 hover:scale-125 hover:shadow-[0_2px_10px_rgba(26,26,24,0.25)] ${
                        c > r ? "opacity-45" : ""
                      }`}
                      style={{
                        // Perceptual boost: attention is heavily skewed toward
                        // a few large weights, so a linear map would leave most
                        // cells indistinguishable. ^0.45 opens up the low end.
                        background:
                          c > r ? "transparent" : heat(Math.pow(v, 0.45)),
                        color:
                          c > r
                            ? "transparent"
                            : isDark(Math.pow(v, 0.45))
                              ? "rgba(255,247,235,0.95)"
                              : "rgba(42,28,5,0.75)",
                        outline:
                          c > r ? "1px dashed var(--line-soft)" : "none",
                        outlineOffset: "-3px",
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
        <div className="mt-5 flex flex-wrap items-center gap-5 text-[10px] text-[var(--dim)]">
          {/* the actual scale, not a single swatch standing in for it */}
          <span className="flex items-center gap-2">
            <span>0%</span>
            <span
              className="h-2.5 w-32 rounded-full"
              style={{
                background: `linear-gradient(to right, ${[0, 0.2, 0.4, 0.6, 0.8, 1]
                  .map((s) => heat(s))
                  .join(",")})`,
              }}
            />
            <span>100% attention</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="h-2.5 w-2.5 rounded-[2px]"
              style={{ outline: "1px dashed var(--line-soft)", outlineOffset: "-1px" }}
            />{" "}
            masked — the future, which a token can never look at
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
