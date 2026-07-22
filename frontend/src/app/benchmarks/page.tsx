"use client";

import { useCallback, useState } from "react";
import { Badge, Button, Empty, PageHead, Panel, Stat } from "@/components/ui";
import { API_BASE, CacheBench, post } from "@/lib/api";

export default function BenchmarksPage() {
  const [data, setData] = useState<CacheBench | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await post<CacheBench>("/api/benchmark/cache", {
          seq_lens: [64, 128, 256, 512, 900],
          device: "cpu", // see the note below — the GPU can't show this effect
          iters: 3,
        })
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const maxMs = data ? Math.max(...data.rows.map((r) => r.naive_ms)) : 1;
  const best = data ? Math.max(...data.rows.map((r) => r.speedup)) : 0;

  return (
    <div className="space-y-5">
      <PageHead kicker="05 — Benchmarks" title="Measured, not claimed.">
          The KV cache table below runs live on this machine when you press the button —
          real forward passes, timed now. The two panels beneath it are{" "}
          <span className="text-[var(--text)]">recorded</span> results from the repo&apos;s
          benchmark scripts, not live measurements, because each takes minutes to run.
          Every one of them is reproducible from the commands shown.
      </PageHead>

      <Panel
        title="KV Cache — per-step decode latency"
        subtitle="how long one token costs when the sequence is already N tokens long"
        right={
          <Button onClick={run} disabled={loading}>
            {loading ? "Measuring…" : data ? "Re-run" : "Run benchmark"}
          </Button>
        }
      >
        {error && (
          <div className="mb-3 rounded-lg border border-[var(--bad)]/40 bg-[var(--bad)]/10 px-3 py-2 text-xs text-[var(--bad)]">
            {error} — is the backend running at {API_BASE}?
          </div>
        )}
        {!data ? (
          <Empty>
            {loading
              ? "Running real forward passes…"
              : "Press Run to benchmark the engine live."}
          </Empty>
        ) : (
          <div className="space-y-5">
            <div className="grid grid-cols-3 gap-3">
              <Stat label="Device" value={data.device.toUpperCase()} tone="accent" />
              <Stat
                label="Best speedup"
                value={`${best.toFixed(1)}×`}
                tone="good"
                hint="at the longest sequence"
              />
              <Stat
                label="Scaling"
                value="O(N) → O(1)"
                hint="per decode step"
              />
            </div>

            <div className="space-y-2.5">
              {data.rows.map((r) => (
                <div key={r.seq_len} className="space-y-1">
                  <div className="mono flex items-center justify-between text-[11px]">
                    <span className="text-[var(--muted)]">
                      seq&nbsp;=&nbsp;
                      <span className="text-[var(--text)]">{r.seq_len}</span> tokens
                    </span>
                    <span className="font-semibold text-[var(--good)]">
                      {r.speedup.toFixed(1)}× faster
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="mono w-12 shrink-0 text-right text-[10px] text-[var(--bad)]">
                      naive
                    </span>
                    <div className="h-4 flex-1 overflow-hidden rounded bg-[var(--panel-2)]">
                      <div
                        className="flex h-full items-center justify-end rounded bg-[var(--dead)] pr-1.5"
                        style={{ width: `${(r.naive_ms / maxMs) * 100}%` }}
                      >
                        <span className="mono text-[9px] font-semibold text-white">
                          {r.naive_ms.toFixed(0)}ms
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="mono w-12 shrink-0 text-right text-[10px] text-[var(--accent)]">
                      cached
                    </span>
                    <div className="h-4 flex-1 overflow-hidden rounded bg-[var(--panel-2)]">
                      <div
                        className="flex h-full items-center justify-end rounded bg-[var(--accent)] pr-1.5"
                        style={{
                          width: `${Math.max((r.cached_ms / maxMs) * 100, 4)}%`,
                        }}
                      >
                        <span className="mono text-[9px] font-semibold text-[var(--line)]">
                          {r.cached_ms.toFixed(0)}ms
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <p className="text-[11px] leading-relaxed text-[var(--dim)]">
              Without a cache, every step re-processes the entire sequence, so the cost
              climbs with length — and generating N tokens costs O(N²) work overall. With
              a KV cache, each step only processes the one new token, so the cost stays
              roughly flat no matter how long the conversation gets. Same output, every
              token identical — just without redoing work we already did.
            </p>

            <div className="rounded-lg border border-[var(--warn)]/30 bg-[var(--warn)]/5 p-3">
              <p className="text-[11px] font-semibold text-[var(--warn)]">
                Why this benchmark runs on CPU, not the GPU
              </p>
              <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--dim)]">
                The KV cache saves <span className="text-[var(--muted)]">compute</span> — so
                you can only see it on hardware where compute is the bottleneck. On this
                laptop GPU it isn&apos;t: GPT-2&apos;s ops are so small that the card never
                leaves its idle power state (measured{" "}
                <span className="mono">210&nbsp;MHz of a 3105&nbsp;MHz clock, 1.65&nbsp;W</span>
                ), so every decode step costs a flat ~80&nbsp;ms of kernel-launch overhead at
                a throttled clock — no matter how long the sequence is. The algorithmic win
                is real, but that overhead floor hides it completely. On CPU the work is
                compute-bound and the O(N)→O(1) difference appears exactly as predicted.
                Both are honest measurements; only one is measuring the thing we asked about.
              </p>
            </div>
          </div>
        )}
      </Panel>

      <div className="grid gap-5 md:grid-cols-2">
        <Panel
          title="Continuous batching"
          subtitle="recorded · 20 users, 8 seats"
          right={<Badge tone="default">recorded</Badge>}
        >
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-left text-[var(--dim)]">
                <th className="pb-2 font-medium">policy</th>
                <th className="pb-2 text-right font-medium">tok/s</th>
                <th className="pb-2 text-right font-medium">TTFT</th>
                <th className="pb-2 text-right font-medium">seats used</th>
              </tr>
            </thead>
            <tbody className="mono">
              {[
                { p: "sequential", t: "9.8", f: "25.8s", u: "100%", tone: "var(--bad)" },
                { p: "static", t: "26.3", f: "6.5s", u: "43%", tone: "var(--warn)" },
                { p: "continuous", t: "36.2", f: "3.4s", u: "80%", tone: "var(--good)" },
              ].map((r) => (
                <tr key={r.p} className="border-t border-[var(--line)]">
                  <td className="py-2" style={{ color: r.tone }}>
                    {r.p}
                  </td>
                  <td className="py-2 text-right tabular-nums">{r.t}</td>
                  <td className="py-2 text-right tabular-nums">{r.f}</td>
                  <td className="py-2 text-right tabular-nums">{r.u}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-[10px] leading-snug text-[var(--dim)]">
            Static batching keeps 100% of its seats <em>allocated</em> but only 43%{" "}
            <em>used</em> — stragglers hold empty seats hostage. Continuous batching
            refills them instantly.
          </p>
          <p className="mono mt-2 text-[10px] text-[var(--dim)]">
            reproduce: python backend/scripts/phase4_showdown.py
          </p>
        </Panel>

        <Panel
          title="Speculative decoding"
          subtitle="recorded · distilgpt2 drafts, gpt2 verifies"
          right={<Badge tone="default">recorded</Badge>}
        >
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-left text-[var(--dim)]">
                <th className="pb-2 font-medium">metric</th>
                <th className="pb-2 text-right font-medium">value</th>
              </tr>
            </thead>
            <tbody className="mono">
              {[
                ["output identical to GPT-2", "yes"],
                ["draft acceptance rate", "65–86%"],
                ["tokens per target pass", "3.4–4.0"],
                ["big-model passes saved", "3.4–4.0×"],
                ["wall-clock speedup", "1.0–1.3×"],
              ].map(([k, v]) => (
                <tr key={k} className="border-t border-[var(--line)]">
                  <td className="py-2 text-[var(--muted)]">{k}</td>
                  <td className="py-2 text-right text-[var(--text)]">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-[10px] leading-snug text-[var(--dim)]">
            Honest result: we cut big-model passes by ~4×, but wall-clock barely moves —
            distilgpt2 is only 2× cheaper than gpt2, so the draft&apos;s own passes eat the
            savings. Speculation pays off when the target dwarfs the draft (a 70B verified
            by a 1B), not at GPT-2 scale. Correct implementation, honest verdict.
          </p>
          <p className="mono mt-2 text-[10px] text-[var(--dim)]">
            reproduce: python backend/scripts/phase6_speculative.py
          </p>
        </Panel>
      </div>
    </div>
  );
}
