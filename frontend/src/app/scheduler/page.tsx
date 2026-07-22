"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Empty, PageHead, Panel, Stat } from "@/components/ui";
import { API_BASE, SimDone, SimMsg, stream, TickMsg } from "@/lib/api";
import { useRuntime } from "@/components/runtime";

const PROMPTS = [
  "The history of computing began",
  "A recipe for perfect pasta:",
  "The best way to learn a language is",
  "In space, no one can",
  "Once upon a time in a village",
  "The theory of relativity says",
  "My favorite programming language is",
  "The ocean is home to",
  "Climate change will require",
  "To build a house you first",
  "Music has the power to",
  "The human brain contains",
];
// deliberately unequal — this is what creates stragglers
const LENGTHS = [5, 40, 12, 30, 8, 36, 16, 10, 28, 14, 6, 22];

const POLICIES = [
  { id: "sequential", label: "Sequential", desc: "one request at a time" },
  { id: "static", label: "Static batching", desc: "batch runs to completion" },
  { id: "continuous", label: "Continuous batching", desc: "batch rebuilt every step" },
] as const;

type Policy = (typeof POLICIES)[number]["id"];

export default function SchedulerPage() {
  const runtime = useRuntime();
  const [policy, setPolicy] = useState<Policy>("continuous");
  const [batchSize, setBatchSize] = useState(4);
  const [running, setRunning] = useState(false);
  const [frames, setFrames] = useState<TickMsg[]>([]);
  const [playhead, setPlayhead] = useState(-1);
  const [replaying, setReplaying] = useState(false);
  const [result, setResult] = useState<SimDone["stats"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeRef = useRef<(() => void) | null>(null);

  const run = useCallback(() => {
    setRunning(true);
    setFrames([]);
    setPlayhead(-1);
    setReplaying(false);
    setResult(null);
    setError(null);

    closeRef.current = stream<SimMsg>(
      "/api/simulate",
      {
        prompts: PROMPTS,
        lengths: LENGTHS,
        policy,
        max_batch_size: batchSize,
        paged: true,
      },
      (m) => {
        if (m.type === "tick") {
          setFrames((current) => [...current, m]);
        } else if (m.type === "done") {
          setResult(m.stats);
          setRunning(false);
          setPlayhead(0);
          setReplaying(true);
        } else if (m.type === "error") {
          setError(m.message);
          setRunning(false);
        }
      },
      () => setRunning(false)
    );
  }, [policy, batchSize]);

  useEffect(() => {
    if (!replaying || playhead < 0) return;
    const atEnd = playhead >= frames.length - 1;
    const timer = window.setTimeout(() => {
      if (atEnd) setReplaying(false);
      else setPlayhead((current) => current + 1);
    }, atEnd ? 0 : 110);
    return () => window.clearTimeout(timer);
  }, [frames, playhead, replaying]);

  const replay = () => {
    if (!frames.length) return;
    setPlayhead(0);
    setReplaying(true);
  };

  const tick = playhead >= 0 ? frames[playhead] ?? null : null;
  const history = playhead >= 0 ? frames.slice(0, playhead + 1) : [];

  const capacity = tick?.capacity ?? batchSize;
  const seats = Array.from({ length: capacity });
  const avgUtil = history.length
    ? history.reduce((s, t) => s + t.utilization, 0) / history.length
    : 0;
  const selectedToken = runtime.tokens[runtime.selected];

  return (
    <div className="space-y-5">
      <PageHead kicker="02 — Scheduler" title="Fill every seat.">
          12 users arrive at once wanting answers of very different lengths. Watch how
          each policy fills the GPU&apos;s batch slots. With{" "}
          <span className="text-[var(--text)]">static batching</span>, short requests
          finish early and their seats sit <span className="text-[var(--bad)]">empty</span>{" "}
          while one straggler runs on. With{" "}
          <span className="text-[var(--text)]">continuous batching</span>, a freed seat is
          refilled from the queue on the very next step.
      </PageHead>

      <Panel tone="amber">
        <div className="flex flex-wrap items-end gap-5">
          <div className="flex gap-1.5">
            {POLICIES.map((p) => (
              <button
                key={p.id}
                disabled={running}
                onClick={() => setPolicy(p.id)}
                className={`rounded-md border px-3 py-2 text-left transition-colors disabled:opacity-50 ${
                  policy === p.id
                    ? "border-[var(--accent)] bg-[var(--accent)]/10"
                    : "border-[var(--line)] hover:border-[var(--dim)]"
                }`}
              >
                <div
                  className={`text-xs font-semibold ${
                    policy === p.id ? "text-[var(--accent)]" : "text-[var(--text)]"
                  }`}
                >
                  {p.label}
                </div>
                <div className="text-[10px] text-[var(--dim)]">{p.desc}</div>
              </button>
            ))}
          </div>
          <label className="flex items-center gap-2 text-xs text-[var(--muted)]">
            <span>batch seats</span>
            <input
              type="range"
              min={1}
              max={8}
              value={batchSize}
              disabled={running || policy === "sequential"}
              onChange={(e) => setBatchSize(+e.target.value)}
              className="w-24 accent-[var(--accent)]"
            />
            <span className="mono w-4 tabular-nums text-[var(--text)]">
              {policy === "sequential" ? 1 : batchSize}
            </span>
          </label>
          <div className="ml-auto">
            <Button onClick={frames.length && !running ? replay : run} disabled={running || replaying}>
              {running ? "Recording trace…" : replaying ? "Playing traffic…" : frames.length ? "Replay movie" : "Run simulation"}
            </Button>
          </div>
        </div>
      </Panel>

      {error && (
        <div className="rounded-lg border border-[var(--bad)]/40 bg-[var(--bad)]/10 px-4 py-3 text-xs text-[var(--bad)]">
          {error} — is the backend running at {API_BASE}?
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Tick" value={tick?.tick ?? "—"} hint="one batched forward pass" />
        <Stat
          label="Seat utilization"
          value={history.length ? `${(avgUtil * 100).toFixed(0)}%` : "—"}
          tone={avgUtil > 0.75 ? "good" : avgUtil > 0.5 ? "warn" : "bad"}
          hint="filled slots / available slots"
        />
        <Stat
          label="Throughput"
          value={result ? result.throughput_tok_s.toFixed(1) : "—"}
          unit="tok/s"
          tone="accent"
        />
        <Stat
          label="Avg TTFT"
          value={result ? result.avg_ttft_s.toFixed(2) : "—"}
          unit="s"
          tone="violet"
          hint="time staring at a blank screen"
        />
      </div>

      {selectedToken ? (
        // The debugger's trace and this simulation are separate runs: that
        // token came from a single /api/generate request, which never entered
        // the scheduler. Claiming it occupied a seat here would invent a link
        // that doesn't exist — so state the actual relationship instead.
        <div className="linked-seat">
          <span className="section-label">Your debugger trace</span>
          <strong className="mono">Token #{selectedToken.index} {JSON.stringify(selectedToken.token)}</strong>
          <span className="mono">{selectedToken.kind} · {selectedToken.latency_ms.toFixed(1)} ms</span>
          <em>ran as a single request — it never queued for a seat. The simulation below is what changes when 12 arrive at once.</em>
        </div>
      ) : null}

      <Panel title="Request traffic" subtitle="each lane is a request; the playhead is the GPU's current scheduling step">
        {!tick ? (
          <Empty>Run once to record the scheduler, then watch requests move through the GPU.</Empty>
        ) : (
          <TrafficLanes history={history} current={tick} />
        )}
      </Panel>

      <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
        <Panel
          title="Batch Seats"
          subtitle="each slot is a request riding this forward pass"
        >
          {!tick ? (
            <Empty>Run the simulation to watch the batch fill.</Empty>
          ) : (
            <div className="space-y-3">
              <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(capacity, 4)}, minmax(0,1fr))` }}>
                {seats.map((_, i) => {
                  const seat = tick.seated[i];
                  const justAdmitted = seat && tick.admitted.includes(seat.id);
                  return (
                    <div
                      key={i}
                      className={`rounded-lg border p-3 transition-all ${
                        seat
                          ? justAdmitted
                            ? "border-[var(--good)] bg-[var(--good)]/10"
                            : "border-[var(--accent)]/40 bg-[var(--accent)]/5"
                          : "border-dashed border-[var(--line)] bg-transparent"
                      }`}
                    >
                      {seat ? (
                        <>
                          <div className="mono flex items-center justify-between text-[11px]">
                            <span className="font-semibold text-[var(--accent)]">
                              req #{seat.id}
                            </span>
                            <span className="text-[var(--dim)]">
                              {seat.tokens}/{seat.target}
                            </span>
                          </div>
                          <div className="mt-2 h-1 overflow-hidden rounded-full bg-[var(--panel-2)]">
                            <div
                              className="h-full rounded-full bg-[var(--accent)] transition-all"
                              style={{
                                width: `${(seat.tokens / seat.target) * 100}%`,
                              }}
                            />
                          </div>
                        </>
                      ) : result ? (
                        // the run is over — an empty seat here isn't waste, it's just done
                        <div className="mono py-1.5 text-center text-[11px] text-[var(--dim)]">
                          idle
                        </div>
                      ) : (
                        <div className="mono py-1.5 text-center text-[11px] text-[var(--bad)]/60">
                          empty — wasted
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="flex items-center gap-4 border-t border-[var(--line)] pt-3 text-[11px]">
                <span className="text-[var(--muted)]">
                  queued{" "}
                  <span className="mono font-semibold text-[var(--warn)]">
                    {tick.queued}
                  </span>
                </span>
                <span className="text-[var(--muted)]">
                  completed{" "}
                  <span className="mono font-semibold text-[var(--good)]">
                    {tick.completed}
                  </span>
                </span>
                {tick.pool && (
                  <span className="ml-auto text-[var(--muted)]">
                    KV blocks{" "}
                    <span className="mono font-semibold text-[var(--violet)]">
                      {tick.pool.blocks_used}
                    </span>
                    <span className="text-[var(--dim)]">/{tick.pool.num_blocks}</span>
                  </span>
                )}
              </div>
            </div>
          )}
        </Panel>

        <Panel title="Utilization over time" subtitle="how full the batch stayed">
          {history.length === 0 ? (
            <Empty>—</Empty>
          ) : (
            <>
              <div className="flex h-[120px] items-end gap-[2px]">
                {history.map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t"
                    style={{
                      height: `${Math.max(2, h.utilization * 110)}px`,
                      // Two inks, three tiers: a full batch is solid amber, a
                      // half-full one is the same amber faded, and a starved
                      // batch turns rust. Fading rather than introducing a
                      // third hue keeps the page to charcoal + amber + rust.
                      background:
                        h.utilization > 0.5 ? "var(--good-fill)" : "var(--dead)",
                      opacity: h.utilization > 0.75 ? 0.9 : h.utilization > 0.5 ? 0.5 : 0.85,
                    }}
                    title={`tick ${h.tick}: ${(h.utilization * 100).toFixed(0)}%`}
                  />
                ))}
              </div>
              <p className="mt-3 text-[10px] leading-snug text-[var(--dim)]">
                Grey bars are wasted GPU capacity — seats sitting empty because a policy
                won&apos;t let a queued request take them.
              </p>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}

function TrafficLanes({ history, current }: { history: TickMsg[]; current: TickMsg }) {
  const requests = new Map<number, { id: number; tokens: number; target: number; lastTick: number }>();
  for (const frame of history) {
    for (const seat of frame.seated) requests.set(seat.id, { ...seat, lastTick: frame.tick });
  }
  const active = new Set(current.seated.map((seat) => seat.id));
  return (
    <div className="traffic-movie">
      <div className="traffic-gpu">
        <span>GPU · TICK {current.tick}</span>
        <div>{Array.from({ length: current.capacity }).map((_, index) => {
          const seat = current.seated[index];
          return <i key={index} className={seat ? "occupied" : ""}>{seat ? `#${seat.id}` : "—"}</i>;
        })}</div>
      </div>
      <div className="traffic-lanes">
        {[...requests.values()].map((request) => {
          const isActive = active.has(request.id);
          const done = !isActive && request.lastTick < current.tick;
          return (
            <div key={request.id} className={isActive ? "is-active" : done ? "is-done" : ""}>
              <span className="mono">REQ #{request.id}</span>
              <div><i style={{ width: `${Math.min(100, request.tokens / request.target * 100)}%` }} /></div>
              <b className="mono">{done ? "done" : isActive ? "on GPU" : "queued"}</b>
            </div>
          );
        })}
      </div>
      <p className="traffic-caption">A seat empties, then the next queued request takes it on the following tick. Replay another policy to feel the difference.</p>
    </div>
  );
}
