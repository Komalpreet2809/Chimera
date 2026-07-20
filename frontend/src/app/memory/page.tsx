"use client";

import { useCallback, useEffect, useState } from "react";
import { Button, Empty, PageHead, Panel, Stat } from "@/components/ui";
import { API_BASE, PagedBench, post } from "@/lib/api";

export default function MemoryPage() {
  const [budget, setBudget] = useState(512);
  const [maxSeq, setMaxSeq] = useState(512);
  const [blockSize, setBlockSize] = useState(16);
  const [data, setData] = useState<PagedBench | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await post<PagedBench>("/api/benchmark/paged", {
          budget_mb: budget,
          max_seq: maxSeq,
          block_size: blockSize,
        })
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [budget, maxSeq, blockSize]);

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-5">
      <PageHead kicker="03 — Paged Memory" title="Stop wasting it.">
          A serving engine can&apos;t know how long a request will run, so the naive
          approach <span className="text-[var(--bad)]">reserves the worst case</span> —
          a contiguous slab big enough for the longest possible answer. Most requests
          stop far short of it, and all that reserved memory is dead. PagedAttention
          borrows the operating system&apos;s answer: hand out small fixed{" "}
          <span className="text-[var(--good)]">blocks on demand</span>, and reclaim them
          the moment a request finishes.
      </PageHead>

      <Panel tone="amber">
        <div className="flex flex-wrap items-center gap-6">
          {[
            { label: "KV budget", value: budget, set: setBudget, min: 128, max: 2048, step: 128, unit: "MB" },
            { label: "worst-case length", value: maxSeq, set: setMaxSeq, min: 128, max: 1024, step: 128, unit: "tok" },
            { label: "block size", value: blockSize, set: setBlockSize, min: 4, max: 64, step: 4, unit: "tok" },
          ].map((s) => (
            <label key={s.label} className="flex items-center gap-2 text-xs text-[var(--muted)]">
              <span className="w-28">{s.label}</span>
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={s.value}
                onChange={(e) => s.set(+e.target.value)}
                className="w-28 accent-[var(--accent)]"
              />
              <span className="mono w-14 tabular-nums text-[var(--text)]">
                {s.value} {s.unit}
              </span>
            </label>
          ))}
          <div className="ml-auto">
            <Button onClick={run} disabled={loading}>
              {loading ? "Measuring…" : "Recompute"}
            </Button>
          </div>
        </div>
      </Panel>

      {error && (
        <div className="rounded-lg border border-[var(--bad)]/40 bg-[var(--bad)]/10 px-4 py-3 text-xs text-[var(--bad)]">
          {error} — is the backend running at {API_BASE}?
        </div>
      )}

      {!data ? (
        <Empty>Loading…</Empty>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat
              label="Users — naive"
              value={data.naive.fit}
              tone="bad"
              hint={`of ${data.workload} waiting`}
            />
            <Stat
              label="Users — paged"
              value={data.paged.fit}
              tone="good"
              hint={`${(data.paged.fit / Math.max(data.naive.fit, 1)).toFixed(1)}× more`}
            />
            <Stat
              label="Waste — naive"
              value={`${(data.naive.waste * 100).toFixed(0)}%`}
              tone="bad"
              hint="reserved but never used"
            />
            <Stat
              label="Waste — paged"
              value={`${(data.paged.waste * 100).toFixed(1)}%`}
              tone="good"
              hint="only the last partial block"
            />
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <MemoryMap
              title="Naive — reserve the worst case"
              subtitle={`each request reserves ${maxSeq} tokens whether it needs them or not`}
              fit={data.naive.fit}
              usedMb={data.naive.used_mb}
              totalMb={data.naive.reserved_mb}
              budget={data.budget_mb}
              waste={data.naive.waste}
              tone="bad"
            />
            <MemoryMap
              title="Paged — blocks on demand"
              subtitle={`${data.paged.num_blocks} blocks of ${blockSize} tokens, handed out as needed`}
              fit={data.paged.fit}
              usedMb={data.paged.used_mb}
              totalMb={data.paged.allocated_mb}
              budget={data.budget_mb}
              waste={data.paged.waste}
              tone="good"
            />
          </div>

          <Panel title="Why this works" tone="dark">
            <p className="text-xs leading-relaxed text-[var(--muted)]">
              The naive engine wastes{" "}
              <span className="mono text-[var(--bad)]">
                {(data.naive.waste * 100).toFixed(0)}%
              </span>{" "}
              of its memory on space it reserved but never touched — so only{" "}
              <span className="mono text-[var(--text)]">{data.naive.fit}</span> users fit
              in {data.budget_mb} MB. Paging cuts that waste to{" "}
              <span className="mono text-[var(--good)]">
                {(data.paged.waste * 100).toFixed(1)}%
              </span>{" "}
              (just the unfilled tail of each request&apos;s last block), fitting{" "}
              <span className="mono text-[var(--good)]">{data.paged.fit}</span> users in
              the exact same memory. Nothing about the model changed — only how its
              memory is handed out. Shrink the block size and waste falls further, at the
              cost of a longer block table per request.
            </p>
          </Panel>
        </>
      )}
    </div>
  );
}

function MemoryMap({
  title,
  subtitle,
  fit,
  usedMb,
  totalMb,
  budget,
  waste,
  tone,
}: {
  title: string;
  subtitle: string;
  fit: number;
  usedMb: number;
  totalMb: number;
  budget: number;
  waste: number;
  tone: "good" | "bad";
}) {
  const CELLS = 240;
  const usedCells = Math.round((usedMb / budget) * CELLS);
  const wastedCells = Math.round(((totalMb - usedMb) / budget) * CELLS);
  // Chart marks carry no text, so they take the brighter fill steps.
  const color = tone === "good" ? "var(--good-fill)" : "var(--accent-fill)";

  return (
    <Panel title={title} subtitle={subtitle}>
      <div className="grid grid-cols-[repeat(24,1fr)] gap-[2px]">
        {Array.from({ length: CELLS }).map((_, i) => {
          const kind =
            i < usedCells ? "used" : i < usedCells + wastedCells ? "wasted" : "free";
          return (
            <div
              key={i}
              className="aspect-square rounded-[2px]"
              style={{
                background:
                  kind === "used"
                    ? color
                    : kind === "wasted"
                      ? "var(--dead)"
                      : "var(--panel-2)",
                // Full strength throughout: any fade of rust over cream
                // composites toward blush. The inks are already chosen to sit
                // together, so they don't need turning down.
                opacity: 1,
              }}
            />
          );
        })}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3 text-[10px] text-[var(--dim)]">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm" style={{ background: color }} /> real KV
          data ({usedMb.toFixed(0)} MB)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-[var(--dead)]" /> wasted (
          {(totalMb - usedMb).toFixed(0)} MB)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-[var(--panel-2)]" /> free
        </span>
        <span className="mono ml-auto font-semibold" style={{ color }}>
          {fit} users · {(waste * 100).toFixed(1)}% wasted
        </span>
      </div>
    </Panel>
  );
}
