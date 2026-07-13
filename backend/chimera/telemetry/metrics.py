"""Metrics — turn the raw StepEvent stream into the numbers people care about.

The engine emits one event per token. This aggregates them into the standard
serving metrics (the same ones vLLM/TGI report), so the UI and benchmarks read
from one source of truth instead of recomputing stats ad hoc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..engine.events import StepEvent


@dataclass
class RequestMetrics:
    request_id: int
    arrived_at: float
    first_token_at: float | None = None
    finished_at: float | None = None
    tokens: int = 0
    decode_latencies_ms: list[float] = field(default_factory=list)
    peak_cache_bytes: int = 0

    @property
    def ttft_s(self) -> float | None:
        """Time to first token — how long the user stared at a blank screen."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.arrived_at

    @property
    def tpot_ms(self) -> float:
        """Time per output token (mean decode latency) — the 'typing speed'."""
        lat = self.decode_latencies_ms
        return sum(lat) / len(lat) if lat else 0.0

    @property
    def total_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.arrived_at


class MetricsCollector:
    """Consumes StepEvents; exposes per-request and system-wide metrics."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.per_request: dict[int, RequestMetrics] = {}
        self.total_tokens = 0
        self.forward_passes = 0
        # rolling window of (timestamp, tokens) for live throughput
        self._recent: list[tuple[float, int]] = []

    def on_arrival(self, request_id: int, arrived_at: float) -> None:
        self.per_request[request_id] = RequestMetrics(request_id, arrived_at)

    def on_event(self, ev: StepEvent) -> None:
        m = self.per_request.get(ev.request_id)
        if m is None:
            m = RequestMetrics(ev.request_id, ev.timestamp)
            self.per_request[ev.request_id] = m

        if m.first_token_at is None:
            m.first_token_at = ev.timestamp
        if ev.kind == "decode":
            m.decode_latencies_ms.append(ev.latency_ms)
        m.tokens = ev.num_generated
        m.peak_cache_bytes = max(m.peak_cache_bytes, ev.cache_bytes)

        self.total_tokens += 1
        self._recent.append((ev.timestamp, 1))

    def on_finish(self, request_id: int, at: float | None = None) -> None:
        m = self.per_request.get(request_id)
        if m:
            m.finished_at = at or time.perf_counter()

    # ---- system-wide ----
    def throughput_tok_s(self, window_s: float = 2.0) -> float:
        """Tokens/sec over a recent window (what a live gauge would show)."""
        now = time.perf_counter()
        self._recent = [(t, n) for t, n in self._recent if now - t <= window_s]
        if not self._recent:
            return 0.0
        span = max(now - self._recent[0][0], 1e-6)
        return sum(n for _, n in self._recent) / span

    def summary(self) -> dict:
        done = [m for m in self.per_request.values() if m.finished_at is not None]
        ttfts = [m.ttft_s for m in self.per_request.values() if m.ttft_s is not None]
        tpots = [m.tpot_ms for m in self.per_request.values() if m.decode_latencies_ms]
        wall = time.perf_counter() - self.started_at
        return {
            "wall_s": round(wall, 2),
            "total_tokens": self.total_tokens,
            "throughput_tok_s": round(self.total_tokens / wall, 1) if wall else 0.0,
            "completed_requests": len(done),
            "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0.0,
            "avg_tpot_ms": round(sum(tpots) / len(tpots), 1) if tpots else 0.0,
            "peak_cache_mb": round(
                max((m.peak_cache_bytes for m in self.per_request.values()), default=0)
                / 1024 / 1024, 2
            ),
        }
