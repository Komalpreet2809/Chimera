"""StepEvent — the flight recorder. Phase 3, Bite 3 (Idea 3).

Every engine step emits one of these: what happened, to whom, how long it
took, and how much memory the request's cache now holds. This per-token
event stream is the raw material for telemetry (Phase 7) and the live UI
timeline (Phase 8) — we build the emission point now so those just plug in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class StepEvent:
    request_id: int
    kind: str                  # "prefill" | "decode" | "finish"
    token_id: int | None       # the token produced this step (None on finish)
    token_text: str | None     # decoded piece, for humans/UI
    latency_ms: float          # how long this step's forward pass took
    cache_tokens: int          # tokens held in this request's KV cache
    cache_bytes: int           # cache memory footprint right now (Bite 4 live)
    num_generated: int         # total tokens produced for this request so far
    probability: float = 0.0   # probability assigned to the sampled token
    timestamp: float = field(default_factory=time.perf_counter)
