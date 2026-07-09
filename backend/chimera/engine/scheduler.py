"""The Scheduler — who gets a seat this step? Phase 4, Bites 3-5.

Three policies, so they can be raced against each other:

  sequential : one request at a time, start to finish (the Phase-3 baseline)
  static     : seat up to K, run the WHOLE batch to completion, then next K.
               Suffers stragglers (empty seats ride along) and the locked
               door (arrivals wait even while seats sit empty).
  continuous : rebuild the batch EVERY step — finished seats free instantly,
               waiting requests join instantly. The vLLM idea.

The scheduler owns a FIFO queue and a set of seated (DECODING) requests.
tick() = one scheduling round: admit -> one batched decode -> evict finished.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .engine import InferenceEngine
from .request import Request


@dataclass
class SchedulerStats:
    """The scoreboard (Bite 5): throughput, latency, and seat utilization."""

    started_at: float = field(default_factory=time.perf_counter)
    tokens_out: int = 0
    ticks: int = 0
    seats_used: int = 0          # sum over ticks of seated requests
    ttft: dict[int, float] = field(default_factory=dict)   # req id -> time to first token

    def snapshot(self, capacity: int, done: list[Request]) -> dict:
        wall = time.perf_counter() - self.started_at
        return {
            "wall_s": wall,
            "throughput_tok_s": self.tokens_out / wall if wall else 0.0,
            "avg_ttft_s": (sum(self.ttft.values()) / len(self.ttft)) if self.ttft else 0.0,
            "utilization": self.seats_used / (self.ticks * capacity) if self.ticks else 0.0,
            "completed": len(done),
        }


class Scheduler:
    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 8,
        policy: str = "continuous",   # "sequential" | "static" | "continuous"
    ) -> None:
        assert policy in ("sequential", "static", "continuous")
        self.engine = engine
        self.capacity = 1 if policy == "sequential" else max_batch_size
        self.policy = policy
        self.queue: deque[Request] = deque()   # WAITING requests, FIFO
        self.seated: list[Request] = []        # DECODING requests (in the batch)
        self.done: list[Request] = []
        self.stats = SchedulerStats()

    def submit(self, prompt: str, max_new_tokens: int = 20) -> Request:
        req = self.engine.submit(prompt, max_new_tokens)
        self.queue.append(req)
        return req

    @property
    def has_work(self) -> bool:
        return bool(self.queue or self.seated)

    def _admit_one(self) -> None:
        """Seat the next waiting request: run its prefill (emits first token)."""
        req = self.queue.popleft()
        ev = self.engine.step(req)               # prefill pass
        self.stats.tokens_out += 1
        self.stats.ttft[req.id] = ev.timestamp - req.arrived_at
        if req.is_finished:                       # (1-token requests exist)
            self.done.append(req)
        else:
            self.seated.append(req)

    def tick(self) -> None:
        """One scheduling round: admit -> one batched decode -> evict finished."""
        # ---- 1. admission: who may sit down? ----
        if self.policy == "continuous" or self.policy == "sequential":
            # any free seat fills IMMEDIATELY, every tick
            while self.queue and len(self.seated) < self.capacity:
                self._admit_one()
        elif self.policy == "static":
            # the locked door: only admit when the WHOLE batch has finished
            if not self.seated:
                while self.queue and len(self.seated) < self.capacity:
                    self._admit_one()

        # ---- 2. one batched decode step for everyone seated ----
        if self.seated:
            events = self.engine.step_decode_batch(self.seated)
            self.stats.tokens_out += len(events)
            self.stats.seats_used += len(self.seated)
        self.stats.ticks += 1

        # ---- 3. eviction: finished seats free up ----
        still = [r for r in self.seated if not r.is_finished]
        self.done.extend(r for r in self.seated if r.is_finished)
        self.seated = still

    def run_to_completion(self) -> dict:
        while self.has_work:
            self.tick()
        return self.stats.snapshot(self.capacity, self.done)
