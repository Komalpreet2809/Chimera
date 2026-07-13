"""Request — every prompt gets a file folder. Phase 3, Bite 2.

One user prompt = one Request: the prompt, progress so far, its OWN KV cache
(conversations can't share memory), and a lifecycle state.

    WAITING -> PREFILLING -> DECODING -> FINISHED
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum

from ..model.config import GPTConfig
from ..model.kv_cache import KVCache


class RequestState(Enum):
    WAITING = "waiting"        # arrived, nothing done yet
    PREFILLING = "prefilling"  # about to run the one fat pass over the prompt
    DECODING = "decoding"      # one token per step, cache grows +1 each
    FINISHED = "finished"      # done — cache memory can be reclaimed


class FinishReason(Enum):
    LENGTH = "length"          # hit max_new_tokens
    EOS = "eos"                # model produced the end-of-text token
    CANCELLED = "cancelled"


_ids = itertools.count()


@dataclass
class Request:
    prompt_ids: list[int]
    max_new_tokens: int = 40

    # -- bookkeeping (filled in by the engine as life happens) --
    id: int = field(default_factory=lambda: next(_ids))
    state: RequestState = RequestState.WAITING
    generated_ids: list[int] = field(default_factory=list)
    kv_cache: KVCache | None = None            # created at prefill time
    finish_reason: FinishReason | None = None
    arrived_at: float = field(default_factory=time.perf_counter)

    @property
    def num_generated(self) -> int:
        return len(self.generated_ids)

    @property
    def is_finished(self) -> bool:
        return self.state == RequestState.FINISHED

    def init_cache(self, config: GPTConfig, pool=None) -> None:
        """Contiguous KVCache by default; a paged cache if a block pool is given."""
        self.kv_cache = pool.new_cache() if pool is not None else KVCache(config)

    def finish(self, reason: FinishReason) -> None:
        self.state = RequestState.FINISHED
        self.finish_reason = reason
        # Reclaim memory. A paged cache hands its blocks back to the pool, where
        # they're instantly reusable by a queued request (Phase 5).
        if self.kv_cache is not None and hasattr(self.kv_cache, "free"):
            self.kv_cache.free()
        self.kv_cache = None
