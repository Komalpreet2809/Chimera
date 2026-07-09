"""InferenceEngine — the chassis. Phase 3, Bites 1-3.

Owns the model/tokenizer/device, holds a table of live Requests, and exposes
ONE core operation:

    step(request) -> StepEvent      "advance this request by exactly one step"

A full generation is just step() called repeatedly — which means steps from
different requests can be INTERLEAVED. That interleaving is what the Phase 4
scheduler will orchestrate; here we make it possible.
"""

from __future__ import annotations

import time

import torch

from ..model.generate import SampleConfig, _pick_next
from ..model.gpt import GPT
from ..model.tokenizer import Tokenizer
from .events import StepEvent
from .request import FinishReason, Request, RequestState

GPT2_EOS = 50256  # GPT-2's end-of-text token id


class InferenceEngine:
    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        device: str | None = None,
        sample_cfg: SampleConfig | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.sample_cfg = sample_cfg or SampleConfig()
        self.requests: dict[int, Request] = {}   # the table of live requests

    # ---- intake ----
    def submit(self, prompt: str, max_new_tokens: int = 40) -> Request:
        """Wrap a prompt in a Request and register it. State: WAITING."""
        req = Request(
            prompt_ids=self.tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
        )
        self.requests[req.id] = req
        return req

    # ---- the core operation ----
    @torch.no_grad()
    def step(self, req: Request) -> StepEvent:
        """Advance one request by exactly one unit of progress.

        WAITING    -> run PREFILL (one fat pass over the prompt) -> first token
        DECODING   -> run one decode step (feed only the newest token)
        """
        if req.is_finished:
            raise ValueError(f"request {req.id} is already finished")

        if req.state == RequestState.WAITING:
            # ---- PREFILL: fill the cache with the whole prompt (Phase 2, Bite 5)
            req.state = RequestState.PREFILLING
            req.init_cache(self.model.config)
            input_ids = torch.tensor([req.prompt_ids], device=self.device)
            kind = "prefill"
        else:
            # ---- DECODE: feed only the newest token (Phase 2, Bite 3)
            input_ids = torch.tensor([[req.generated_ids[-1]]], device=self.device)
            kind = "decode"

        t0 = time.perf_counter()
        logits, _ = self.model(input_ids, cache=req.kv_cache)
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        next_id = _pick_next(logits[0, -1], self.sample_cfg)
        req.generated_ids.append(next_id)
        req.state = RequestState.DECODING

        # Snapshot cache stats BEFORE a potential finish frees the cache.
        cache_tokens = req.kv_cache.seq_len
        cache_bytes = req.kv_cache.memory_bytes()

        # ---- finish checks (Bite 2: length / EOS)
        if next_id == GPT2_EOS:
            req.finish(FinishReason.EOS)
        elif req.num_generated >= req.max_new_tokens:
            req.finish(FinishReason.LENGTH)

        return StepEvent(
            request_id=req.id,
            kind=kind,
            token_id=next_id,
            token_text=self.tokenizer.decode([next_id]),
            latency_ms=latency_ms,
            cache_tokens=cache_tokens,
            cache_bytes=cache_bytes,
            num_generated=req.num_generated,
        )

    # ---- the batched operation (Phase 4) ----
    @torch.no_grad()
    def step_decode_batch(self, reqs: list[Request]) -> list[StepEvent]:
        """One decode step for MANY requests in a single forward pass.

        All requests must already be DECODING (prefill runs per-request via
        step()). The weights get hauled from memory once and applied to every
        row — this is the whole point of batching (Bite 1).
        """
        from .batch import BatchedKVCache

        assert all(r.state == RequestState.DECODING for r in reqs)

        # Stack each request's newest token: (B, 1).
        input_ids = torch.tensor(
            [[r.generated_ids[-1]] for r in reqs], device=self.device
        )
        bcache = BatchedKVCache([r.kv_cache for r in reqs])

        t0 = time.perf_counter()
        logits, _ = self.model(
            input_ids,
            cache=bcache,
            pos_offset=bcache.pos_offsets(self.device),
            key_padding_mask=bcache.padding_mask(self.device),
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        events = []
        for i, req in enumerate(reqs):
            next_id = _pick_next(logits[i, -1], self.sample_cfg)
            req.generated_ids.append(next_id)

            cache_tokens = req.kv_cache.seq_len
            cache_bytes = req.kv_cache.memory_bytes()
            if next_id == GPT2_EOS:
                req.finish(FinishReason.EOS)
            elif req.num_generated >= req.max_new_tokens:
                req.finish(FinishReason.LENGTH)

            events.append(
                StepEvent(
                    request_id=req.id,
                    kind="decode",
                    token_id=next_id,
                    token_text=self.tokenizer.decode([next_id]),
                    latency_ms=latency_ms,  # shared: one pass served all rows
                    cache_tokens=cache_tokens,
                    cache_bytes=cache_bytes,
                    num_generated=req.num_generated,
                )
            )
        return events

    # ---- convenience ----
    def text_of(self, req: Request) -> str:
        """The full text (prompt + generation) of a request so far."""
        return self.tokenizer.decode(req.prompt_ids + req.generated_ids)
