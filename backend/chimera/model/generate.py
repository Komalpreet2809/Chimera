"""Generation — the autoregressive loop. Phase 0 (the loop) + Bite 14 (sampling).

We yield one token at a time (a Python generator) rather than returning the whole
string, because the per-token stream is the raw material for the live telemetry
timeline in later phases.

IMPORTANT — the crime scene (Phase 0, Bite 5): this loop feeds the ENTIRE
sequence-so-far back through the model every single step. Step N reprocesses all
N tokens from scratch, even though the first N-1 haven't changed. That wasted
quadratic recompute is exactly what KV Cache (Phase 2) will fix. We build the
honest, slow version first so the speedup is measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from .gpt import GPT


@dataclass
class SampleConfig:
    max_new_tokens: int = 40
    temperature: float = 0.8   # randomness dial: low=safe, high=adventurous
    top_k: int | None = 40     # only sample from the top-k candidates (None = all)
    greedy: bool = False       # if True, always take the argmax (ignores temp/top_k)


def _pick_next(logits: torch.Tensor, cfg: SampleConfig) -> int:
    """Turn the final-position logits into one chosen token id. (Bite 14)"""
    if cfg.greedy:
        return int(logits.argmax().item())

    logits = logits / max(cfg.temperature, 1e-6)          # temperature scaling

    if cfg.top_k is not None:                              # top-k filtering
        k = min(cfg.top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=-1)                     # -> probabilities
    return int(torch.multinomial(probs, num_samples=1).item())  # sample one


@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: list[int],
    cfg: SampleConfig | None = None,
    device: str | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time, autoregressively."""
    cfg = cfg or SampleConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ids = torch.tensor([prompt_ids], device=device)      # (1, seq)

    for _ in range(cfg.max_new_tokens):
        # THE CRIME: feed the whole sequence every step (no cache yet).
        logits, _ = model(ids)
        next_id = _pick_next(logits[0, -1], cfg)          # only last position matters
        yield next_id
        # append and loop — the sequence grows by one each step (Phase 0, Bite 2).
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
