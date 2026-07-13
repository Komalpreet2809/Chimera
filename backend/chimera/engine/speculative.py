"""Speculative decoding — break the one-token-per-big-pass barrier. Phase 6.

The bottleneck (Phase 4, Bite 1): a decode step is memory-bound. Hauling the
big model's weights out of memory costs the same whether you score 1 token or
20. We're paying full price to produce a single token.

The trick:
  1. A small, cheap DRAFT model autoregressively guesses K tokens ahead.
  2. The big TARGET model scores all K+1 positions in ONE forward pass —
     nearly free, because it was memory-bound anyway.
  3. We ACCEPT the draft's guesses that the target agrees with, and correct
     the first one it doesn't.

If the draft guesses well, we get several tokens per big-model pass instead of
one. If it guesses badly, we still always get at least one correct token — so
this can never be slower than a wasted pass, only less profitable.

CORRECTNESS: with greedy decoding, we accept a draft token only where it equals
the target's own argmax, and otherwise substitute the target's argmax. So the
output is EXACTLY what the target model would have produced alone. Speculation
changes the speed, never the answer. (The sampling case uses the modified
rejection-sampling rule from Leviathan et al. 2023, which preserves the target
distribution in expectation; we implement the greedy case here and verify it.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..model.gpt import GPT
from ..model.kv_cache import KVCache


@dataclass
class SpecStats:
    proposed: int = 0        # draft tokens guessed
    accepted: int = 0        # draft tokens the target agreed with
    target_passes: int = 0   # big-model forward passes (the expensive thing)
    tokens_out: int = 0      # tokens actually produced

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_target_pass(self) -> float:
        """The headline number. Plain decoding = 1.0. Higher is faster."""
        return self.tokens_out / self.target_passes if self.target_passes else 0.0


@torch.no_grad()
def speculative_generate(
    target: GPT,
    draft: GPT,
    prompt_ids: list[int],
    max_new_tokens: int = 40,
    lookahead: int = 4,          # K: how many tokens the draft guesses per round
    device: str = "cpu",
) -> tuple[list[int], SpecStats]:
    """Greedy speculative decoding. Returns (generated_ids, stats)."""
    stats = SpecStats()

    t_cache = KVCache(target.config)
    d_cache = KVCache(draft.config)

    # --- prime the target on the prompt; it gives us token #1 for free ---
    t_logits, _ = target(torch.tensor([prompt_ids], device=device), cache=t_cache)
    stats.target_passes += 1
    out = [int(t_logits[0, -1].argmax())]
    stats.tokens_out += 1

    def sync(cache: KVCache, model: GPT) -> list[int]:
        """Hold the invariant: a cache contains every COMMITTED token except
        the last one (which gets fed as this round's input).

        Rewinds a cache that ran ahead on rejected guesses, and catches up a
        cache that lagged behind — so we never do off-by-one arithmetic on
        accept counts.
        """
        committed = prompt_ids + out
        need = len(committed) - 1
        cache.truncate_to(need)
        if cache.seq_len < need:
            missing = committed[cache.seq_len : need]
            model(torch.tensor([missing], device=device), cache=cache)
        return committed

    while len(out) < max_new_tokens:
        # ---- 1. DRAFT: cheaply guess K tokens ahead, one at a time ----
        committed = sync(d_cache, draft)
        guesses: list[int] = []
        nxt = committed[-1]
        for _ in range(lookahead):
            d_logits, _ = draft(torch.tensor([[nxt]], device=device), cache=d_cache)
            nxt = int(d_logits[0, -1].argmax())
            guesses.append(nxt)
        stats.proposed += len(guesses)

        # ---- 2. VERIFY: target scores [last] + guesses in ONE pass ----
        # K+1 tokens in -> K+1 next-token predictions out: position i says what
        # the target would produce after seeing everything up through i.
        sync(t_cache, target)
        check = torch.tensor([[committed[-1]] + guesses], device=device)
        t_logits, _ = target(check, cache=t_cache)
        stats.target_passes += 1
        target_preds = t_logits[0].argmax(dim=-1).tolist()   # K+1 predictions

        # ---- 3. ACCEPT/REJECT: keep the prefix the target agrees with ----
        n_accepted = 0
        for i, guess in enumerate(guesses):
            if guess != target_preds[i]:
                break
            n_accepted += 1
        stats.accepted += n_accepted

        # Accepted guesses, PLUS one free correct token from the target:
        # target_preds[n_accepted] is what the target itself would say next.
        # So even a total rejection still yields one correct token.
        new_tokens = guesses[:n_accepted] + [target_preds[n_accepted]]
        new_tokens = new_tokens[: max_new_tokens - len(out)]
        out.extend(new_tokens)
        stats.tokens_out += len(new_tokens)
        if not new_tokens:
            break
        # Caches are now ahead (they hold rejected guesses) — the next round's
        # sync() rewinds them to the committed sequence.

    return out[:max_new_tokens], stats
