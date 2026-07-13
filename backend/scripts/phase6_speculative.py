"""Phase 6: speculative decoding — distilgpt2 drafts, gpt2 verifies.

Two things to prove:
  1. CORRECTNESS: the output is EXACTLY what GPT-2 would have produced alone.
     Speculation buys speed, never a different answer.
  2. SPEEDUP: tokens produced per big-model forward pass. Plain decoding is
     stuck at 1.0. Speculation gets several.

Run:  python backend/scripts/phase6_speculative.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.engine.speculative import speculative_generate
from chimera.model.config import GPTConfig
from chimera.model.kv_cache import KVCache
from chimera.model.tokenizer import Tokenizer
from chimera.model.weights import load_pretrained

# distilgpt2: same architecture and tokenizer as gpt2, but 6 layers instead of 12.
DISTIL = GPTConfig(n_layer=6)

PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network is",
    "The best way to learn programming is to",
]


@torch.no_grad()
def baseline(model, prompt_ids, n, device):
    """Plain KV-cached greedy decoding: one big-model pass per token."""
    cache = KVCache(model.config)
    logits, _ = model(torch.tensor([prompt_ids], device=device), cache=cache)
    nxt = int(logits[0, -1].argmax())
    out = [nxt]
    passes = 1
    while len(out) < n:
        logits, _ = model(torch.tensor([[nxt]], device=device), cache=cache)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        passes += 1
    return out, passes


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer("gpt2")
    print("loading target (gpt2, 12 layers) and draft (distilgpt2, 6 layers)...")
    target = load_pretrained("gpt2").to(device)
    draft = load_pretrained("distilgpt2", DISTIL).to(device)

    # warm up
    with torch.no_grad():
        target(torch.zeros(1, 8, dtype=torch.long, device=device))
        draft(torch.zeros(1, 8, dtype=torch.long, device=device))
    if device == "cuda":
        torch.cuda.synchronize()

    N = 48
    print(f"\ngenerating {N} tokens per prompt, lookahead K=4\n")
    print(f"{'prompt':<34} {'identical':>10} {'accept':>7} {'tok/pass':>9} {'passes':>8} {'wall':>7}")

    for p in PROMPTS:
        ids = tok.encode(p)

        t0 = time.perf_counter()
        base_out, base_passes = baseline(target, ids, N, device)
        if device == "cuda":
            torch.cuda.synchronize()
        base_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        spec_out, st = speculative_generate(
            target, draft, ids, max_new_tokens=N, lookahead=4, device=device
        )
        if device == "cuda":
            torch.cuda.synchronize()
        spec_s = time.perf_counter() - t0

        print(
            f"{p[:32]:<34} {str(base_out == spec_out):>10}"
            f" {st.acceptance_rate:>6.0%} {st.tokens_per_target_pass:>9.2f}"
            f" {base_passes / st.target_passes:>7.2f}x {base_s / spec_s:>6.2f}x"
        )

    print("\ntok/pass = tokens per big-model forward pass (plain decoding = 1.00)")
    print("passes   = reduction in BIG-model passes (the memory-bound bottleneck)")
    print("wall     = end-to-end wall-clock speedup. LOWER than the pass reduction,")
    print("           because the draft model's own passes aren't free — that gap is")
    print("           the real cost of speculation, and it shrinks as the target model")
    print("           grows relative to the draft (the regime this technique is for).")

    # a full example
    ids = tok.encode(PROMPTS[0])
    out, st = speculative_generate(target, draft, ids, max_new_tokens=N, lookahead=4, device=device)
    print(f"\nsample: {tok.decode(ids + out)!r}")
    print(f"  draft proposed {st.proposed} tokens, target accepted {st.accepted} "
          f"({st.acceptance_rate:.0%}) across {st.target_passes} passes")


if __name__ == "__main__":
    main()
