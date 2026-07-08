"""Phase 2 payoff: naive vs KV-cached generation, measured.

Two experiments:
  1. Per-step forward latency as the sequence grows — naive reprocesses
     everything (climbs), cached processes 1 token (flat).
  2. End-to-end generation wall time from a long prompt.

Run:  python backend/scripts/phase2_benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.model.config import GPT2_SMALL
from chimera.model.generate import SampleConfig, generate
from chimera.model.kv_cache import KVCache
from chimera.model.tokenizer import Tokenizer
from chimera.model.weights import load_pretrained


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer("gpt2")
    model = load_pretrained("gpt2").to(device)

    # warm up the GPU so one-time kernel costs don't pollute the numbers
    with torch.no_grad():
        for _ in range(3):
            model(torch.zeros(1, 16, dtype=torch.long, device=device))
    if device == "cuda":
        torch.cuda.synchronize()

    # ---- 1. per-step decode latency at various sequence lengths ----
    print("Per-step latency when the sequence is already N tokens long:")
    print(f"  {'N':>6}  {'naive (ms)':>12}  {'cached (ms)':>12}")
    for n in [64, 256, 512, 900]:
        seq = torch.zeros(1, n, dtype=torch.long, device=device)

        # naive: one step = reprocess all N tokens
        with torch.no_grad():
            t = time.perf_counter()
            for _ in range(10):
                model(seq)
            if device == "cuda":
                torch.cuda.synchronize()
            naive_ms = (time.perf_counter() - t) / 10 * 1000

        # cached: prefill N tokens once, then one step = process 1 token
        cache = KVCache(GPT2_SMALL)
        one = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            model(seq, cache=cache)  # prefill (not timed — happens once)
            t = time.perf_counter()
            for _ in range(10):
                model(one, cache=cache)
            if device == "cuda":
                torch.cuda.synchronize()
            cached_ms = (time.perf_counter() - t) / 10 * 1000

        print(f"  {n:>6}  {naive_ms:>12.2f}  {cached_ms:>12.2f}")

    # ---- 2. end-to-end: generate 200 tokens from a 400-token prompt ----
    prompt_ids = tok.encode("The story of computing " * 100)[:400]
    cfg = SampleConfig(max_new_tokens=200, greedy=True)

    results = {}
    for label, use_cache in [("naive", False), ("cached", True)]:
        t = time.perf_counter()
        out = list(generate(model, prompt_ids, cfg, device=device, use_cache=use_cache))
        if device == "cuda":
            torch.cuda.synchronize()
        results[label] = (time.perf_counter() - t, out)

    naive_s, naive_out = results["naive"]
    cached_s, cached_out = results["cached"]
    print(f"\nEnd-to-end: 400-token prompt, 200 new tokens (greedy):")
    print(f"  naive : {naive_s:6.2f} s")
    print(f"  cached: {cached_s:6.2f} s")
    print(f"  speedup: {naive_s / cached_s:.1f}x")
    print(f"  outputs identical: {naive_out == cached_out}")


if __name__ == "__main__":
    main()
