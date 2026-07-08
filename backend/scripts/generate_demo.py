"""Phase 1 finale: our from-scratch GPT generates real text, with per-token timing.

The timing foreshadows Phase 2: because there's no KV cache yet, each step
reprocesses the whole growing sequence, so per-token latency drifts UP.

Run:  python backend/scripts/generate_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.model.generate import SampleConfig, generate
from chimera.model.tokenizer import Tokenizer
from chimera.model.weights import load_pretrained

PROMPT = "The key to understanding how large language models work is"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("loading our hand-built GPT with real GPT-2 weights...\n")

    tok = Tokenizer("gpt2")
    model = load_pretrained("gpt2")
    prompt_ids = tok.encode(PROMPT)

    cfg = SampleConfig(max_new_tokens=40, temperature=0.8, top_k=40, greedy=False)

    print(f"PROMPT: {PROMPT}")
    print("-" * 60)
    print(PROMPT, end="", flush=True)

    # Measure the wall time between consecutive tokens. Since the generator runs
    # the forward pass right before yielding, this interval ~= per-token compute.
    latencies = []
    last = time.perf_counter()
    for tid in generate(model, prompt_ids, cfg, device=device):
        if device == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        latencies.append((now - last) * 1000.0)  # ms
        last = now
        print(tok.decode([tid]), end="", flush=True)

    print("\n" + "-" * 60)
    print(f"generated {len(latencies)} tokens")
    # Show the crime: compare the first few tokens vs the last few.
    first5 = sum(latencies[:5]) / 5
    last5 = sum(latencies[-5:]) / 5
    print(f"avg latency, first 5 tokens: {first5:6.1f} ms")
    print(f"avg latency, last  5 tokens: {last5:6.1f} ms")
    print(f"slowdown as sequence grew  : {last5 / first5:5.2f}x   <- the recompute crime")
    print("\nStep 7 complete. Phase 1 done: a GPT built from first principles, talking.")


if __name__ == "__main__":
    main()
