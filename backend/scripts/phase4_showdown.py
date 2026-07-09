"""Phase 4 showdown: sequential vs static vs continuous batching.

Same workload for all three: 20 users arrive at once with answers of very
different lengths (5 to 60 tokens) — the straggler-heavy case that separates
the policies.

Run:  python backend/scripts/phase4_showdown.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chimera.engine.engine import InferenceEngine
from chimera.engine.scheduler import Scheduler
from chimera.model.generate import SampleConfig
from chimera.model.tokenizer import Tokenizer
from chimera.model.weights import load_pretrained

PROMPTS = [
    "The history of computing began",
    "A recipe for perfect pasta:",
    "The best way to learn a language is",
    "In space, no one can",
    "The stock market today",
    "Once upon a time in a small village",
    "The theory of relativity says",
    "My favorite programming language is",
    "The ocean is home to",
    "Climate change will require",
    "The fastest land animal is",
    "To build a house you first",
    "Music has the power to",
    "The internet was invented",
    "A good leader always",
    "The human brain contains",
    "Coffee is made by",
    "The tallest mountain on Earth",
    "Democracy depends on",
    "The future of transportation",
]
# very unequal answer lengths -> stragglers (5 to 60 tokens)
LENGTHS = [5, 60, 12, 45, 8, 55, 20, 10, 50, 15, 6, 40, 25, 9, 35, 18, 7, 30, 14, 22]


def race(policy: str) -> dict:
    engine = InferenceEngine(
        load_pretrained("gpt2"), Tokenizer("gpt2"),
        sample_cfg=SampleConfig(greedy=True),
    )
    sched = Scheduler(engine, max_batch_size=8, policy=policy)
    for prompt, n in zip(PROMPTS, LENGTHS):
        sched.submit(prompt, max_new_tokens=n)
    return sched.run_to_completion()


def main() -> None:
    print(f"workload: {len(PROMPTS)} users at once, 5-60 tokens each, 8 seats\n")
    print(f"{'policy':>12} {'wall(s)':>9} {'tok/s':>8} {'avg TTFT(s)':>12} {'seat util':>10}")
    for policy in ("sequential", "static", "continuous"):
        r = race(policy)
        print(
            f"{policy:>12} {r['wall_s']:>9.1f} {r['throughput_tok_s']:>8.1f}"
            f" {r['avg_ttft_s']:>12.2f} {r['utilization']:>9.0%}"
        )
    print("\nTTFT = time-to-first-token (how long a user stares at a blank screen)")
    print("seat util = filled batch slots / available batch slots")


if __name__ == "__main__":
    main()
