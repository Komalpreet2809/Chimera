"""Phase 3 demo: the InferenceEngine interleaving two requests.

The old generate() ran one user start-to-finish while everyone else waited.
The engine's step() lets us alternate: A, B, A, B, ... — both users watch
their answers grow at the same time. This is the door Phase 4's scheduler
walks through.

Run:  python backend/scripts/phase3_engine_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chimera.engine.engine import InferenceEngine
from chimera.model.generate import SampleConfig
from chimera.model.tokenizer import Tokenizer
from chimera.model.weights import load_pretrained


def main() -> None:
    print("loading model...")
    engine = InferenceEngine(
        model=load_pretrained("gpt2"),
        tokenizer=Tokenizer("gpt2"),
        sample_cfg=SampleConfig(greedy=True),  # deterministic for the demo
    )

    # two users arrive with different prompts
    a = engine.submit("The best way to learn programming is", max_new_tokens=12)
    b = engine.submit("In the distant future, humanity will", max_new_tokens=12)
    print(f"submitted: request #{a.id} and request #{b.id}\n")

    print("interleaved steps (A, B, A, B, ...):")
    print(f"  {'req':>4} {'kind':>8} {'token':<14} {'lat(ms)':>8} {'cache':>6} {'cacheKB':>8}")
    live = [a, b]
    while live:
        for req in list(live):
            ev = engine.step(req)
            print(
                f"  #{ev.request_id:>3} {ev.kind:>8} {ev.token_text!r:<14}"
                f" {ev.latency_ms:>8.1f} {ev.cache_tokens:>6} {ev.cache_bytes/1024:>8.0f}"
            )
            if req.is_finished:
                live.remove(req)
                print(f"  #{req.id:>3} FINISHED ({req.finish_reason.value})")

    print("\nfinal outputs:")
    for req in (a, b):
        print(f"  #{req.id}: {engine.text_of(req)!r}")


if __name__ == "__main__":
    main()
