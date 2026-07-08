"""Build Step 1 smoke test.

Proves two things before we write a single line of the model:
  1. The tokenizer round-trips (text -> ints -> text).  [Phase 0, Bites 3-4]
  2. Real GPT-2 weights actually download and load, and we can see their
     names and shapes -- these are the exact numbers our hand-built modules
     will pour themselves into in later steps.

Run:  python backend/scripts/step1_smoke.py
"""

import sys
from pathlib import Path

# Make `import chimera` work whether run from repo root or backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GPT2LMHeadModel

from chimera.model.tokenizer import Tokenizer


def demo_tokenizer() -> None:
    print("=" * 60)
    print("1) TOKENIZER  — text is just integers to the model")
    print("=" * 60)
    tok = Tokenizer("gpt2")
    text = "The cat sat on the mat"

    ids = tok.encode(text)
    back = tok.decode(ids)

    print(f"  text     : {text!r}")
    print(f"  encoded  : {ids}")
    print(f"  decoded  : {back!r}")
    print(f"  round-trip OK: {back == text}")
    print(f"  vocab size: {tok.vocab_size}")
    print("\n  token-by-token (id -> the text chunk it means):")
    for tid, piece in tok.tokens(text):
        print(f"    {tid:>6}  ->  {piece!r}")


def demo_weights() -> None:
    print("\n" + "=" * 60)
    print("2) REAL GPT-2 WEIGHTS — what our modules will load")
    print("=" * 60)
    print("  downloading/loading gpt2 (first run pulls ~500MB)...")
    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = hf.state_dict()

    total = sum(p.numel() for p in hf.parameters())
    print(f"  total parameters: {total:,}  (~124M — 'GPT-2 small')")

    print("\n  a few weight tensors (name -> shape):")
    interesting = [
        "transformer.wte.weight",          # token embedding table  (Bite 2)
        "transformer.wpe.weight",          # positional embedding    (Bite 3)
        "transformer.h.0.attn.c_attn.weight",  # block 0 Q,K,V proj  (Bite 6)
        "transformer.h.0.mlp.c_fc.weight",     # block 0 MLP expand  (Bite 13)
        "transformer.ln_f.weight",         # final LayerNorm         (Bite 12)
    ]
    for name in interesting:
        if name in sd:
            print(f"    {name:<40} {tuple(sd[name].shape)}")

    n_blocks = sum(1 for k in sd if k.startswith("transformer.h.") and k.endswith(".ln_1.weight"))
    print(f"\n  number of transformer blocks stacked: {n_blocks}  (our '×12')")
    print(f"  torch device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")


if __name__ == "__main__":
    demo_tokenizer()
    demo_weights()
    print("\nStep 1 complete. Tokenizer works; real weights are in hand.")
