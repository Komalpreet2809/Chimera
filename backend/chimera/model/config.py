"""GPTConfig — the handful of numbers that fully define the model's shape.

Every dimension here is something you already understand from Phase 1:
these are the exact values behind the weight shapes we saw in Step 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257   # number of tokens the model knows        (Bite 3)
    n_ctx: int = 1024         # max sequence length (positions)         (Bite 3, wpe rows)
    n_embd: int = 768         # width of every vector — d_model         (Bite 2)
    n_head: int = 12          # attention heads run in parallel         (Bite 10)
    n_layer: int = 12         # transformer blocks stacked              (Bite 4)

    @property
    def head_dim(self) -> int:
        """Width of each head's slice. 768 / 12 = 64 — the '64' from Bite 8/10."""
        return self.n_embd // self.n_head

    def __post_init__(self) -> None:
        # n_embd must divide evenly into heads, or the split in Bite 10 breaks.
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


# The canonical "GPT-2 small" preset — the weights we loaded in Step 1.
GPT2_SMALL = GPTConfig()
