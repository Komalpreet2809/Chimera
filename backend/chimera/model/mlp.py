"""MLP — the 'compute' half of the block. Bite 13.

One word: COMPUTATION. After attention lets a token gather context, the MLP is
where that single token 'thinks' about what it gathered — alone, no looking at
neighbors. Structure is just: expand 4x -> nonlinearity -> contract back.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd                    # 768 -> 3072 (the 4x, Bite 13)
        self.c_fc = nn.Linear(config.n_embd, hidden)  # expand
        self.c_proj = nn.Linear(hidden, config.n_embd)  # contract
        # GPT-2's exact nonlinearity: GELU (a smooth cousin of ReLU). This is
        # the 'lets it represent non-straight-line patterns' function from Bite 13.
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)     # (B, T, 768) -> (B, T, 3072)   widen: room to think
        x = self.act(x)      # nonlinearity
        x = self.c_proj(x)   # (B, T, 3072) -> (B, T, 768)   narrow: back to the pipe
        return x
