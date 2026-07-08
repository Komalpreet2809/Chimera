"""LayerNorm — the 'voltage regulator'. Bite 12.

Takes a vector's numbers and re-scales them to a consistent, well-behaved range
(centered, unit spread), then applies a learned scale (weight) and shift (bias).
We implement it by hand rather than nn.LayerNorm so the 'center + scale' idea is
explicit: it doesn't change what the token *means*, it tames the magnitudes so
the signal stays clean across 12 stacked blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import GPTConfig


class LayerNorm(nn.Module):
    def __init__(self, config: GPTConfig, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Learned per-feature scale (gamma) and shift (beta), both width n_embd.
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Statistics over the last dim (the 768 features) — per token, per position.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Center to mean 0, scale to unit spread (eps guards divide-by-zero).
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Then let the model re-scale/shift with learned parameters.
        return self.weight * x_norm + self.bias
