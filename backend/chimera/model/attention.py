"""Multi-Head Causal Self-Attention — the heart. Bites 5-11.

We implement the math explicitly (scores -> mask -> softmax -> weighted sum of
values) instead of calling a fused attention kernel, because:
  1. It maps 1:1 onto what you learned.
  2. We want to *expose* the attention weights later (the heatmap module).

The one word for this whole file: COMMUNICATION. This is the only place tokens
look at each other.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        # One matrix produces Q, K, and V at once: 768 -> 2304 (=768*3).
        # This is the (768, 2304) c_attn weight we saw in Step 1. (Bite 6)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # After heads recombine, one more matrix mixes them: 768 -> 768.
        # This is W_O from Bite 10.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (batch, seq, n_embd) -> same shape, but context-enriched."""
        B, T, C = x.shape

        # --- 1. Produce Q, K, V for every token (Bite 6) ---
        # Project to 3*C, then split into three (B, T, C) tensors.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # --- 2. Split each into heads (Bite 10) ---
        # (B, T, C) -> (B, n_head, T, head_dim): each head gets its own 64-wide slice.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- 3. Scores: every query dotted with every key (Bites 7-8) ---
        # (B, h, T, d) @ (B, h, d, T) -> (B, h, T, T): the relevance grid.
        scores = q @ k.transpose(-2, -1)
        # Scale by sqrt(head_dim) = sqrt(64) = 8, to keep magnitudes sane (Bite 8).
        scores = scores / math.sqrt(self.head_dim)

        # --- 4. Causal mask: forbid looking at the future (Bite 11) ---
        # Build a lower-triangular allow-map; set forbidden cells to -inf so
        # softmax gives them exactly 0 weight.
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))

        # --- 5. Softmax: scores -> weights that sum to 1 per row (Bite 8) ---
        attn = F.softmax(scores, dim=-1)                    # (B, h, T, T)

        # --- 6. Gather values: weighted blend of V (Bite 9) ---
        # (B, h, T, T) @ (B, h, T, d) -> (B, h, T, d): info physically moves.
        out = attn @ v

        # --- 7. Merge heads back to (B, T, C) and mix with W_O (Bite 10) ---
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out, (attn if return_attn else None)
