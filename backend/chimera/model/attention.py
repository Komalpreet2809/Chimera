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
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        cache=None,
        layer_idx: int = 0,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (batch, seq, n_embd) -> same shape, but context-enriched.

        With a KVCache (Phase 2): x holds only the NEW tokens (whole prompt at
        prefill, 1 token per decode step). We append their K,V to the cache and
        attend over everything cached so far. Queries are never cached (Bite 2).
        """
        B, T, C = x.shape

        # --- 1. Produce Q, K, V for the new tokens only (Bite 6) ---
        # Project to 3*C, then split into three (B, T, C) tensors.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # --- 2. Split each into heads (Bite 10) ---
        # (B, T, C) -> (B, n_head, T, head_dim): each head gets its own 64-wide slice.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- 2b. KV cache (Phase 2, Bite 3) ---
        # Append the new K,V; get back the FULL history to attend over.
        # NOTE: past_len must come from THIS layer's tensors (total - new), not
        # cache.seq_len — earlier layers have already appended this step's
        # token by the time later layers run, so seq_len would be off by one.
        if cache is not None:
            k, v = cache.append(layer_idx, k, v)
        T_total = k.size(2)   # cached tokens + new tokens
        past_len = T_total - T  # 0 when no cache / prefill-from-empty

        # --- 3. Scores: every (new) query dotted with every key (Bites 7-8) ---
        # (B, h, T, d) @ (B, h, d, T_total) -> (B, h, T, T_total).
        scores = q @ k.transpose(-2, -1)
        # Scale by sqrt(head_dim) = sqrt(64) = 8, to keep magnitudes sane (Bite 8).
        scores = scores / math.sqrt(self.head_dim)

        # --- 4. Causal mask: forbid looking at the future (Bite 11) ---
        # Query i sits at ABSOLUTE position past_len+i, so it may see keys
        # 0..past_len+i. With no cache (past_len=0) this is the usual triangle;
        # for a single decode token it's a full row of ✓ (all past is visible).
        q_pos = torch.arange(T, device=x.device).unsqueeze(1) + past_len
        k_pos = torch.arange(T_total, device=x.device).unsqueeze(0)
        causal = k_pos <= q_pos                              # (T, T_total)
        scores = scores.masked_fill(~causal, float("-inf"))

        # --- 4b. Padding mask (Phase 4) ---
        # In a batched step, rows with shorter caches are left-padded to the
        # longest; those pad slots hold garbage and must get zero attention.
        # key_padding_mask: (B, T_total), True = real slot, False = padding.
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                ~key_padding_mask[:, None, None, :], float("-inf")
            )

        # --- 5. Softmax: scores -> weights that sum to 1 per row (Bite 8) ---
        attn = F.softmax(scores, dim=-1)                    # (B, h, T, T)

        # --- 6. Gather values: weighted blend of V (Bite 9) ---
        # (B, h, T, T) @ (B, h, T, d) -> (B, h, T, d): info physically moves.
        out = attn @ v

        # --- 7. Merge heads back to (B, T, C) and mix with W_O (Bite 10) ---
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out, (attn if return_attn else None)
