"""Embedding — the input bookend. Bites 2-3.

Turns token integers into vectors, and adds a 'where am I' position vector.
An nn.Embedding is *exactly* the lookup table from Bite 2: a big matrix with
one row per index, and 'embedding' just means 'grab the row.' No math, no
intelligence — pure lookup, on both the token side and the position side.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import GPTConfig


class Embedding(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        # wte: token embedding table  — (vocab_size, n_embd), one row per token
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # wpe: position embedding table — (n_ctx, n_embd), one row per position
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)

    def forward(
        self, token_ids: torch.Tensor, pos_offset: int | torch.Tensor = 0
    ) -> torch.Tensor:
        """token_ids: (batch, seq) integers  ->  (batch, seq, n_embd) vectors.

        pos_offset (Phase 2): during cached decode we feed only the newest
        token, but its position is NOT 0 — it's wherever the sequence
        currently ends (= how many tokens are already in the KV cache).

        Phase 4: in a batched decode step, each row is at a DIFFERENT length,
        so pos_offset may be a (batch,) tensor of per-row offsets.
        """
        _, seq_len = token_ids.shape

        # "what each token is" — grab each token's row from wte.
        tok_vecs = self.wte(token_ids)                     # (batch, seq, n_embd)

        # "where each token is" — positions offset..offset+seq-1 (per row if batched).
        steps = torch.arange(seq_len, device=token_ids.device)
        if isinstance(pos_offset, torch.Tensor):
            positions = pos_offset.unsqueeze(1) + steps    # (batch, seq)
        else:
            positions = pos_offset + steps                 # (seq,)
        pos_vecs = self.wpe(positions)                     # (seq, n_embd)

        # Add them: every vector now carries BOTH identity and position (Bite 3).
        # pos_vecs broadcasts across the batch dimension.
        return tok_vecs + pos_vecs
