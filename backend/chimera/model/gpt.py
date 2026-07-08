"""GPT — the whole model, assembled. Bite 15.

text-ids -> embedding -> 12 blocks -> final LayerNorm -> unembed -> logits.
This is the entire skeleton of every LLM you've used, just small.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import TransformerBlock
from .config import GPTConfig, GPT2_SMALL
from .embedding import Embedding
from .layernorm import LayerNorm


class GPT(nn.Module):
    def __init__(self, config: GPTConfig = GPT2_SMALL) -> None:
        super().__init__()
        self.config = config

        self.embedding = Embedding(config)                       # input bookend
        self.blocks = nn.ModuleList(                             # the stack of 12
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config)                           # final stabilizer
        # Unembed: 768 -> vocab logits. No bias (matches GPT-2).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the unembed matrix IS the token embedding matrix.
        # Same table used to turn tokens->vectors is reused vectors->tokens.
        self.lm_head.weight = self.embedding.wte.weight

    def forward(
        self, token_ids: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """token_ids: (batch, seq) -> logits: (batch, seq, vocab_size)."""
        x = self.embedding(token_ids)                # ints -> vectors (+position)

        attentions: list[torch.Tensor] = []
        for block in self.blocks:                    # communicate/compute x12
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attentions.append(attn)

        x = self.ln_f(x)                             # final normalize
        logits = self.lm_head(x)                      # -> score per vocab token
        return logits, attentions
