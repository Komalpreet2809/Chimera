"""TransformerBlock — one repeated unit. Bites 4, 12.

The rhythm: communicate, then compute. Each star (attention, MLP) is preceded
by a LayerNorm (stabilizer) and wrapped in a residual add (the highway). GPT-2
uses the 'pre-norm' arrangement: normalize BEFORE the sublayer, add the result
back to the untouched input.

    x = x + Attention(LayerNorm(x))    # communicate, then add to highway
    x = x + MLP(LayerNorm(x))          # compute,     then add to highway
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .config import GPTConfig
from .layernorm import LayerNorm
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config)          # stabilizer before attention
        self.attn = MultiHeadAttention(config)  # communicate
        self.ln_2 = LayerNorm(config)          # stabilizer before MLP
        self.mlp = MLP(config)                 # compute

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        cache=None,
        layer_idx: int = 0,
        key_padding_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Communicate: attention reads normalized x, result added to the highway.
        attn_out, attn_weights = self.attn(
            self.ln_1(x),
            return_attn=return_attn,
            cache=cache,
            layer_idx=layer_idx,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        # Compute: MLP reads normalized x, result added to the highway.
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights
