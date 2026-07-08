"""Load real pretrained GPT-2 weights into our hand-built GPT.

The only fiddly bit: GPT-2's linear layers are stored as HuggingFace `Conv1D`,
whose weight is (in_features, out_features) — the transpose of what nn.Linear
expects (out_features, in_features). So the four projection matrices per block
(attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj) get transposed on the way in.
Everything else copies straight across.
"""

from __future__ import annotations

import torch

from .config import GPTConfig, GPT2_SMALL
from .gpt import GPT

# Names whose weights are Conv1D and must be transposed.
_TRANSPOSE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def load_pretrained(model_name: str = "gpt2", config: GPTConfig = GPT2_SMALL) -> GPT:
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(model_name).state_dict()
    model = GPT(config)

    with torch.no_grad():
        # --- embeddings ---
        model.embedding.wte.weight.copy_(hf["transformer.wte.weight"])
        model.embedding.wpe.weight.copy_(hf["transformer.wpe.weight"])

        # --- each block ---
        for i in range(config.n_layer):
            src = f"transformer.h.{i}."
            blk = model.blocks[i]
            # LayerNorms
            blk.ln_1.weight.copy_(hf[src + "ln_1.weight"])
            blk.ln_1.bias.copy_(hf[src + "ln_1.bias"])
            blk.ln_2.weight.copy_(hf[src + "ln_2.weight"])
            blk.ln_2.bias.copy_(hf[src + "ln_2.bias"])
            # Attention (weights transposed, biases straight)
            blk.attn.c_attn.weight.copy_(hf[src + "attn.c_attn.weight"].t())
            blk.attn.c_attn.bias.copy_(hf[src + "attn.c_attn.bias"])
            blk.attn.c_proj.weight.copy_(hf[src + "attn.c_proj.weight"].t())
            blk.attn.c_proj.bias.copy_(hf[src + "attn.c_proj.bias"])
            # MLP (weights transposed, biases straight)
            blk.mlp.c_fc.weight.copy_(hf[src + "mlp.c_fc.weight"].t())
            blk.mlp.c_fc.bias.copy_(hf[src + "mlp.c_fc.bias"])
            blk.mlp.c_proj.weight.copy_(hf[src + "mlp.c_proj.weight"].t())
            blk.mlp.c_proj.bias.copy_(hf[src + "mlp.c_proj.bias"])

        # --- final layernorm ---
        model.ln_f.weight.copy_(hf["transformer.ln_f.weight"])
        model.ln_f.bias.copy_(hf["transformer.ln_f.bias"])

        # lm_head is tied to wte (already shared in GPT.__init__), nothing to do.

    model.eval()
    return model
