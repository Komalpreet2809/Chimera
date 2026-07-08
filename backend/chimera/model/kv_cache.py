"""KVCache — the fix for the recompute crime. Phase 2, Bites 1-6.

Stores the Keys and Values of every past token, per layer, so decode steps
only process the ONE new token. Queries are never cached (Bite 2): old tokens
are passive — they answer questions (K) and provide content (V), but never ask.

Shape per layer (Bite 6):  K, V each (batch, n_head, seq_len, head_dim)
where seq_len is the dimension that grows by 1 every decode step.
"""

from __future__ import annotations

import torch

from .config import GPTConfig


class KVCache:
    """A growing K/V store: one (K, V) pair per layer. Append and read."""

    def __init__(self, config: GPTConfig) -> None:
        self.n_layer = config.n_layer
        # One slot per layer; starts empty (None) until the first append.
        self._k: list[torch.Tensor | None] = [None] * config.n_layer
        self._v: list[torch.Tensor | None] = [None] * config.n_layer

    @property
    def seq_len(self) -> int:
        """How many tokens are cached so far (0 if empty)."""
        return 0 if self._k[0] is None else self._k[0].size(2)

    def append(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this layer's new K,V and return the FULL cached K,V.

        k_new, v_new: (batch, n_head, new_tokens, head_dim)
          - prefill: new_tokens = whole prompt (fills the cache in one go)
          - decode : new_tokens = 1            (the +1 growth per step)
        """
        if self._k[layer] is None:
            self._k[layer] = k_new
            self._v[layer] = v_new
        else:
            # Concatenate along the seq_len dimension (dim=2) — the growing one.
            self._k[layer] = torch.cat([self._k[layer], k_new], dim=2)
            self._v[layer] = torch.cat([self._v[layer], v_new], dim=2)
        return self._k[layer], self._v[layer]

    def memory_bytes(self) -> int:
        """Total bytes held — the 'price tag' from Bite 4, measurable live.

        This number feeds the telemetry stream later: it's the memory line
        you'll watch grow token-by-token in the UI.
        """
        total = 0
        for k, v in zip(self._k, self._v):
            if k is not None:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total
