"""BatchedKVCache — N per-request caches riding ONE forward pass. Phase 4.

Each request keeps its own KVCache (they can't share — different sequences).
For a batched decode step, this adapter:
  1. routes each row's new K,V into that request's own cache (append),
  2. stacks all caches into one tensor, LEFT-padding shorter ones so every
     row is end-aligned (the newest token sits in the last column),
  3. exposes a padding mask so attention ignores the pad slots.

NOTE the cost: stacking copies every request's whole cache, every step, into
one contiguous padded tensor — and the padding itself is wasted space. This
pain (contiguous-memory bookkeeping for ragged caches) is EXACTLY the problem
PagedAttention (Phase 5) exists to solve. We feel it here on purpose.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..model.kv_cache import KVCache


class BatchedKVCache:
    """Duck-types KVCache's interface (append) for a batch of requests."""

    def __init__(self, caches: list[KVCache]) -> None:
        self.caches = caches
        # Per-row lengths BEFORE this step's append — also the per-row pos_offset.
        self.past_lens = [c.seq_len for c in caches]

    def pos_offsets(self, device) -> torch.Tensor:
        return torch.tensor(self.past_lens, device=device)

    def padding_mask(self, device) -> torch.Tensor:
        """(B, maxT) bool, True = real slot. Lengths are AFTER the +1 append."""
        lens = torch.tensor(self.past_lens, device=device) + 1
        max_t = int(lens.max())
        cols = torch.arange(max_t, device=device).unsqueeze(0)   # (1, maxT)
        # Left-padded: row i's real slots are the LAST lens[i] columns.
        return cols >= (max_t - lens.unsqueeze(1))

    def append(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """k_new, v_new: (B, n_head, 1, head_dim) — one new token per row.

        Routes row i's slice into cache i, then returns the stacked,
        left-padded full K and V: (B, n_head, maxT, head_dim).
        """
        ks, vs = [], []
        for i, cache in enumerate(self.caches):
            fk, fv = cache.append(layer, k_new[i : i + 1], v_new[i : i + 1])
            ks.append(fk)
            vs.append(fv)

        max_t = max(k.size(2) for k in ks)
        # F.pad last-dims order: (d_left, d_right, T_left, T_right) — pad T on the LEFT.
        ks = [F.pad(k, (0, 0, max_t - k.size(2), 0)) for k in ks]
        vs = [F.pad(v, (0, 0, max_t - v.size(2), 0)) for v in vs]
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)
