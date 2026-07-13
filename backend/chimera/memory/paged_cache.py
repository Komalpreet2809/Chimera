"""PagedKVCache — KV storage backed by non-contiguous physical blocks. Phase 5.

The physical pool, per layer, is one big tensor:

    (num_blocks, n_head, block_size, head_dim)

A request's tokens are scattered across whatever blocks it was given. Its
BlockTable translates logical token index -> (physical block, offset), so
attention can still see a normal contiguous (1, n_head, seq, head_dim) view:
we GATHER the request's blocks on the fly.

That indirection is the whole trick — the same one virtual memory plays on
processes. The request thinks it has a contiguous cache; physically it doesn't.
"""

from __future__ import annotations

import torch

from ..model.config import GPTConfig
from .allocator import BlockAllocator, BlockTable


class PagedKVPool:
    """The physical KV memory: block pool + allocator, shared by all requests."""

    def __init__(
        self,
        config: GPTConfig,
        num_blocks: int = 512,
        block_size: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.block_size = block_size
        self.device = device
        self.allocator = BlockAllocator(num_blocks, block_size)

        shape = (num_blocks, config.n_head, block_size, config.head_dim)
        # One physical K pool and V pool per layer.
        self.k_pool = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        self.v_pool = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]

    def to(self, device: str) -> "PagedKVPool":
        """Move the physical block pool onto a device (the engine calls this)."""
        if device != self.device:
            self.k_pool = [t.to(device) for t in self.k_pool]
            self.v_pool = [t.to(device) for t in self.v_pool]
            self.device = device
        return self

    @property
    def total_bytes(self) -> int:
        per = self.k_pool[0].numel() * self.k_pool[0].element_size()
        return per * 2 * self.config.n_layer

    def new_cache(self) -> "PagedKVCache":
        return PagedKVCache(self)

    def stats(self) -> dict:
        a = self.allocator
        return {
            "num_blocks": a.num_blocks,
            "blocks_used": a.num_allocated,
            "blocks_free": a.num_free,
            "block_utilization": a.utilization,
            "pool_bytes": self.total_bytes,
        }


class PagedKVCache:
    """One request's paged cache. Duck-types KVCache (seq_len, append)."""

    def __init__(self, pool: PagedKVPool) -> None:
        self.pool = pool
        self.table: BlockTable = pool.allocator.new_table()

    # --- KVCache-compatible surface ---
    @property
    def seq_len(self) -> int:
        return self.table.num_tokens

    def memory_bytes(self) -> int:
        """Physical bytes actually held (blocks * bytes-per-block), not logical."""
        cfg = self.pool.config
        per_block = (
            cfg.n_head * self.pool.block_size * cfg.head_dim
            * self.pool.k_pool[0].element_size() * 2 * cfg.n_layer
        )
        return self.table.num_blocks * per_block

    def append(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """k_new, v_new: (1, n_head, T_new, head_dim). Writes into blocks, returns full view."""
        n_new = k_new.size(2)

        # Layer 0 grows the table (allocating blocks on demand); the other
        # layers reuse the same block mapping — one table serves all layers.
        # So by the time layer>0 runs, num_tokens is ALREADY advanced: rewind
        # to find where this step's tokens start.
        if layer == 0:
            start = self.table.num_tokens
            self.pool.allocator.append_tokens(self.table, n_new)
        else:
            start = self.table.num_tokens - n_new

        # Scatter the new K/V into their physical slots.
        for j in range(n_new):
            block_id, offset = self.table.slot_for(start + j)
            self.pool.k_pool[layer][block_id, :, offset, :] = k_new[0, :, j, :]
            self.pool.v_pool[layer][block_id, :, offset, :] = v_new[0, :, j, :]

        return self.view(layer)

    def view(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather this request's scattered blocks into a contiguous K, V view."""
        blocks = torch.tensor(self.table.physical_blocks, device=self.pool.device)
        t = self.table.num_tokens
        cfg = self.pool.config

        def gather(pool: list[torch.Tensor]) -> torch.Tensor:
            g = pool[layer][blocks]                    # (nb, n_head, block_size, head_dim)
            g = g.permute(1, 0, 2, 3).reshape(         # (n_head, nb*block_size, head_dim)
                cfg.n_head, -1, cfg.head_dim
            )
            return g[:, :t, :].unsqueeze(0)            # (1, n_head, seq, head_dim)

        return gather(self.pool.k_pool), gather(self.pool.v_pool)

    def free(self) -> list[int]:
        """Return every block to the pool — instantly reusable by other requests."""
        return self.pool.allocator.free_table(self.table)
