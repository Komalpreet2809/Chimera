"""Block allocator + block tables — the OS analogy. Phase 5.

The problem (felt in Phase 4): a KV cache must live in contiguous memory and
grows unpredictably. A naive engine therefore RESERVES each request's
worst-case length up front (max_seq_len). A request that stops after 20 tokens
but reserved 512 wastes 96% of its slab — and that waste is what caps how many
users fit on the GPU.

The fix, borrowed straight from operating systems:
  * Physical memory is carved into fixed-size BLOCKS (pages).
  * Each request gets a BLOCK TABLE: a list mapping its logical block index
    -> a physical block id. Blocks need not be adjacent.
  * Blocks are allocated ON DEMAND (only when the request actually fills one)
    and freed the instant the request finishes.

Waste is then bounded by the tail of the last partial block — a few tokens —
instead of an entire worst-case reservation. Same trick that let 1960s OSes
stop requiring contiguous RAM per process.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OutOfMemoryError(RuntimeError):
    """The block pool is exhausted — no free physical blocks left."""


@dataclass
class BlockTable:
    """One request's logical -> physical block mapping (its 'page table')."""

    block_size: int
    physical_blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.physical_blocks)

    @property
    def capacity(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def slots_free_in_last_block(self) -> int:
        return self.capacity - self.num_tokens

    def slot_for(self, token_index: int) -> tuple[int, int]:
        """Logical token index -> (physical_block_id, offset_within_block).

        This is address translation, exactly like a CPU's MMU: the logical
        address is split into (block number, offset), and the block number is
        looked up in the page table to find where it physically lives.
        """
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size
        return self.physical_blocks[logical_block], offset


class BlockAllocator:
    """A pool of fixed-size physical blocks. allocate() / free() / reuse."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Free list. Blocks are handed out and returned; ids get REUSED, which
        # is why this never fragments the way a contiguous allocator does.
        self._free: list[int] = list(range(num_blocks))
        self._allocated: set[int] = set()

    # ---- stats (these drive the memory-map visualization) ----
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_allocated(self) -> int:
        return len(self._allocated)

    @property
    def utilization(self) -> float:
        return self.num_allocated / self.num_blocks if self.num_blocks else 0.0

    def allocate_block(self) -> int:
        if not self._free:
            raise OutOfMemoryError("no free KV blocks")
        block_id = self._free.pop()
        self._allocated.add(block_id)
        return block_id

    def free_block(self, block_id: int) -> None:
        self._allocated.discard(block_id)
        self._free.append(block_id)

    # ---- the operations a request actually performs ----
    def new_table(self) -> BlockTable:
        return BlockTable(block_size=self.block_size)

    def can_append(self, table: BlockTable, num_new_tokens: int = 1) -> bool:
        needed = self._blocks_needed(table, num_new_tokens)
        return needed <= self.num_free

    def _blocks_needed(self, table: BlockTable, num_new_tokens: int) -> int:
        deficit = num_new_tokens - table.slots_free_in_last_block
        if deficit <= 0:
            return 0
        return (deficit + self.block_size - 1) // self.block_size

    def append_tokens(self, table: BlockTable, num_new_tokens: int = 1) -> list[int]:
        """Grow a request's table by n tokens, allocating blocks ON DEMAND.

        Returns the physical block ids newly allocated (for the UI animation).
        """
        newly = []
        for _ in range(self._blocks_needed(table, num_new_tokens)):
            block_id = self.allocate_block()
            table.physical_blocks.append(block_id)
            newly.append(block_id)
        table.num_tokens += num_new_tokens
        return newly

    def free_table(self, table: BlockTable) -> list[int]:
        """Request finished — return ALL its blocks to the pool for reuse."""
        freed = list(table.physical_blocks)
        for block_id in freed:
            self.free_block(block_id)
        table.physical_blocks.clear()
        table.num_tokens = 0
        return freed
