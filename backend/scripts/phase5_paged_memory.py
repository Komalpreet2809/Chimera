"""Phase 5: PagedAttention vs naive contiguous reservation.

A naive engine can't know how long a request will run, so it RESERVES each
request's worst case (max_seq_len) in contiguous memory. PagedAttention
allocates 16-token blocks on demand and frees them on completion.

Same memory budget, same workload — how much is wasted, and how many users fit?

Run:  python backend/scripts/phase5_paged_memory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chimera.memory.allocator import BlockAllocator, OutOfMemoryError
from chimera.model.config import GPT2_SMALL

MAX_SEQ = 512        # what a request COULD generate (worst case)
BLOCK_SIZE = 16
KB_PER_TOKEN = 72    # measured in Phase 2 (fp32, 12 layers, K+V)
BUDGET_MB = 512

# A realistic mix: mostly short answers, a few long ones. Repeated to a big
# enough workload that BOTH policies hit their true capacity ceiling.
_MIX = [12, 340, 25, 60, 8, 480, 33, 15, 120, 44,
        19, 200, 7, 95, 28, 410, 51, 16, 73, 22]
ACTUAL_LENGTHS = _MIX * 5   # 100 requests


def main() -> None:
    budget_kb = BUDGET_MB * 1024
    print(f"GPU KV budget: {BUDGET_MB} MB   |   worst-case request length: {MAX_SEQ} tokens")
    print(f"workload: {len(ACTUAL_LENGTHS)} requests, actual lengths {min(ACTUAL_LENGTHS)}-{max(ACTUAL_LENGTHS)} tokens\n")

    # ---------- naive: reserve max_seq_len per request, contiguous ----------
    reserve_kb = MAX_SEQ * KB_PER_TOKEN
    naive_capacity = budget_kb // reserve_kb              # how many users fit AT ALL
    served = ACTUAL_LENGTHS[:naive_capacity]
    naive_used = sum(served) * KB_PER_TOKEN
    naive_reserved = len(served) * reserve_kb
    naive_waste = 1 - naive_used / naive_reserved

    print("NAIVE (reserve worst-case, contiguous):")
    print(f"  reservation per request : {reserve_kb/1024:.0f} MB")
    print(f"  requests that fit       : {naive_capacity} of {len(ACTUAL_LENGTHS)}")
    print(f"  memory actually used    : {naive_used/1024:.0f} MB of {naive_reserved/1024:.0f} MB reserved")
    print(f"  WASTED                  : {naive_waste:.0%}")

    # ---------- paged: 16-token blocks, on demand ----------
    kb_per_block = BLOCK_SIZE * KB_PER_TOKEN
    num_blocks = int(budget_kb // kb_per_block)
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)

    tables, admitted = [], 0
    for length in ACTUAL_LENGTHS:
        table = alloc.new_table()
        if not alloc.can_append(table, length):
            break
        alloc.append_tokens(table, length)     # blocks allocated ON DEMAND
        tables.append(table)
        admitted += 1

    paged_used = sum(t.num_tokens for t in tables) * KB_PER_TOKEN
    paged_alloc = sum(t.num_blocks for t in tables) * kb_per_block
    paged_waste = 1 - paged_used / paged_alloc

    print("\nPAGED (16-token blocks, allocated on demand):")
    print(f"  block pool              : {num_blocks} blocks x {BLOCK_SIZE} tokens")
    print(f"  requests that fit       : {admitted} of {len(ACTUAL_LENGTHS)}")
    print(f"  memory actually used    : {paged_used/1024:.0f} MB of {paged_alloc/1024:.0f} MB allocated")
    print(f"  WASTED                  : {paged_waste:.1%}  (only the tail of each last block)")

    print("\n" + "=" * 58)
    print(f"  concurrent users:  {naive_capacity}  ->  {admitted}   ({admitted/naive_capacity:.1f}x more)")
    print(f"  memory waste    :  {naive_waste:.0%}  ->  {paged_waste:.1%}")
    print("=" * 58)

    # ---------- reuse: finished requests hand blocks straight back ----------
    print("\nblock reuse (a request finishes, its blocks are instantly reusable):")
    print(f"  blocks in use: {alloc.num_allocated}/{alloc.num_blocks} ({alloc.utilization:.0%})")
    freed = alloc.free_table(tables[1])   # the 340-token request finishes
    print(f"  request #1 (340 tok) finishes -> freed {len(freed)} blocks")
    print(f"  blocks in use: {alloc.num_allocated}/{alloc.num_blocks} ({alloc.utilization:.0%})")

    more = 0
    for length in ACTUAL_LENGTHS[admitted:]:
        t = alloc.new_table()
        if not alloc.can_append(t, length):
            break
        alloc.append_tokens(t, length)
        more += 1
    print(f"  -> {more} queued request(s) admitted immediately into the freed space")


if __name__ == "__main__":
    main()
