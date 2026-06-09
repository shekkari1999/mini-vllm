"""Naive max_seq_len reservation vs paged block allocator (CPU only)."""

from __future__ import annotations

import math
import random

from minivllm.engine.block_manager import BlockAllocator

GPU_TOKEN_BUDGET = 32_768
MAX_SEQ_LEN = 512
BLOCK_SIZE = 16


def _workload(n: int = 300) -> list[int]:
    random.seed(42)
    lengths = [random.randint(8, 80) for _ in range(200)] + [
        random.randint(150, 512) for _ in range(100)
    ]
    random.shuffle(lengths)
    return lengths[:n]


def naive_capacity(seq_lengths: list[int], max_seq_len: int, budget: int) -> dict:
    slots_per_seq = max_seq_len
    served = min(len(seq_lengths), budget // slots_per_seq)
    allocated = served * slots_per_seq
    used = sum(seq_lengths[:served])
    wasted = allocated - used
    frag = wasted / allocated if allocated else 0.0
    return {
        "strategy": "naive_max_seq_len",
        "served": served,
        "allocated_tokens": allocated,
        "used_tokens": used,
        "wasted_tokens": wasted,
        "fragmentation": frag,
    }


def paged_capacity(seq_lengths: list[int], block_size: int, budget: int) -> dict:
    num_blocks = budget // block_size
    allocator = BlockAllocator(num_blocks, block_size)
    served = 0
    for length in seq_lengths:
        need = math.ceil(length / block_size)
        if len(allocator.free_list) < need:
            break
        for _ in range(need):
            allocator.allocate_block()
        served += 1
    allocated = (num_blocks - len(allocator.free_list)) * block_size
    used = sum(seq_lengths[:served])
    wasted = allocated - used
    frag = wasted / allocated if allocated else 0.0
    return {
        "strategy": f"paged_block_{block_size}",
        "served": served,
        "allocated_tokens": allocated,
        "used_tokens": used,
        "wasted_tokens": wasted,
        "fragmentation": frag,
    }


def run() -> dict:
    seq_lengths = _workload()
    naive = naive_capacity(seq_lengths, MAX_SEQ_LEN, GPU_TOKEN_BUDGET)
    paged = paged_capacity(seq_lengths, BLOCK_SIZE, GPU_TOKEN_BUDGET)

    cap_ratio = naive["served"] / paged["served"] if paged["served"] else 0.0
    # paged serves MORE — headline is paged/naive
    serve_ratio = paged["served"] / naive["served"] if naive["served"] else 0.0
    frag_reduction = (
        (naive["fragmentation"] - paged["fragmentation"]) / naive["fragmentation"] * 100
        if naive["fragmentation"]
        else 0.0
    )

    print("─" * 60)
    print("  Paged KV memory (CPU simulation)")
    print("─" * 60)
    print(f"  Token budget: {GPU_TOKEN_BUDGET:,}   requests: {len(seq_lengths)}")
    print()
    for r in (naive, paged):
        print(
            f"  {r['strategy']:<22}  served={r['served']:>4}  "
            f"frag={r['fragmentation']:>6.1%}  wasted={r['wasted_tokens']:>6,}"
        )
    print()
    print(f"  Paged serves {serve_ratio:.2f}× more concurrent sequences")
    print(f"  Fragmentation down {frag_reduction:.0f}%")
    print("─" * 60)

    return {
        "gpu_token_budget": GPU_TOKEN_BUDGET,
        "max_seq_len": MAX_SEQ_LEN,
        "block_size": BLOCK_SIZE,
        "num_requests": len(seq_lengths),
        "naive": naive,
        "paged": paged,
        "serve_ratio": serve_ratio,
        "fragmentation_reduction_pct": frag_reduction,
    }


if __name__ == "__main__":
    run()
