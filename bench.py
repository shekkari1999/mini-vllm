"""Throughput benchmark shortcut (sequential vs continuous batching)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.benchmark_batching import run
from minivllm.config import BENCHMARK_BATCH_SIZE, BENCHMARK_MAX_TOKENS, Config

if __name__ == "__main__":
    import argparse
    import json

    cfg = Config()
    p = argparse.ArgumentParser(description="Benchmark mini-vllm throughput")
    p.add_argument("--model", default=cfg.model)
    p.add_argument("--num-blocks", type=int, default=cfg.num_blocks)
    p.add_argument("--max-batch-size", type=int, default=cfg.max_batch_size)
    p.add_argument("--max-tokens", type=int, default=BENCHMARK_MAX_TOKENS)
    p.add_argument("--batch-size", type=int, default=BENCHMARK_BATCH_SIZE)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--save", type=Path, default=Path("benchmarks/results/latest.json"))
    args = p.parse_args()

    result = run(
        args.model,
        num_blocks=args.num_blocks,
        max_batch_size=args.max_batch_size,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        warmup=args.warmup,
    )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"batching": result}, indent=2) + "\n")
    print(f"\nSaved → {args.save}")
