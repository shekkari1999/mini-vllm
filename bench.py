"""Throughput benchmark shortcut (sequential vs continuous batching)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.benchmark_batching import run

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from minivllm import Config

    p = argparse.ArgumentParser(description="Benchmark mini-vllm throughput")
    p.add_argument("--model", default=Config().model)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup", action="store_true")
    p.add_argument("--save", type=Path, default=Path("benchmarks/results/latest.json"))
    args = p.parse_args()

    import json

    result = run(args.model, max_tokens=args.max_tokens, batch_size=args.batch_size, warmup=args.warmup)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps({"batching": result}, indent=2) + "\n")
    print(f"\nSaved → {args.save}")
