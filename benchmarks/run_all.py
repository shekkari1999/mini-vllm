"""Run all benchmarks → benchmarks/results/latest.json"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_batching import run as run_batching
from benchmarks.benchmark_memory import run as run_memory
from benchmarks.benchmark_ttft import run as run_ttft
from minivllm.config import BENCHMARK_BATCH_SIZE, BENCHMARK_MAX_TOKENS, BENCHMARK_TTFT_REQUESTS, Config


def main():
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("benchmarks/results/latest.json"),
    )
    parser.add_argument("--model", default=cfg.model)
    parser.add_argument("--num-blocks", type=int, default=cfg.num_blocks)
    parser.add_argument("--max-batch-size", type=int, default=cfg.max_batch_size)
    parser.add_argument("--max-tokens", type=int, default=BENCHMARK_MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=BENCHMARK_BATCH_SIZE)
    parser.add_argument("--num-requests", type=int, default=BENCHMARK_TTFT_REQUESTS)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    save = args.save if args.save.is_absolute() else ROOT / args.save
    out: dict = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "benchmark_config": {
                "num_blocks": args.num_blocks,
                "max_batch_size": args.max_batch_size,
                "max_tokens": args.max_tokens,
                "batch_size": args.batch_size,
                "ttft_num_requests": args.num_requests,
            },
        },
    }

    print("=" * 60)
    print("  mini-vllm benchmark suite")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  KV blocks:   {args.num_blocks} × {cfg.block_size} tokens")
    print(f"  Batching:    {args.batch_size} prompts × {args.max_tokens} max tokens")
    print(f"  TTFT burst:  {args.num_requests} requests")
    print("=" * 60)

    out["memory"] = run_memory()

    if args.skip_gpu or not torch.cuda.is_available():
        if not args.skip_gpu:
            print("\nNo CUDA. Skipping GPU benchmarks.")
    else:
        out["batching"] = run_batching(
            args.model,
            num_blocks=args.num_blocks,
            max_batch_size=args.max_batch_size,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            warmup=args.warmup,
        )
        out["ttft"] = run_ttft(
            args.model,
            num_blocks=args.num_blocks,
            max_tokens=args.max_tokens,
            num_requests=args.num_requests,
        )

    print("\n── Summary ──")
    mem = out["memory"]
    print(f"  Memory: paged serves {mem['serve_ratio']:.2f}× more concurrent seqs")
    if "batching" in out:
        b = out["batching"]
        print(
            f"  Batching: {b['speedup_x']:.2f}× throughput "
            f"({b['sequential']['num_prompts']} × {b['sequential']['max_tokens']} tokens)"
        )
    if "ttft" in out:
        print(f"  TTFT: {out['ttft']['max_ttft_improvement_x']:.2f}× better max TTFT under burst")

    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved → {save}")
    print("Plots → python benchmarks/plot_results.py")


if __name__ == "__main__":
    main()
