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
from minivllm import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("benchmarks/results/latest.json"),
    )
    parser.add_argument("--model", default=Config().model)
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
        },
    }

    print("=" * 60)
    print("  mini-vllm benchmark suite")
    print("=" * 60)

    out["memory"] = run_memory()

    if args.skip_gpu or not torch.cuda.is_available():
        if not args.skip_gpu:
            print("\nNo CUDA. Skipping GPU benchmarks.")
    else:
        out["batching"] = run_batching(args.model, warmup=args.warmup)
        out["ttft"] = run_ttft(args.model)

    print("\n── Summary ──")
    mem = out["memory"]
    print(f"  Memory: paged serves {mem['serve_ratio']:.2f}× more concurrent seqs")
    if "batching" in out:
        print(f"  Batching: {out['batching']['speedup_x']:.2f}× throughput vs sequential")
    if "ttft" in out:
        print(f"  TTFT: {out['ttft']['max_ttft_improvement_x']:.2f}× better max TTFT under burst")

    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nSaved → {save}")
    print("Plots → python benchmarks/plot_results.py")


if __name__ == "__main__":
    main()
