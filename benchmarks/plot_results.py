"""Plot benchmarks/results/latest.json → benchmarks/results/figures/*.png"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "benchmarks" / "results" / "figures"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"No results at {path}. Run: python benchmarks/run_all.py")
    return json.loads(path.read_text())


def plot_memory(data: dict, out_dir: Path):
    mem = data.get("memory")
    if not mem:
        return
    labels = ["Naive (max_seq_len)", f"Paged (block={mem['block_size']})"]
    served = [mem["naive"]["served"], mem["paged"]["served"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, served, color=["#C44E52", "#55A868"])
    ax.set_ylabel("Concurrent sequences served")
    ax.set_title("KV memory: naive vs paged allocation")
    for i, v in enumerate(served):
        ax.text(i, v + max(served) * 0.02, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    path = out_dir / "memory_capacity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_batching(data: dict, out_dir: Path):
    b = data.get("batching")
    if not b:
        return
    labels = ["Sequential", "Continuous"]
    tps = [b["sequential"]["throughput_tps"], b["continuous"]["throughput_tps"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, tps, color=["#8172B2", "#4C72B0"])
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(f"Decode throughput: {data['metadata'].get('model', '')}")
    for i, v in enumerate(tps):
        ax.text(i, v + max(tps) * 0.02, f"{v:.0f}", ha="center", fontsize=10)
    fig.tight_layout()
    path = out_dir / "batching_throughput.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_ttft(data: dict, out_dir: Path):
    t = data.get("ttft")
    if not t:
        return
    n = t["num_requests"]
    x = list(range(n))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, t["static_ttfts_s"], "o-", label="Serial queue (batch=1)", color="#C44E52")
    ax.plot(x, t["continuous_ttfts_s"], "s-", label="Continuous batching", color="#55A868")
    ax.set_xlabel("Request index")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Time to first token under burst load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "ttft_burst.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmarks/results/latest.json"
    if not json_path.is_absolute():
        json_path = ROOT / json_path

    data = _load(json_path)
    out_dir = FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_memory(data, out_dir)
    plot_batching(data, out_dir)
    plot_ttft(data, out_dir)


if __name__ == "__main__":
    main()
