"""TTFT under burst: serial queue (batch=1) vs continuous batching."""

from __future__ import annotations

import statistics
import time

import torch

from minivllm.config import BENCHMARK_MAX_TOKENS, BENCHMARK_TTFT_REQUESTS, Config
from minivllm.engine.llm_engine import LLMEngine
from minivllm.sampling_params import SamplingParams

DEFAULT_PROMPTS = [
    "Explain gradient descent in one sentence.",
    "What is the capital of France?",
    "Summarize the theory of relativity briefly.",
    "Write a haiku about machine learning.",
    "Name three uses of reinforcement learning.",
    "What is attention in transformer models?",
    "How does paged memory work in operating systems?",
    "Describe backpropagation in three sentences.",
    "What is KV cache in large language models?",
    "Explain autoregressive text generation.",
    "How does RoPE positional encoding work?",
    "What causes OOM errors during LLM inference?",
    "Describe the prefill and decode phases.",
    "Why is batching important for GPU utilization?",
    "What is the difference between throughput and latency?",
    "How do block tables map to physical KV memory?",
]


def _stats(values: list[float]) -> dict:
    v = sorted(values)
    n = len(v)
    return {
        "mean": statistics.mean(v),
        "p50": v[n // 2],
        "p90": v[int(n * 0.9)],
        "max": v[-1],
    }


def _measure_burst(
    engine: LLMEngine,
    prompts: list[str],
    max_tokens: int,
    max_batch_size: int,
) -> list[float]:
    """All prompts arrive at t=0; TTFT measured from that moment."""
    engine.max_batch_size = max_batch_size
    engine.scheduler.max_batch_size = max_batch_size
    engine.reset()

    sp = SamplingParams(max_tokens=max_tokens)
    t0 = time.perf_counter()
    rids = []
    for prompt in prompts:
        engine.add_request(prompt, sp)
        rids.append(engine.seq_counter - 1)

    ttfts: dict[int, float | None] = {rid: None for rid in rids}
    while engine.scheduler.waiting or engine.scheduler.running:
        engine.step()
        for seq in engine.scheduler.running:
            rid = seq.request_id
            if rid in ttfts and ttfts[rid] is None and seq.output_token_ids:
                ttfts[rid] = time.perf_counter() - t0

    return [ttfts[rid] or float("nan") for rid in rids]


def run(
    model: str | None = None,
    *,
    num_blocks: int | None = None,
    block_size: int | None = None,
    max_tokens: int | None = None,
    num_requests: int | None = None,
) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for TTFT benchmark.")

    cfg = Config()
    model = model or cfg.model
    num_blocks = num_blocks if num_blocks is not None else cfg.num_blocks
    block_size = block_size if block_size is not None else cfg.block_size
    max_tokens = max_tokens if max_tokens is not None else BENCHMARK_MAX_TOKENS
    num_requests = num_requests if num_requests is not None else BENCHMARK_TTFT_REQUESTS
    prompts = DEFAULT_PROMPTS[:num_requests]

    print("─" * 60)
    print("  TTFT under burst load")
    print("─" * 60)
    print(f"  Model: {model}   N={len(prompts)}   max_tokens={max_tokens}")
    print()

    engine = LLMEngine(model, num_blocks, block_size, max_batch_size=1)
    static_ttfts = _measure_burst(engine, prompts, max_tokens, max_batch_size=1)
    ss = _stats(static_ttfts)
    print(f"  Serial queue (batch=1):  p50={ss['p50']:.3f}s  p90={ss['p90']:.3f}s  max={ss['max']:.3f}s")

    engine = LLMEngine(model, num_blocks, block_size, max_batch_size=len(prompts))
    cont_ttfts = _measure_burst(engine, prompts, max_tokens, max_batch_size=len(prompts))
    sc = _stats(cont_ttfts)
    print(f"  Continuous (batch={len(prompts)}): p50={sc['p50']:.3f}s  p90={sc['p90']:.3f}s  max={sc['max']:.3f}s")

    max_ratio = ss["max"] / sc["max"] if sc["max"] > 0 else 0.0
    print(f"  Max TTFT improvement: {max_ratio:.2f}×")
    print("─" * 60)

    return {
        "model": model,
        "gpu": torch.cuda.get_device_name(0),
        "num_requests": len(prompts),
        "max_tokens": max_tokens,
        "static_ttfts_s": static_ttfts,
        "continuous_ttfts_s": cont_ttfts,
        "static": ss,
        "continuous": sc,
        "max_ttft_improvement_x": max_ratio,
    }


if __name__ == "__main__":
    import argparse

    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=cfg.model)
    p.add_argument("--num-blocks", type=int, default=cfg.num_blocks)
    p.add_argument("--num-requests", type=int, default=BENCHMARK_TTFT_REQUESTS)
    p.add_argument("--max-tokens", type=int, default=BENCHMARK_MAX_TOKENS)
    args = p.parse_args()
    run(
        args.model,
        num_blocks=args.num_blocks,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
    )
