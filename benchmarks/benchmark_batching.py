"""Sequential vs continuous batching throughput (GPU)."""

from __future__ import annotations

import time

import torch

from minivllm import Config
from minivllm.engine.llm_engine import LLMEngine
from minivllm.sampling_params import SamplingParams

DEFAULT_PROMPTS = [
    "The capital of France is",
    "In machine learning, a tensor is",
    "Paged attention improves GPU memory usage by",
    "The speed of light in a vacuum is",
]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _active(engine: LLMEngine):
    return list(engine.scheduler.waiting) + list(engine.scheduler.running)


def _output_delta(engine: LLMEngine, before: dict[int, int]) -> int:
    total = 0
    for seq in _active(engine):
        total += len(seq.output_token_ids) - before.get(seq.request_id, 0)
    return total


def bench_throughput(
    engine: LLMEngine,
    prompts: list[str],
    max_tokens: int,
    *,
    sequential: bool,
) -> dict:
    sp = SamplingParams(max_tokens=max_tokens)
    engine.reset()
    _sync()
    t0 = time.perf_counter()
    output_tokens = 0

    if sequential:
        for prompt in prompts:
            engine.add_request(prompt, sp)
            while engine.scheduler.waiting or engine.scheduler.running:
                before = {s.request_id: len(s.output_token_ids) for s in _active(engine)}
                engine.step()
                output_tokens += _output_delta(engine, before)
            engine.reset()
    else:
        for prompt in prompts:
            engine.add_request(prompt, sp)
        while engine.scheduler.waiting or engine.scheduler.running:
            before = {s.request_id: len(s.output_token_ids) for s in _active(engine)}
            engine.step()
            output_tokens += _output_delta(engine, before)

    elapsed = time.perf_counter() - t0
    _sync()
    return {
        "mode": "sequential" if sequential else "continuous",
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "throughput_tps": output_tokens / elapsed if elapsed > 0 else 0.0,
    }


def compare(
    engine: LLMEngine,
    prompts: list[str],
    max_tokens: int,
) -> dict:
    seq = bench_throughput(engine, prompts, max_tokens, sequential=True)
    cont = bench_throughput(engine, prompts, max_tokens, sequential=False)
    speedup = (
        cont["throughput_tps"] / seq["throughput_tps"] if seq["throughput_tps"] > 0 else 0.0
    )
    return {"sequential": seq, "continuous": cont, "speedup_x": speedup}


def run(
    model: str | None = None,
    *,
    num_blocks: int = 128,
    block_size: int = 16,
    max_batch_size: int = 4,
    max_tokens: int = 64,
    batch_size: int = 4,
    warmup: bool = False,
) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for batching benchmark.")

    cfg = Config()
    model = model or cfg.model
    prompts = DEFAULT_PROMPTS[:batch_size]

    print("─" * 60)
    print("  Continuous batching vs sequential decode")
    print("─" * 60)
    print(f"  Model: {model}")
    print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    print(f"  Batch: {len(prompts)} prompts × {max_tokens} max tokens")
    print()

    engine = LLMEngine(model, num_blocks, block_size, max_batch_size)
    if warmup:
        engine.generate("Hello", SamplingParams(max_tokens=4))
        engine = LLMEngine(model, num_blocks, block_size, max_batch_size)

    result = compare(engine, prompts, max_tokens)
    seq, cont = result["sequential"], result["continuous"]
    print(f"  Sequential:   {seq['throughput_tps']:.1f} tok/s  ({seq['elapsed_s']:.2f} s)")
    print(f"  Continuous:   {cont['throughput_tps']:.1f} tok/s  ({cont['elapsed_s']:.2f} s)")
    print(f"  Speedup:      {result['speedup_x']:.2f}×")
    print("─" * 60)

    return {
        "model": model,
        "gpu": torch.cuda.get_device_name(0),
        "config": {
            "num_blocks": num_blocks,
            "block_size": block_size,
            "max_batch_size": max_batch_size,
        },
        **result,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=Config().model)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--warmup", action="store_true")
    args = p.parse_args()
    run(args.model, max_tokens=args.max_tokens, batch_size=args.batch_size, warmup=args.warmup)
