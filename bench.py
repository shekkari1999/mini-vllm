"""Benchmark mini-vllm inference on a CUDA GPU."""

import argparse
import time

import torch

from minivllm import Config, SamplingParams
from minivllm.engine.llm_engine import LLMEngine


DEFAULT_PROMPTS = [
    "The capital of France is",
    "In machine learning, a tensor is",
    "Paged attention improves GPU memory usage by",
    "The speed of light in a vacuum is",
]


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarking. No GPU detected.")


def _active_seqs(engine: LLMEngine):
    return list(engine.scheduler.waiting) + list(engine.scheduler.running)


def bench_single_request(engine: LLMEngine, prompt: str, max_tokens: int) -> dict:
    """Measure TTFT, decode throughput, and end-to-end latency for one request."""
    sampling_params = SamplingParams(max_tokens=max_tokens)
    engine.add_request(prompt, sampling_params)

    t_start = time.perf_counter()
    t_prefill = time.perf_counter()
    engine.step()
    ttft = time.perf_counter() - t_prefill

    decode_times = []
    output_tokens = 1
    while engine.scheduler.running:
        t_decode = time.perf_counter()
        engine.step()
        decode_times.append(time.perf_counter() - t_decode)
        output_tokens += 1

    e2e = time.perf_counter() - t_start
    decode_tokens = max(output_tokens - 1, 0)
    decode_tps = decode_tokens / sum(decode_times) if decode_times else 0.0
    prompt_tokens = len(engine.model_runner.tokenizer.encode(prompt))

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_s": ttft,
        "prefill_tps": prompt_tokens / ttft if ttft > 0 else 0.0,
        "decode_tps": decode_tps,
        "e2e_s": e2e,
        "throughput_tps": output_tokens / e2e if e2e > 0 else 0.0,
    }


def bench_batch(engine: LLMEngine, prompts: list[str], max_tokens: int) -> dict:
    """Measure batched end-to-end throughput."""
    sampling_params = SamplingParams(max_tokens=max_tokens)
    for prompt in prompts:
        engine.add_request(prompt, sampling_params)

    t_start = time.perf_counter()
    output_tokens = 0
    while engine.scheduler.waiting or engine.scheduler.running:
        before = {seq.request_id: len(seq.output_token_ids) for seq in _active_seqs(engine)}
        engine.step()
        for seq in _active_seqs(engine):
            output_tokens += len(seq.output_token_ids) - before.get(seq.request_id, 0)

    elapsed = time.perf_counter() - t_start
    return {
        "num_prompts": len(prompts),
        "elapsed_s": elapsed,
        "output_tokens": output_tokens,
        "throughput_tps": output_tokens / elapsed if elapsed > 0 else 0.0,
    }


def print_single_metrics(metrics: dict):
    print("\n--- Single request ---")
    print(f"  Prompt tokens:   {metrics['prompt_tokens']}")
    print(f"  Output tokens:   {metrics['output_tokens']}")
    print(f"  TTFT:            {metrics['ttft_s']:.3f} s")
    print(f"  Prefill tok/s:   {metrics['prefill_tps']:.1f}")
    print(f"  Decode tok/s:    {metrics['decode_tps']:.1f}")
    print(f"  E2E latency:     {metrics['e2e_s']:.3f} s")
    print(f"  E2E throughput:  {metrics['throughput_tps']:.1f} tok/s")


def print_batch_metrics(metrics: dict):
    print("\n--- Batch ---")
    print(f"  Prompts:         {metrics['num_prompts']}")
    print(f"  Output tokens:   {metrics['output_tokens']}")
    print(f"  Elapsed:         {metrics['elapsed_s']:.3f} s")
    print(f"  Throughput:      {metrics['throughput_tps']:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mini-vllm")
    parser.add_argument("--model", default=Config().model, help="HuggingFace model id")
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Number of prompts in batch benchmark"
    )
    parser.add_argument("--warmup", action="store_true", help="Run a short warmup before timing")
    args = parser.parse_args()

    require_cuda()

    config = Config(
        model=args.model,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch_size=args.max_batch_size,
    )

    print(f"Model:           {config.model}")
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Blocks:          {config.num_blocks} x {config.block_size} tokens")
    print(f"Max batch size:  {config.max_batch_size}")

    engine = LLMEngine(
        config.model, config.num_blocks, config.block_size, config.max_batch_size
    )

    if args.warmup:
        print("\nWarmup...")
        engine.generate("Hello", SamplingParams(max_tokens=4))
        engine = LLMEngine(
            config.model, config.num_blocks, config.block_size, config.max_batch_size
        )
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    single_metrics = bench_single_request(
        engine, DEFAULT_PROMPTS[0], args.max_tokens
    )
    torch.cuda.synchronize()
    print_single_metrics(single_metrics)

    batch_prompts = DEFAULT_PROMPTS[: args.batch_size]
    torch.cuda.synchronize()
    batch_metrics = bench_batch(engine, batch_prompts, args.max_tokens)
    torch.cuda.synchronize()
    print_batch_metrics(batch_metrics)


if __name__ == "__main__":
    main()
