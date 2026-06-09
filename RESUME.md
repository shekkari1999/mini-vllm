# Resume numbers

Run on a CUDA GPU (H100 or A100 80GB recommended for 7B):

```bash
uv sync
export HF_TOKEN=hf_...
uv run python benchmarks/run_all.py --warmup
uv run python benchmarks/plot_results.py
```

Defaults: **Qwen2.5-7B**, 8 concurrent requests, **128** output tokens, 16-request TTFT burst.

Numbers come from `benchmarks/results/latest.json` only.

## Three bullets (fill after GPU run)

1. Built a single-GPU **LLM inference engine** with **paged KV cache** and **PagedAttention** for **Qwen2.5-7B** on CUDA.

2. Implemented **FCFS scheduler** and **continuous batching** with a ref-counted block allocator over a shared KV pool (512 blocks × 16 tokens).

3. Benchmarked vs naive baselines: **{serve_ratio}×** more concurrent sequences (paged vs max-length KV), **{speedup_x}×** decode throughput vs sequential (8 × 128 tokens), **{max_ttft_improvement_x}×** better max TTFT under 16-request burst.

## Do not claim here

- FlashAttention / Triton speedups → [triton-kernels](https://github.com/shekkari1999/triton-kernels)
- Speculative decoding (not implemented yet)
- Numbers not printed by `run_all.py`
