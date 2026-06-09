# Resume numbers

Run on a CUDA GPU:

```bash
uv sync
export HF_TOKEN=hf_...
uv run python benchmarks/run_all.py
uv run python benchmarks/plot_results.py
```

Numbers come from `benchmarks/results/latest.json` only.

## Three bullets (fill after GPU run)

1. Built a single-GPU **LLM inference engine** with **paged KV cache** and **PagedAttention** for **Qwen2.5-3B** on CUDA.

2. Implemented **FCFS scheduler** and **continuous batching** with a ref-counted block allocator over a shared KV pool.

3. Benchmarked vs naive baselines: **{serve_ratio}×** more concurrent sequences (paged vs max-length KV), **{speedup_x}×** decode throughput vs sequential, **{max_ttft_improvement_x}×** better max TTFT under burst load.

## Do not claim here

- FlashAttention / Triton speedups → [triton-kernels](https://github.com/shekkari1999/triton-kernels)
- Speculative decoding (not implemented yet)
- Numbers not printed by `run_all.py`
