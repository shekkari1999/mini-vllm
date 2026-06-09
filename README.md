# mini-vllm

Single-GPU LLM inference with paged KV cache, an FCFS scheduler, and continuous batching.

Default model: [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

Triton kernels (fused attention, etc.): [triton-kernels](https://github.com/shekkari1999/triton-kernels)

## How it works

Prompts become `Sequence` objects. The scheduler hands out fixed-size KV blocks from a GPU pool. Prefill and decode run through HuggingFace with `PagedAttention` patched in place of the stock attention layers. Sampling is greedy (`argmax`) for now.

## Layout

```
minivllm/
  config.py
  llm.py
  engine/       scheduler, block allocator, model runner, generation loop
  layers/       PagedAttention
benchmarks/
  run_all.py
  plot_results.py
  benchmark_memory.py
  benchmark_batching.py
  benchmark_ttft.py
```

## Quick start

```bash
uv sync
export HF_TOKEN=hf_...   # or: huggingface-cli login

uv run python -c "
from minivllm import LLM, SamplingParams
llm = LLM()
print(llm.generate('The capital of France is', SamplingParams(max_tokens=32)))
"
```

## API

```python
from minivllm import LLM, Config, SamplingParams

llm = LLM(config=Config(
    model="Qwen/Qwen2.5-3B-Instruct",
    num_blocks=128,
    block_size=16,
    max_batch_size=4,
))

out = llm.generate(["Hello", "The sky is"], SamplingParams(max_tokens=64))
```

| Config | Default | Meaning |
|--------|---------|---------|
| `model` | Qwen2.5-3B-Instruct | HuggingFace model id |
| `num_blocks` | 128 | KV blocks in the GPU pool |
| `block_size` | 16 | Tokens per block |
| `max_batch_size` | 4 | Max concurrent sequences |

## Benchmarks

Three tracks, each with a naive baseline:

| Track | Baseline | This repo | Metric |
|-------|----------|-----------|--------|
| Memory | `max_seq_len` reserved per request | Paged block allocator | Concurrent seqs under a fixed KV budget (CPU) |
| Batching | One request decoded at a time | Continuous batching | tok/s (GPU) |
| TTFT | Burst load, `max_batch_size=1` | All requests batched | p50 / p90 / max time to first token (GPU) |

```bash
uv run python benchmarks/run_all.py
uv run python benchmarks/plot_results.py
uv run python benchmarks/benchmark_memory.py   # CPU only
```

Output: `benchmarks/results/latest.json`, `benchmarks/results/figures/`

## Results

Run `benchmarks/run_all.py` on a CUDA GPU and fill this in.

| Benchmark | Metric | Value | GPU | Date |
|-----------|--------|-------|-----|------|
| Memory | paged / naive serve ratio | | | |
| Batching | speedup vs sequential | | | |
| TTFT | max TTFT improvement | | | |

## Not implemented yet

- Speculative decoding (Qwen3-0.6B draft + Qwen3-4B target)
- Temperature / top-p sampling
- Prefix caching, chunked prefill
