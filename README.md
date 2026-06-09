# mini-vllm

A small, readable LLM inference engine for one GPU. The goal is to make vLLM-style serving tangible: **paged KV cache**, **continuous batching**, and a simple **FCFS scheduler**, without hiding the wiring behind a big framework.

Ships with [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) by default (Qwen2 / Qwen2.5 only). For fused GPU kernels (FlashAttention, etc.), see [triton-kernels](https://github.com/shekkari1999/triton-kernels).

## How it works

You pass prompts; the engine turns each into a `Sequence` and tracks it through prefill and decode. Instead of reserving `max_seq_len` KV slots per request, a scheduler hands out fixed-size blocks from a shared GPU pool. HuggingFace runs the model, with `PagedAttention` patched in place of the stock attention layers. Each step is one batched forward (`model([B, seq])`: right-padded prefill, then one token per sequence on decode). Sampling is greedy (`argmax`) for now.

## Layout

```
minivllm/
  config.py
  llm.py
  sampling_params.py
  engine/       scheduler, block allocator, model runner, generation loop
  layers/       PagedAttention
benchmarks/
  run_all.py
  plot_results.py
  benchmark_memory.py
  benchmark_batching.py
  benchmark_ttft.py
bench.py          shortcut for batching benchmark only
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

llm = LLM()  # defaults below

out = llm.generate(["Hello", "The sky is"], SamplingParams(max_tokens=128))
```

| Config | Default | Meaning |
|--------|---------|---------|
| `model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model id |
| `num_blocks` | 512 | KV blocks in the GPU pool |
| `block_size` | 16 | Tokens per block (8,192 token pool) |
| `max_batch_size` | 8 | Max concurrent sequences |

## Benchmarks

Three tracks, each with a naive baseline:

| Track | Baseline | This repo | Default run |
|-------|----------|-----------|-------------|
| Memory | `max_seq_len` reserved per request | Paged block allocator | CPU sim, 32K token budget, 300 mixed-length requests |
| Batching | Sequential decode (one request at a time) | Continuous batching + batched forwards | 8 prompts × 128 output tokens |
| TTFT | Serial queue (`max_batch_size=1`) | All requests scheduled together | 16 requests × 128 max tokens |

```bash
uv run python benchmarks/run_all.py --warmup
uv run python benchmarks/plot_results.py
```

Output: `benchmarks/results/latest.json`, `benchmarks/results/figures/`

Override defaults:

```bash
uv run python benchmarks/run_all.py --warmup \
  --max-tokens 256 \
  --batch-size 8 \
  --num-requests 16
```

Batching only (no full suite):

```bash
uv run python bench.py --warmup
```

## Results

`Qwen/Qwen2.5-7B-Instruct`, engine config `512` blocks × `16` tokens, `max_batch_size=8`. Source: `benchmarks/results/latest.json`.

| Benchmark | Metric | Value | GPU | Date |
|-----------|--------|-------|-----|------|
| Memory | paged / naive serve ratio | **3.53×** (64 → 226 seqs) | CPU sim | 2026-06 |
| Batching | speedup vs sequential (8 × 128 tok) | **2.61×** (53.6 → 139.8 tok/s) | H100 80GB | 2026-06 |
| TTFT | max TTFT (16 req burst) | **35.5s → 0.22s** (160×) | H100 80GB | 2026-06 |

Figures: `benchmarks/results/figures/` (`memory_capacity.png`, `batching_throughput.png`, `ttft_burst.png`).

## Future scope

- Speculative decoding (Qwen3-0.6B draft + Qwen3-4B target)
- Temperature / top-p sampling
- Prefix caching, chunked prefill
