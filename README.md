# mini-vllm

Single-GPU LLM inference with paged KV cache, an FCFS scheduler, and continuous batching.

Default model: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). Qwen2 / Qwen2.5 only.

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

llm = LLM()  # defaults below

out = llm.generate(["Hello", "The sky is"], SamplingParams(max_tokens=128))
```

| Config | Default | Meaning |
|--------|---------|---------|
| `model` | Qwen2.5-7B-Instruct | HuggingFace model id |
| `num_blocks` | 512 | KV blocks in the GPU pool |
| `block_size` | 16 | Tokens per block |
| `max_batch_size` | 8 | Max concurrent sequences |

## Benchmarks

Three tracks, each with a naive baseline:

| Track | Baseline | This repo | Default run |
|-------|----------|-----------|-------------|
| Memory | `max_seq_len` reserved per request | Paged block allocator | CPU, 300 mixed-length requests |
| Batching | Sequential decode | Continuous batching | 8 prompts × 128 output tokens |
| TTFT | Burst load, `max_batch_size=1` | All requests batched | 16 requests × 128 max tokens |

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

## Results

Run `benchmarks/run_all.py` on a CUDA GPU and fill this in.

| Benchmark | Metric | Value | GPU | Date |
|-----------|--------|-------|-----|------|
| Memory | paged / naive serve ratio | | | |
| Batching | speedup vs sequential (8 × 128 tok) | | | |
| TTFT | max TTFT improvement (16 req burst) | | | |

## Not implemented yet

- Speculative decoding (Qwen3-0.6B draft + Qwen3-4B target)
- Temperature / top-p sampling
- Prefix caching, chunked prefill
