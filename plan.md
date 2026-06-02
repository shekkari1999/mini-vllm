# mini-vllm — Implementation Plan

Single-GPU, offline inference engine inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [Inside vLLM (Aleksa Gordić)](https://www.aleksagordic.com/blog/vllm).

**Scope (v0):** One GPU · synchronous · offline · `LLM.generate()` API · no HTTP / no tensor parallel.

**Out of scope (until v0.1 ships):** Multi-GPU, async serving, prefix caching, CUDA graphs, speculative/guided decoding.

---

## North-star milestone (v0.1)

- [ ] `LLM(model_path)` loads on `cuda:0`
- [ ] `generate(prompts, SamplingParams)` batches multiple requests with continuous batching
- [ ] Paged KV cache + block allocator
- [ ] Greedy + temperature / top-p sampling
- [ ] `enforce_eager=True` (CUDA graphs deferred)
- [ ] `example.py` runs end-to-end
- [ ] README + benchmark table (TTFT, decode tok/s, throughput)

---

## Architecture map

| vLLM V1 (blog) | mini-vllm module | Phase |
|----------------|------------------|-------|
| `LLM` + config | `llm.py`, `config.py`, `sampling_params.py` | 0 |
| Input processor | Tokenizer in `LLMEngine` | 0–1 |
| `InprocClient` ≈ `EngineCore` | `engine/llm_engine.py` | 1 |
| Scheduler | `engine/scheduler.py` | 2 |
| KV cache manager | `engine/block_manager.py` | 3 |
| `Worker` / `ModelRunner` | `engine/model_runner.py` | 1–4 |
| `InputBatch` + metadata | Buffers in `model_runner` | 4 |
| Model + layers | `models/`, `layers/` | 1 |
| Paged attention | `layers/attention.py` | 4–5 |

---

## Phase 0 — Skeleton & contracts

**Goal:** Repo layout, config knobs, imports work.

### Repo & docs

- [ ] Package layout: `minivllm/` (`config`, `engine`, `layers`, `models`, `sampling_params`, `llm`)
- [ ] `pyproject.toml` or `requirements.txt` (torch, transformers, etc.)
- [ ] `README.md` — scope, install, quick start
- [ ] `example.py` stub (imports only)
- [ ] `docs/architecture.md` — map modules to vLLM blog sections

### Config & API surface

- [ ] `Config`: `model`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`
- [ ] `Config`: `kvcache_block_size` (default 16), `gpu_memory_utilization`, `enforce_eager`
- [ ] `SamplingParams`: `max_tokens`, `temperature`, `top_p`, `top_k`, `ignore_eos`
- [ ] `LLM` facade wrapping `LLMEngine` (nano-vllm-style API)

---

## Phase 1 — Single-sequence baseline (correctness)

**Goal:** Greedy decode matches reference; no paging yet.

### Model & runner

- [ ] Pick one architecture (e.g. Qwen3 / Llama-style — match nano-vllm if comparing)
- [ ] `models/<arch>.py` — load weights, `eval()` mode
- [ ] `layers/`: embed, norm, MLP, lm_head
- [ ] `ModelRunner`: load model on `cuda:0`, dtype check (bf16/fp16)
- [ ] Static per-sequence KV cache (full `max_model_len`)

### Engine loop (single request)

- [ ] `engine/sequence.py` — prompt tokens, completion tokens, `is_finished`
- [ ] `LLMEngine.add_request` — tokenize string → `Sequence`
- [ ] Prefill: forward all prompt tokens, sample one token
- [ ] Decode: one token per step until EOS or `max_tokens`
- [ ] Greedy sampling only (`temperature=0`)
- [ ] Detokenize output; return `[{"text": ..., "token_ids": ...}]`

### Correctness gates

- [ ] Golden test: token IDs match HuggingFace `generate` (greedy) on 3–5 fixed prompts
- [ ] Document model + commit used for golden files

---

## Phase 2 — Continuous batching (simple KV)

**Goal:** Multiple concurrent sequences; scheduler + token budget.

### Scheduler

- [ ] `WAITING` / `RUNNING` / `FINISHED` on `Sequence`
- [ ] FCFS policy
- [ ] Decode-first scheduling (running queue before waiting)
- [ ] Per-step token budget: `max_num_batched_tokens`, `max_num_seqs`
- [ ] Mix prefill + decode in same step when budget allows (V1-style, not V0 either/or)

### Engine `step()` loop

- [ ] `schedule()` → list of seqs + `is_prefill`
- [ ] `model_runner.run(seqs, is_prefill)`
- [ ] `postprocess()` — append tokens, stop checks
- [ ] `generate()` — `while not is_finished(): step()`
- [ ] Progress / throughput display (optional, like nano-vllm tqdm)

### Correctness & perf gates

- [ ] Each request’s token stream matches isolated single-seq run
- [ ] N concurrent seqs faster than N × sequential (rough check)
- [ ] No cross-request attention leakage (manual spot-check)

---

## Phase 3 — Paged KV cache

**Goal:** Block pool; VRAM scales with concurrent work, not `max_len × num_seqs`.

### Block manager

- [ ] `BlockManager`: `free_block_queue`, `req_id → [block_ids]`
- [ ] `allocate_slots(request, num_new_tokens)` — `ceil(tokens / block_size)` blocks
- [ ] `free(request)` on finish — return blocks to pool
- [ ] Block size = `kvcache_block_size` (default 16)

### KV memory

- [ ] Profile VRAM after model load (dummy forward)
- [ ] Compute `num_gpu_blocks` from `gpu_memory_utilization`
- [ ] Per-layer paged K/V tensors (single allocation, block-indexed)
- [ ] Bind KV views to attention layers

### Correctness gates

- [ ] Many short + few long requests without OOM (where static cache would fail)
- [ ] No block leaks after 1000 random-length request cycles
- [ ] Invariant: `len(block_table) * block_size >= seq_len` for all running seqs

---

## Phase 4 — Paged attention forward

**Goal:** Production-shaped forward pass; sampling beyond greedy.

### Model runner (batched forward)

- [ ] Flatten batch into one “super sequence”
- [ ] Per-seq `positions`, `input_ids`, attention metadata
- [ ] `slot_mapping` for paged KV read/write (per blog forward-pass section)
- [ ] CPU-side block tables → GPU each step
- [ ] Prefill path: variable prompt lengths in one batch
- [ ] Decode path: one new token per seq per step

### Attention & sampling

- [ ] Paged attention via block tables (FlashAttention-2 or equivalent first)
- [ ] Gather last-token hidden states → logits
- [ ] Sampler: greedy, temperature, top-p, top-k
- [ ] Optional: Gumbel-max sampling (nano-vllm style)

### Perf gates

- [ ] Log prefill tok/s vs decode tok/s separately
- [ ] Benchmark vs nano-vllm on same GPU/model (ballpark, not day-one parity)
- [ ] Scheduled tokens ≤ `max_num_batched_tokens` every step

---

## Phase 5 — Engine polish (still single GPU)

**Goal:** Stop conditions, eager default, basic overload behavior.

### Stop conditions

- [ ] `max_tokens` / `max_model_len`
- [ ] EOS token (respect `ignore_eos` for benchmarks)
- [ ] `stop_token_ids`
- [ ] Stop strings (truncate output, abort request)

### Robustness

- [ ] On block pool exhaustion: skip scheduling or wait (document policy)
- [ ] Optional: simple preemption (free finished only first; full recompute later)
- [ ] `enforce_eager=True` as default
- [ ] CUDA graph capture — **deferred** (backlog)

### Documentation

- [ ] `example.py` full demo (multi-prompt batch)
- [ ] Known gaps vs vLLM / nano-vllm listed in README

---

## Backlog (post v0.1)

### Advanced engine (Aleksa part 2)

- [ ] Chunked prefill (`long_prefill_token_threshold`)
- [ ] Prefix caching (block hashes, ref counts, cache hit on new requests)
- [ ] `torch.compile` on model
- [ ] Priority scheduling (heap vs FCFS)

### Scale-up (Aleksa part 3) — not single-GPU

- [ ] Tensor parallelism
- [ ] `MultiProcExecutor` / worker processes
- [ ] Pipeline / expert parallelism

### Serving (Aleksa part 4)

- [ ] Async engine — inject requests between steps
- [ ] OpenAI-compatible HTTP API
- [ ] Streaming tokens to client

### Exotic features

- [ ] Speculative decoding
- [ ] Guided decoding (grammar / FSM)
- [ ] Disaggregated prefill/decode

---

## Tracking checklist

### Invariants (test or debug assert)

- [ ] Block table covers sequence length for every running request
- [ ] Freed blocks re-enter `free_block_queue`
- [ ] Finished sequences never rescheduled
- [ ] `sum(scheduled_tokens) <= max_num_batched_tokens` per step

### Metrics to log (`bench.py` or CSV)

- [ ] **TTFT** — time to first token per request
- [ ] **ITL / TPOT** — inter-token latency (decode)
- [ ] **E2E latency** — prompt in → completion out
- [ ] **Throughput** — total generated tokens / wall time
- [ ] **Prefill tok/s** and **decode tok/s** (separate)

### Fixed benchmark config (record in repo)

- [ ] Model name + path
- [ ] GPU model
- [ ] `max_num_seqs`, `max_num_batched_tokens`
- [ ] Input length, output length, batch size
- [ ] Results table in README or `benchmarks/results.md`

### Engineering hygiene

- [ ] Decision log (what we deferred and why)
- [ ] Comparison notes: mini-vllm vs nano-vllm vs vLLM when run

---

## Module implementation order

1. [ ] `config.py`, `sampling_params.py`
2. [ ] `engine/sequence.py`
3. [ ] `models/<arch>.py` + `layers/`
4. [ ] `engine/model_runner.py`
5. [ ] `engine/block_manager.py`
6. [ ] `engine/scheduler.py`
7. [ ] `engine/llm_engine.py` + `llm.py`
8. [ ] `example.py`
9. [ ] `bench.py` + results table

---

## References

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
