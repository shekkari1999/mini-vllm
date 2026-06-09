from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Benchmark defaults (H100 / A100 80GB, decode-heavy runs)
BENCHMARK_MAX_TOKENS = 128
BENCHMARK_BATCH_SIZE = 8
BENCHMARK_TTFT_REQUESTS = 16


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    num_blocks: int = 512
    block_size: int = 16
    max_batch_size: int = 8
