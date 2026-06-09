from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    num_blocks: int = 128
    block_size: int = 16
    max_batch_size: int = 4
