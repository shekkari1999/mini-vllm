from dataclasses import dataclass

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    num_blocks: int = 256
    block_size: int = 16
    max_batch_size: int = 8
