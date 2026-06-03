from dataclasses import dataclass

DEFAULT_MODEL = "meta-llama/Llama-2-7b-hf"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    num_blocks: int = 128
    block_size: int = 16
    max_batch_size: int = 4
