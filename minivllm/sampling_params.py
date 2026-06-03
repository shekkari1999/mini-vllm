from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 128
