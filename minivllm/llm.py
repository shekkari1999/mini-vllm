from minivllm.config import Config
from minivllm.engine.llm_engine import LLMEngine
from minivllm.sampling_params import SamplingParams


class LLM:
    def __init__(self, model: str | None = None, config: Config | None = None):
        config = config or Config()
        if model is not None:
            config = Config(
                model=model,
                num_blocks=config.num_blocks,
                block_size=config.block_size,
                max_batch_size=config.max_batch_size,
            )
        self.config = config
        self.engine = LLMEngine(
            config.model,
            config.num_blocks,
            config.block_size,
            config.max_batch_size,
        )

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> dict[int, str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        sampling_params = sampling_params or SamplingParams()
        return self.engine.generate(prompts, sampling_params)
