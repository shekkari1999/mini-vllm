from dataclasses import dataclass, field
from enum import Enum, auto

from minivllm.sampling_params import SamplingParams


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    request_id: int
    sampling_params: SamplingParams
    prompt_token_ids: list
    output_token_ids: list = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    block_table: list = field(default_factory=list)

    def get_len(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def is_finished(self):
        return self.status == RequestStatus.FINISHED

    def get_last_token_id(self):
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]
