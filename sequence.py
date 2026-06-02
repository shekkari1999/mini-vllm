from enum import Enum, auto
from dataclasses import dataclass, field

class RequestStatus(Enum):
    '''
    enum is there to enforce naming value. we can use this when there are 
    fixed set of values like days of a week, signs of traffic light. Just 
    so that python would catch if there are any typos
    '''
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    
@dataclass
class SamplingParams:
    temperature: float = 0.8
    max_tokens: int = 128

@dataclass
class Sequence:
    # fields with default value must come after fields without default value
    request_id: int
    sampling_params: SamplingParams
    prompt_token_ids: list 
    output_token_ids: list = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    block_table: list = field(default_factory=list)

    def get_len(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)
    
   
