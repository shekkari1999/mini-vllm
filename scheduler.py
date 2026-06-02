from collections import deque
from block_manager import BlockAllocator
from sequence import RequestStatus, Sequence, SamplingParams
import math

class Scheduler:
    def __init__(self, max_batch_size, block_allocator):
        self.max_batch_size = max_batch_size
        self.waiting = deque()
        self.running = list()
        self.block_allocator = block_allocator
    
    def add_request(self, seq):
        self.waiting.append(seq)
        
    def schedule(self):
        for seq in list(self.running):
            if seq.get_len() % self.block_allocator.block_size == 0:
                block = self.block_allocator.allocate_block()
                seq.block_table.append(block.block_id)

        for seq in list(self.running):
            if seq.status == RequestStatus.FINISHED:
                self.running.remove(seq)
        while(self.waiting) and len(self.running) < self.max_batch_size:
                seq = self.waiting.popleft()
                prompt_length = len(seq.prompt_token_ids)
                num_blocks = math.ceil(prompt_length/self.block_allocator.block_size)
                for _ in range(num_blocks):
                    block = self.block_allocator.allocate_block()
                    seq.block_table.append(block.block_id)
                seq.status = RequestStatus.RUNNING
                self.running.append(seq)
        return self.running


## create a scheduler object

blk = BlockAllocator(10, 2)
sch = Scheduler(3, blk)
sp = SamplingParams()
seq1 = Sequence(request_id=1, sampling_params=sp, prompt_token_ids=[1, 2, 3, 4, 5])
seq2 = Sequence(request_id=2, sampling_params=sp, prompt_token_ids=[6, 7, 8, 9, 10])
sch.add_request(seq1)
sch.add_request(seq2)
running = sch.schedule()
print(len(running))

for seq in running:
    print(seq)






            




