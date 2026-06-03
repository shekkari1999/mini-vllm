import math
from collections import deque

from minivllm.engine.sequence import RequestStatus


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
            if seq.status == RequestStatus.FINISHED:
                self.running.remove(seq)

        for seq in list(self.running):
            if seq.get_len() % self.block_allocator.block_size == 0:
                block = self.block_allocator.allocate_block()
                seq.block_table.append(block)

        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting.popleft()
            prompt_length = len(seq.prompt_token_ids)
            num_blocks = math.ceil(prompt_length / self.block_allocator.block_size)
            for _ in range(num_blocks):
                block = self.block_allocator.allocate_block()
                seq.block_table.append(block)
            seq.status = RequestStatus.RUNNING
            self.running.append(seq)
        return self.running
