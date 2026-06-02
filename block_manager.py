'''
This has physical block as well as block allocator.
'''
from dataclasses import dataclass
from collections import deque

@dataclass
class PhysicalBlock:
    block_id: int 
    ref_count: int = 0

class BlockAllocator:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_list = deque(PhysicalBlock(block_id=i, ref_count=0) for i in range(num_blocks))

    def allocate_block(self):
        if not self.free_list:
            raise MemoryError("No free blocks available")
        else: 
            block = self.free_list.popleft()
            block.ref_count += 1
        return block
    def free_block(self, block):
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_list.append(block)





