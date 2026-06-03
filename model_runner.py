## has 2 methods
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class ModelRunner:
    def __init__(self, model_path, num_blocks, block_size):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16).to('cuda')
        config = self.model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.block_size = block_size
        self.kv_cache = torch.zeros(
            num_layers, num_blocks, self.block_size, num_heads, head_dim, dtype = torch.float16, device = 'cuda'
        )
    
    def run_prefill(self, seqs):
        input_ids = []
        position_ids = []
        slot_mapping = []
        for seq in seqs:
            input_ids.extend(seq.prompt_token_ids)
            position_ids.extend(range(len(seq.prompt_token_ids)))
            for i in range(len(seq.prompt_token_ids)):
                block_id = seq.block_table[i // self.block_size]
                slot = i % self.block_size
                physical_slot = block_id * self.block_size + slot
                slot_mapping.append(physical_slot)
        input_ids = torch.tensor(input_ids, device = 'cuda')
        position_ids = torch.tensor(position_ids, device = 'cuda')
        slot_mapping = torch.tensor(slot_mapping, device = 'cuda')
            


            



