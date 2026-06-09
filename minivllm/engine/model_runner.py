import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minivllm.layers.attention import PagedAttention


class ModelRunner:
    def __init__(self, model_path, num_blocks, block_size):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        config = self.model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.block_size = block_size
        self.block_table_holder = [None]
        self.kv_cache = torch.zeros(
            2,
            num_layers,
            num_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device="cuda",
        )
        self._patch_attention_layers()

    def _get_decoder_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise ValueError("Unsupported model architecture for paged attention")

    def _patch_attention_layers(self):
        for layer_idx, layer in enumerate(self._get_decoder_layers()):
            layer.self_attn = PagedAttention(
                layer.self_attn,
                self.kv_cache,
                self.block_size,
                layer_idx,
                self.block_table_holder,
            )

    def _forward(self, input_ids, position_ids, block_table):
        self.block_table_holder[0] = block_table
        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )

    def run_prefill(self, seqs):
        logits = []
        for seq in seqs:
            input_ids = torch.tensor([seq.prompt_token_ids], device="cuda")
            position_ids = torch.tensor(
                [list(range(len(seq.prompt_token_ids)))], device="cuda"
            )
            outputs = self._forward(input_ids, position_ids, seq.block_table)
            logits.append(outputs.logits[0, -1, :])
        return torch.stack(logits)

    def run_decode(self, seqs):
        logits = []
        for seq in seqs:
            input_ids = torch.tensor([[seq.get_last_token_id()]], device="cuda")
            position_ids = torch.tensor([[seq.get_len() - 1]], device="cuda")
            outputs = self._forward(input_ids, position_ids, seq.block_table)
            logits.append(outputs.logits[0, -1, :])
        return torch.stack(logits)
