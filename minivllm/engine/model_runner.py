import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minivllm.layers.attention import PagedAttention


class ModelRunner:
    def __init__(self, model_path, num_blocks, block_size):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        config = self.model.config
        if not getattr(config, "model_type", "").startswith("qwen"):
            raise ValueError(
                f"Only Qwen models are supported (got model_type={config.model_type!r})"
            )
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
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
        raise ValueError("Expected Qwen model with model.layers decoder stack")

    def _patch_attention_layers(self):
        for layer_idx, layer in enumerate(self._get_decoder_layers()):
            layer.self_attn = PagedAttention(
                layer.self_attn,
                self.kv_cache,
                self.block_size,
                layer_idx,
                self.block_table_holder,
            )

    def _forward(self, input_ids, position_ids, block_tables, seq_lens=None):
        self.block_table_holder[0] = {
            "block_tables": block_tables,
            "seq_lens": seq_lens,
        }
        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            )

    def run_prefill(self, seqs):
        if not seqs:
            return torch.empty(0, device="cuda")

        lengths = [len(s.prompt_token_ids) for s in seqs]
        max_len = max(lengths)
        batch_size = len(seqs)

        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long, device="cuda"
        )
        position_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device="cuda")
        for b, seq in enumerate(seqs):
            n = lengths[b]
            input_ids[b, :n] = torch.tensor(seq.prompt_token_ids, device="cuda")
            position_ids[b, :n] = torch.arange(n, device="cuda")

        outputs = self._forward(
            input_ids,
            position_ids,
            [s.block_table for s in seqs],
            seq_lens=lengths,
        )
        return torch.stack([outputs.logits[b, lengths[b] - 1, :] for b in range(batch_size)])

    def run_decode(self, seqs):
        if not seqs:
            return torch.empty(0, device="cuda")

        input_ids = torch.tensor(
            [[s.get_last_token_id()] for s in seqs], dtype=torch.long, device="cuda"
        )
        position_ids = torch.tensor(
            [[s.get_len() - 1] for s in seqs], dtype=torch.long, device="cuda"
        )
        outputs = self._forward(
            input_ids,
            position_ids,
            [s.block_table for s in seqs],
        )
        return outputs.logits[:, -1, :]
