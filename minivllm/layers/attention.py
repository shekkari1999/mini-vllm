import torch
import torch.nn as nn


def _apply_rotary_pos_emb(q, k, cos, sin):
    for path in (
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.llama.modeling_llama",
    ):
        try:
            mod = __import__(path, fromlist=["apply_rotary_pos_emb"])
            return mod.apply_rotary_pos_emb(q, k, cos, sin)
        except ImportError:
            continue
    raise ImportError("apply_rotary_pos_emb not found")


class PagedAttention(nn.Module):
    def __init__(self, attn_module, kv_cache, block_size, layer_idx, block_table_holder):
        super().__init__()
        self.q_proj = attn_module.q_proj
        self.k_proj = attn_module.k_proj
        self.v_proj = attn_module.v_proj
        self.o_proj = attn_module.o_proj
        self.kv_cache = kv_cache
        self.block_size = block_size
        self.layer_idx = layer_idx
        self.block_table_holder = block_table_holder
        self.num_heads = attn_module.num_heads
        self.num_kv_heads = attn_module.num_key_value_heads
        self.head_dim = attn_module.head_dim
        self.scaling = getattr(attn_module, "scaling", self.head_dim**-0.5)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        block_table = self.block_table_holder[0]
        if block_table is None:
            raise RuntimeError("block_table must be set before forward")
        if position_embeddings is None:
            raise RuntimeError("position_embeddings must be provided for RoPE")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        Q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        K = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        V = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        Q, K = _apply_rotary_pos_emb(Q, K, cos, sin)

        _, _, seq_len, _ = Q.shape

        for i in range(seq_len):
            pos = position_ids[0, i].item()
            block_idx = pos // self.block_size
            slot = pos % self.block_size
            block_id = block_table[block_idx].block_id
            self.kv_cache[0, self.layer_idx, block_id, slot] = K[0, :, i, :]
            self.kv_cache[1, self.layer_idx, block_id, slot] = V[0, :, i, :]

        keys = []
        values = []
        for block in block_table:
            keys.append(self.kv_cache[0, self.layer_idx, block.block_id])
            values.append(self.kv_cache[1, self.layer_idx, block.block_id])
        K_cache = torch.cat(keys, dim=0).unsqueeze(0).transpose(1, 2)
        V_cache = torch.cat(values, dim=0).unsqueeze(0).transpose(1, 2)

        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            K_cache = K_cache.repeat_interleave(n_rep, dim=1)
            V_cache = V_cache.repeat_interleave(n_rep, dim=1)

        scores = torch.matmul(Q, K_cache.transpose(-2, -1)) * self.scaling

        total_len = K_cache.shape[-2]
        key_idx = torch.arange(total_len, device=scores.device)
        causal_mask = key_idx.unsqueeze(0) > position_ids[0].unsqueeze(1)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V_cache)
        out = out.transpose(1, 2).contiguous().view(*input_shape, -1)
        out = self.o_proj(out)
        return out, None
