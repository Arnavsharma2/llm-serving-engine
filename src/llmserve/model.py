from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from llmserve.config import ModelConfig
from llmserve.kv_cache import CacheSlot, PagedKVCache
from llmserve.triton_kernels import paged_attention_decode


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().pow(2).mean(-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, max_positions: int) -> None:
        super().__init__()
        inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_positions).float()
        frequencies = torch.outer(positions, inverse)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # values: [batch, sequence, heads, head_dim]
        cos = self.cos[positions].unsqueeze(-2).to(values.dtype)
        sin = self.sin[positions].unsqueeze(-2).to(values.dtype)
        even, odd = values[..., 0::2], values[..., 1::2]
        output = torch.empty_like(values)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = odd * cos + even * sin
        return output


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )

    def _project(self, inputs: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, sequence, _ = inputs.shape
        queries = self.q_proj(inputs).view(
            batch, sequence, self.config.num_attention_heads, self.config.head_dim
        )
        keys = self.k_proj(inputs).view(
            batch, sequence, self.config.num_key_value_heads, self.config.head_dim
        )
        values = self.v_proj(inputs).view(
            batch, sequence, self.config.num_key_value_heads, self.config.head_dim
        )
        return self.rotary(queries, positions), self.rotary(keys, positions), values

    def _expand_kv(self, values: torch.Tensor) -> torch.Tensor:
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        return values.repeat_interleave(repeats, dim=2)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        queries, keys, values = self._project(inputs, positions)
        queries = queries.transpose(1, 2)
        keys = self._expand_kv(keys).transpose(1, 2)
        values = self._expand_kv(values).transpose(1, 2)
        output = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        output = output.transpose(1, 2).contiguous().view_as(inputs)
        return self.o_proj(output)

    def forward_paged(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        sequence_ids: list[str],
        slots: list[CacheSlot],
        cache: PagedKVCache,
        *,
        backend: str = "pytorch",
        block_tables: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries, keys, values = self._project(inputs, positions[:, None])
        cache.write(self.layer_index, slots, keys[:, 0], values[:, 0])

        if backend == "triton":
            if block_tables is None or sequence_lengths is None:
                raise ValueError("Triton attention requires block tables and sequence lengths")
            output = paged_attention_decode(
                queries[:, 0].contiguous(),
                cache.storage[self.layer_index, 0],
                cache.storage[self.layer_index, 1],
                block_tables,
                sequence_lengths,
                block_size=cache.config.block_size,
                scale=1.0 / math.sqrt(self.config.head_dim),
            )
            return self.o_proj(output.reshape(inputs.shape))
        if backend != "pytorch":
            raise ValueError(f"unknown paged-attention backend: {backend}")

        # Online softmax over physical blocks. This is a deliberately readable PyTorch
        # implementation of paged attention: no contiguous per-sequence KV tensor is built.
        # A fused Triton/CUDA kernel can replace this function without changing cache semantics.
        outputs: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(self.config.head_dim)
        for row, sequence_id in enumerate(sequence_ids):
            query = queries[row, 0].float()  # [heads, head_dim]
            running_max = torch.full(
                (self.config.num_attention_heads,),
                -torch.inf,
                device=inputs.device,
                dtype=torch.float32,
            )
            running_sum = torch.zeros_like(running_max)
            accumulator = torch.zeros_like(query)
            for key_block, value_block in cache.iter_kv_blocks(self.layer_index, sequence_id):
                expanded_keys = self._expand_kv(key_block[None])[0].transpose(0, 1).float()
                expanded_values = self._expand_kv(value_block[None])[0].transpose(0, 1).float()
                scores = torch.einsum("hd,htd->ht", query, expanded_keys) * scale
                block_max = scores.max(dim=-1).values
                next_max = torch.maximum(running_max, block_max)
                correction = torch.exp(running_max - next_max)
                probabilities = torch.exp(scores - next_max[:, None])
                accumulator = accumulator * correction[:, None] + torch.einsum(
                    "ht,htd->hd", probabilities, expanded_values
                )
                running_sum = running_sum * correction + probabilities.sum(dim=-1)
                running_max = next_max
            outputs.append((accumulator / running_sum[:, None]).to(inputs.dtype))
        output = torch.stack(outputs).reshape(inputs.shape)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_index)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.self_attn(self.input_layernorm(inputs), positions)
        return inputs + self.mlp(self.post_attention_layernorm(inputs))

    def forward_paged(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        sequence_ids: list[str],
        slots: list[CacheSlot],
        cache: PagedKVCache,
        *,
        backend: str = "pytorch",
        block_tables: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention = self.self_attn.forward_paged(
            self.input_layernorm(inputs),
            positions,
            sequence_ids,
            slots,
            cache,
            backend=backend,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
        )
        inputs = inputs + attention
        return inputs + self.mlp(self.post_attention_layernorm(inputs))


class Transformer(nn.Module):
    """Qwen2-style decoder with both uncached and custom paged-attention paths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_index) for layer_index in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
        positions = positions.expand(token_ids.shape[0], -1)
        hidden = self.embed_tokens(token_ids)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.norm(hidden))

    def forward_paged(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        sequence_ids: list[str],
        slots: list[CacheSlot],
        cache: PagedKVCache,
        *,
        backend: str = "pytorch",
    ) -> torch.Tensor:
        if token_ids.ndim != 1:
            raise ValueError("paged decode accepts exactly one token per active sequence")
        block_tables = sequence_lengths = None
        if backend == "triton":
            block_tables, sequence_lengths = cache.block_table_tensors(sequence_ids)
        hidden = self.embed_tokens(token_ids)[:, None, :]
        for layer in self.layers:
            hidden = layer.forward_paged(
                hidden,
                positions,
                sequence_ids,
                slots,
                cache,
                backend=backend,
                block_tables=block_tables,
                sequence_lengths=sequence_lengths,
            )
        return self.lm_head(self.norm(hidden[:, 0]))
