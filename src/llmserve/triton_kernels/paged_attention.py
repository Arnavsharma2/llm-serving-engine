from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is optional so CPU and unsupported GPU paths keep working.
    triton = None
    tl = None


def is_triton_paged_attention_available(device: torch.device | str | None = None) -> bool:
    """Return whether the fused kernel can run on the requested CUDA device."""

    if triton is None or not torch.cuda.is_available():
        return False
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(resolved)
    return (major, minor) >= (7, 5)


if triton is not None:

    @triton.jit
    def _paged_attention_decode_kernel(
        queries,
        keys,
        values,
        block_tables,
        sequence_lengths,
        output,
        stride_q_batch: tl.constexpr,
        stride_q_head: tl.constexpr,
        stride_q_dim: tl.constexpr,
        stride_k_block: tl.constexpr,
        stride_k_token: tl.constexpr,
        stride_k_head: tl.constexpr,
        stride_k_dim: tl.constexpr,
        stride_v_block: tl.constexpr,
        stride_v_token: tl.constexpr,
        stride_v_head: tl.constexpr,
        stride_v_dim: tl.constexpr,
        stride_table_batch: tl.constexpr,
        stride_table_block: tl.constexpr,
        stride_o_batch: tl.constexpr,
        stride_o_head: tl.constexpr,
        stride_o_dim: tl.constexpr,
        scale: tl.constexpr,
        NUM_QUERY_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PADDED_HEAD_DIM: tl.constexpr,
    ):
        sequence_index = tl.program_id(0)
        query_head = tl.program_id(1)
        queries_per_kv_head: tl.constexpr = NUM_QUERY_HEADS // NUM_KV_HEADS
        kv_head = query_head // queries_per_kv_head

        dimension_offsets = tl.arange(0, PADDED_HEAD_DIM)
        dimension_mask = dimension_offsets < HEAD_DIM
        query = tl.load(
            queries
            + sequence_index * stride_q_batch
            + query_head * stride_q_head
            + dimension_offsets * stride_q_dim,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        sequence_length = tl.load(sequence_lengths + sequence_index)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((PADDED_HEAD_DIM,), dtype=tl.float32)
        token_offsets = tl.arange(0, BLOCK_SIZE)

        for logical_block in range(0, tl.cdiv(sequence_length, BLOCK_SIZE)):
            physical_block = tl.load(
                block_tables
                + sequence_index * stride_table_batch
                + logical_block * stride_table_block
            )
            positions = logical_block * BLOCK_SIZE + token_offsets
            token_mask = positions < sequence_length

            key_offsets = (
                physical_block * stride_k_block
                + token_offsets[:, None] * stride_k_token
                + kv_head * stride_k_head
                + dimension_offsets[None, :] * stride_k_dim
            )
            key = tl.load(
                keys + key_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * query[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max)

            value_offsets = (
                physical_block * stride_v_block
                + token_offsets[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dimension_offsets[None, :] * stride_v_dim
            )
            value = tl.load(
                values + value_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(probabilities[:, None] * value, axis=0)
            running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
            running_max = next_max

        normalized = accumulator / running_sum
        tl.store(
            output
            + sequence_index * stride_o_batch
            + query_head * stride_o_head
            + dimension_offsets * stride_o_dim,
            normalized,
            mask=dimension_mask,
        )


def paged_attention_decode(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Run fused paged-attention decode without reconstructing contiguous KV tensors.

    Args:
        queries: ``[batch, query_heads, head_dim]`` FP16 decode queries.
        keys/values: ``[physical_blocks, block_size, kv_heads, head_dim]`` caches.
        block_tables: ``[batch, logical_blocks]`` physical block ids.
        sequence_lengths: valid cached token count for each sequence.
    """

    if not is_triton_paged_attention_available(queries.device):
        raise RuntimeError(
            "Triton paged attention requires a CUDA GPU with compute capability 7.5+"
        )
    if (
        queries.dtype != torch.float16
        or keys.dtype != torch.float16
        or values.dtype != torch.float16
    ):
        raise TypeError("Triton paged attention requires FP16 queries, keys, and values")
    if block_size != 16:
        raise ValueError("the initial Triton paged-attention kernel requires block_size=16")
    if queries.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("invalid paged-attention tensor ranks or cache shapes")
    batch, num_query_heads, head_dim = queries.shape
    if keys.shape[1] != block_size or keys.shape[3] != head_dim:
        raise ValueError("cache block size and head dimension must match the query")
    num_kv_heads = keys.shape[2]
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    if block_tables.shape[0] != batch or sequence_lengths.shape != (batch,):
        raise ValueError("block tables and sequence lengths must match the query batch")
    if block_tables.device != queries.device or sequence_lengths.device != queries.device:
        raise ValueError("block tables and sequence lengths must be on the query device")
    if block_tables.dtype != torch.int32 or sequence_lengths.dtype != torch.int32:
        raise TypeError("block tables and sequence lengths must use torch.int32")
    if not queries.is_contiguous() or not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError("queries and KV cache views must be contiguous")

    padded_head_dim = triton.next_power_of_2(head_dim)
    if padded_head_dim > 256:
        raise ValueError("head dimensions above 256 are unsupported")
    output = torch.empty_like(queries)
    grid = (batch, num_query_heads)
    _paged_attention_decode_kernel[grid](
        queries,
        keys,
        values,
        block_tables,
        sequence_lengths,
        output,
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        scale=scale if scale is not None else 1.0 / math.sqrt(head_dim),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        PADDED_HEAD_DIM=padded_head_dim,
        num_warps=4,
        num_stages=2,
    )
    return output
