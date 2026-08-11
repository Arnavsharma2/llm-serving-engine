from __future__ import annotations

import math

import pytest
import torch

from llmserve.config import CacheConfig, ModelConfig
from llmserve.kv_cache import PagedKVCache
from llmserve.model import Attention
from llmserve.triton_kernels import (
    is_triton_paged_attention_available,
    paged_attention_decode,
)

pytestmark = pytest.mark.skipif(
    not is_triton_paged_attention_available(),
    reason="Triton paged attention requires a CUDA GPU with compute capability 7.5+",
)


def _make_paged_inputs(
    lengths: list[int], *, query_heads: int = 12, kv_heads: int = 2, head_dim: int = 32
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(31)
    blocks_per_sequence = [math.ceil(length / 16) for length in lengths]
    physical_blocks = sum(blocks_per_sequence) + 3
    keys = torch.randn(
        physical_blocks,
        16,
        kv_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    values = torch.randn_like(keys)
    queries = torch.randn(
        len(lengths),
        query_heads,
        head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    max_blocks = max(blocks_per_sequence)
    block_tables = torch.full((len(lengths), max_blocks), -1, device="cuda", dtype=torch.int32)
    available = torch.randperm(physical_blocks, generator=generator, device="cuda")
    cursor = 0
    for row, count in enumerate(blocks_per_sequence):
        block_tables[row, :count] = available[cursor : cursor + count].to(torch.int32)
        cursor += count
    sequence_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return queries, keys, values, block_tables, sequence_lengths


def _full_attention_reference(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    repeats = queries.shape[1] // keys.shape[2]
    for row in range(queries.shape[0]):
        length = int(sequence_lengths[row].item())
        logical_keys = []
        logical_values = []
        for logical_block in range(math.ceil(length / 16)):
            physical_block = int(block_tables[row, logical_block].item())
            take = min(16, length - 16 * logical_block)
            logical_keys.append(keys[physical_block, :take])
            logical_values.append(values[physical_block, :take])
        sequence_keys = torch.cat(logical_keys).repeat_interleave(repeats, dim=1)
        sequence_values = torch.cat(logical_values).repeat_interleave(repeats, dim=1)
        scores = torch.einsum(
            "hd,thd->ht", queries[row].float(), sequence_keys.float()
        ) / math.sqrt(queries.shape[-1])
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probabilities, sequence_values.float()).half())
    return torch.stack(outputs)


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33])
def test_triton_paged_attention_matches_full_attention_at_boundaries(length: int) -> None:
    inputs = _make_paged_inputs([length])
    actual = paged_attention_decode(*inputs, block_size=16)
    expected = _full_attention_reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_paged_attention_handles_multiple_sequence_lengths() -> None:
    inputs = _make_paged_inputs([1, 15, 16, 17, 31, 32, 33])
    actual = paged_attention_decode(*inputs, block_size=16)
    expected = _full_attention_reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_paged_attention_maps_twelve_query_heads_to_two_kv_heads() -> None:
    inputs = _make_paged_inputs([17, 33], query_heads=12, kv_heads=2)
    actual = paged_attention_decode(*inputs, block_size=16)
    expected = _full_attention_reference(*inputs)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_matches_existing_pytorch_paged_implementation() -> None:
    lengths = [1, 15, 16, 17, 31, 32, 33]
    config = ModelConfig(
        vocab_size=64,
        hidden_size=192,
        intermediate_size=256,
        num_layers=1,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(47)
    attention = Attention(config, layer_index=0).cuda().half().eval()
    reference_cache = PagedKVCache(
        config, CacheConfig(block_size=16, num_blocks=32), device="cuda", dtype=torch.float16
    )
    triton_cache = PagedKVCache(
        config, CacheConfig(block_size=16, num_blocks=32), device="cuda", dtype=torch.float16
    )
    generator = torch.Generator(device="cuda").manual_seed(53)
    sequence_ids = [f"sequence-{index}" for index in range(len(lengths))]
    for sequence_id, length in zip(sequence_ids, lengths, strict=True):
        reference_cache.create(sequence_id)
        triton_cache.create(sequence_id)
        for _ in range(length - 1):
            reference_slot = reference_cache.reserve_token(sequence_id)
            triton_slot = triton_cache.reserve_token(sequence_id)
            key = torch.randn(
                1,
                config.num_key_value_heads,
                config.head_dim,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            value = torch.randn_like(key)
            reference_cache.write(0, [reference_slot], key, value)
            triton_cache.write(0, [triton_slot], key, value)

    reference_slots = [reference_cache.reserve_token(item) for item in sequence_ids]
    triton_slots = [triton_cache.reserve_token(item) for item in sequence_ids]
    hidden = torch.randn(
        len(lengths),
        1,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    positions = torch.tensor([length - 1 for length in lengths], device="cuda")
    with torch.inference_mode():
        expected = attention.forward_paged(
            hidden,
            positions,
            sequence_ids,
            reference_slots,
            reference_cache,
            backend="pytorch",
        )
        block_tables, sequence_lengths = triton_cache.block_table_tensors(sequence_ids)
        actual = attention.forward_paged(
            hidden,
            positions,
            sequence_ids,
            triton_slots,
            triton_cache,
            backend="triton",
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
        )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
