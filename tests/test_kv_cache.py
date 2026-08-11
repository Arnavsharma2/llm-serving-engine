from __future__ import annotations

import pytest
import torch

from llmserve.config import CacheConfig, ModelConfig
from llmserve.kv_cache import BlockAllocator, CacheFullError, PagedKVCache


def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_allocator_reuses_freed_blocks_and_checks_owner() -> None:
    allocator = BlockAllocator(2)
    first = allocator.allocate("a")
    allocator.allocate("b")
    with pytest.raises(CacheFullError):
        allocator.allocate("c")
    with pytest.raises(ValueError):
        allocator.free(first, "b")
    allocator.free(first, "a")
    assert allocator.allocate("c") == first


def test_cache_block_table_fragmentation_and_eviction() -> None:
    cache = PagedKVCache(config(), CacheConfig(block_size=4, num_blocks=4))
    cache.create("short")
    for _ in range(5):
        cache.reserve_token("short")
    assert len(cache.sequences["short"].block_table) == 2
    assert cache.stats(32)["paged_fragmentation_pct"] == 37.5
    cache.create("long")
    for _ in range(8):
        cache.reserve_token("long")
    victim, recompute = cache.evict("largest")
    assert (victim, recompute) == ("long", 8)
    assert cache.allocator.free_blocks == 2


def test_cache_writes_and_reads_in_logical_order() -> None:
    cache = PagedKVCache(config(), CacheConfig(block_size=2, num_blocks=4))
    cache.create("s")
    for value in range(3):
        slot = cache.reserve_token("s")
        tensor = torch.full((1, 2, 4), float(value))
        cache.write(0, [slot], tensor, tensor + 10)
    blocks = list(cache.iter_kv_blocks(0, "s"))
    assert [block[0].shape[0] for block in blocks] == [2, 1]
    assert torch.cat([block[0] for block in blocks])[:, 0, 0].tolist() == [0, 1, 2]


def test_cache_materializes_only_block_table_metadata() -> None:
    cache = PagedKVCache(config(), CacheConfig(block_size=2, num_blocks=8))
    cache.create("short")
    cache.create("long")
    cache.reserve_token("short")
    for _ in range(3):
        cache.reserve_token("long")

    block_tables, lengths = cache.block_table_tensors(["short", "long"])

    assert block_tables.dtype == torch.int32
    assert lengths.dtype == torch.int32
    assert block_tables.tolist() == [[0, -1], [1, 2]]
    assert lengths.tolist() == [1, 3]


def test_device_metadata_tracks_allocation_growth_and_completion() -> None:
    cache = PagedKVCache(config(), CacheConfig(block_size=2, num_blocks=4))
    cache.create("first")
    sequence = cache.sequences["first"]
    row = sequence.metadata_row
    cache.reserve_tokens("first", 3)

    assert cache.device_block_tables[row].tolist() == [0, 1, -1, -1]
    assert int(cache.device_context_lengths[row]) == 3
    assert cache.device_block_tables[row, :2].tolist() == sequence.block_table

    cache.free("first")
    assert cache.device_block_tables[row].tolist() == [-1, -1, -1, -1]
    assert int(cache.device_context_lengths[row]) == 0

    cache.create("replacement")
    assert cache.sequences["replacement"].metadata_row != row
    cache.reserve_token("replacement")
    replacement_row = cache.sequences["replacement"].metadata_row
    assert cache.device_block_tables[replacement_row, 1:].tolist() == [-1, -1, -1]


def test_device_metadata_clears_evicted_rows_without_stale_blocks() -> None:
    cache = PagedKVCache(config(), CacheConfig(block_size=2, num_blocks=4))
    cache.create("short")
    cache.reserve_token("short")
    cache.create("long")
    cache.reserve_tokens("long", 4)
    long_row = cache.sequences["long"].metadata_row

    victim, recompute = cache.evict("largest")

    assert (victim, recompute) == ("long", 4)
    assert cache.device_block_tables[long_row].tolist() == [-1, -1, -1, -1]
    assert int(cache.device_context_lengths[long_row]) == 0


def test_iteration_metadata_reuses_buffers_across_block_boundaries() -> None:
    cache = PagedKVCache(config(), CacheConfig(block_size=2, num_blocks=8))
    cache.create("a")
    cache.create("b")
    slots_a = cache.reserve_tokens("a", 3)
    slots_b = cache.reserve_tokens("b", 2)
    metadata = cache.allocate_iteration_metadata(5)
    sequence_ids = ["a", "a", "a", "b", "b"]
    slots = slots_a + slots_b
    lengths = [1, 2, 3, 1, 2]

    rows, device_lengths, block_ids, offsets = cache.prepare_iteration_metadata(
        metadata, sequence_ids, slots, lengths
    )

    assert rows.tolist() == [0, 0, 0, 1, 1]
    assert device_lengths.tolist() == lengths
    assert block_ids.tolist() == [0, 0, 1, 2, 2]
    assert offsets.tolist() == [0, 1, 0, 0, 1]
    assert cache.device_block_tables[0, :2].tolist() == cache.sequences["a"].block_table
    assert cache.device_block_tables[1, :1].tolist() == cache.sequences["b"].block_table
