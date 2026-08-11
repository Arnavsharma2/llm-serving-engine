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
