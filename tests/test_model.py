from __future__ import annotations

import torch

from llmserve.config import CacheConfig, ModelConfig
from llmserve.kv_cache import PagedKVCache
from llmserve.model import Transformer


def tiny_model() -> Transformer:
    torch.manual_seed(11)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Transformer(config).eval()


def test_paged_attention_matches_full_causal_attention() -> None:
    model = tiny_model()
    tokens = torch.tensor([[4, 9, 7, 12, 5]])
    with torch.inference_mode():
        expected = model(tokens)[0]
        cache = PagedKVCache(model.config, CacheConfig(block_size=2, num_blocks=8))
        cache.create("sequence")
        actual = []
        for position, token in enumerate(tokens[0]):
            slot = cache.reserve_token("sequence")
            logits = model.forward_paged(
                token[None], torch.tensor([position]), ["sequence"], [slot], cache
            )
            actual.append(logits[0])
    assert torch.allclose(torch.stack(actual), expected, atol=1e-4, rtol=1e-4)


def test_paged_attention_handles_different_sequence_lengths() -> None:
    model = tiny_model()
    cache = PagedKVCache(model.config, CacheConfig(block_size=2, num_blocks=8))
    cache.create("a")
    cache.create("b")
    with torch.inference_mode():
        for token in [4, 5]:
            slot = cache.reserve_token("a")
            model.forward_paged(
                torch.tensor([token]),
                torch.tensor([cache.sequences["a"].length - 1]),
                ["a"],
                [slot],
                cache,
            )
        slots = [cache.reserve_token("a"), cache.reserve_token("b")]
        logits = model.forward_paged(
            torch.tensor([6, 9]), torch.tensor([2, 0]), ["a", "b"], slots, cache
        )
    assert logits.shape == (2, model.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_packed_chunked_prefill_matches_full_causal_attention() -> None:
    model = tiny_model()
    tokens = torch.tensor([4, 9, 7, 12, 5])
    with torch.inference_mode():
        expected = model(tokens[None])[0]
        cache = PagedKVCache(model.config, CacheConfig(block_size=2, num_blocks=8))
        cache.create("sequence")
        slots = cache.reserve_tokens("sequence", len(tokens))
        metadata = cache.allocate_iteration_metadata(len(tokens))
        sequence_ids = ["sequence"] * len(tokens)
        context_lengths = list(range(1, len(tokens) + 1))
        rows, lengths, block_ids, offsets = cache.prepare_iteration_metadata(
            metadata, sequence_ids, slots, context_lengths
        )
        actual = model.forward_paged(
            tokens,
            torch.arange(len(tokens)),
            sequence_ids,
            slots,
            cache,
            context_lengths=context_lengths,
            block_table_rows=rows,
            sequence_lengths=lengths,
            block_ids=block_ids,
            offsets=offsets,
        )
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_packed_prefill_handles_mixed_lengths_and_boundary_crossings() -> None:
    model = tiny_model()
    cache = PagedKVCache(model.config, CacheConfig(block_size=2, num_blocks=8))
    cache.create("a")
    cache.create("b")
    tokens = torch.tensor([4, 5, 6, 9, 10])
    sequence_ids = ["a", "a", "a", "b", "b"]
    context_lengths = [1, 2, 3, 1, 2]
    slots = cache.reserve_tokens("a", 3) + cache.reserve_tokens("b", 2)
    metadata = cache.allocate_iteration_metadata(len(tokens))
    rows, lengths, block_ids, offsets = cache.prepare_iteration_metadata(
        metadata, sequence_ids, slots, context_lengths
    )

    with torch.inference_mode():
        actual = model.forward_paged(
            tokens,
            torch.tensor([0, 1, 2, 0, 1]),
            sequence_ids,
            slots,
            cache,
            context_lengths=context_lengths,
            block_table_rows=rows,
            sequence_lengths=lengths,
            block_ids=block_ids,
            offsets=offsets,
        )
        expected_a = model(torch.tensor([[4, 5, 6]]))[0]
        expected_b = model(torch.tensor([[9, 10]]))[0]
    torch.testing.assert_close(actual[:3], expected_a, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual[3:], expected_b, atol=1e-4, rtol=1e-4)
