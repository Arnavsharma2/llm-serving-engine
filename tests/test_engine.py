from __future__ import annotations

import asyncio

import pytest
import torch

from llmserve.config import CacheConfig, ModelConfig, SchedulerConfig
from llmserve.engine import GenerationConfig, LLMEngine, sample_token, sample_tokens
from llmserve.kv_cache import CacheFullError
from llmserve.model import Transformer
from llmserve.scheduler import IterationScheduler, RequestState
from llmserve.tokenizer import ByteTokenizer


def make_model() -> Transformer:
    torch.manual_seed(3)
    return Transformer(
        ModelConfig(
            vocab_size=260,
            hidden_size=32,
            intermediate_size=64,
            num_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )


def test_batched_sampling_matches_reference_with_one_result_per_row() -> None:
    torch.manual_seed(17)
    logits = torch.randn(3, 31)
    configs = [
        GenerationConfig(temperature=0.0),
        GenerationConfig(temperature=0.8),
        GenerationConfig(temperature=0.7, top_p=0.8),
    ]
    expected = [
        sample_token(logits[row], config, torch.Generator().manual_seed(101 + row))
        for row, config in enumerate(configs)
    ]
    actual = sample_tokens(
        logits,
        configs,
        [torch.Generator().manual_seed(101 + row) for row in range(len(configs))],
    )
    assert actual == expected


@pytest.mark.asyncio
async def test_continuous_engine_generates_concurrent_requests() -> None:
    engine = LLMEngine(
        make_model(),
        ByteTokenizer(),
        cache_config=CacheConfig(2, 32),
        scheduler_config=SchedulerConfig(max_batch_size=3),
    )
    outputs = await asyncio.gather(
        *[
            engine.generate(text, GenerationConfig(max_new_tokens=3), request_id=f"r{index}")
            for index, text in enumerate(["a", "medium", "third"])
        ]
    )
    await engine.close()
    assert [len(output) for output in outputs] == [3, 3, 3]
    assert engine.cache.allocator.used_blocks == 0
    assert engine.cache_stats["peak_blocks_used"] > 0


@pytest.mark.asyncio
async def test_continuous_iteration_mixes_decode_with_late_prefill() -> None:
    model = make_model()
    calls: list[list[str]] = []
    original_forward = model.forward_paged

    def record_forward(token_ids, positions, sequence_ids, slots, cache, **kwargs):
        calls.append(list(sequence_ids))
        return original_forward(token_ids, positions, sequence_ids, slots, cache, **kwargs)

    model.forward_paged = record_forward  # type: ignore[method-assign]
    engine = LLMEngine(
        model,
        ByteTokenizer(),
        cache_config=CacheConfig(2, 64),
        scheduler_config=SchedulerConfig(max_batch_size=2, max_tokens_per_step=32),
        prefill_chunk_size=16,
    )
    first_token = asyncio.Event()

    def on_long_token(_: int) -> None:
        first_token.set()

    long_task = asyncio.create_task(
        engine.generate(
            list(range(1, 41)),
            GenerationConfig(max_new_tokens=5),
            on_long_token,
            request_id="long",
        )
    )
    await first_token.wait()
    short_task = asyncio.create_task(
        engine.generate([7, 8, 9, 10], GenerationConfig(max_new_tokens=2), request_id="short")
    )
    await asyncio.gather(long_task, short_task)
    await engine.close()

    assert any(call.count("long") == 1 and call.count("short") > 1 for call in calls)


@pytest.mark.asyncio
async def test_static_batching_keeps_late_request_behind_barrier() -> None:
    model = make_model()
    calls: list[set[str]] = []
    original_forward = model.forward_paged

    def record_forward(token_ids, positions, sequence_ids, slots, cache, **kwargs):
        calls.append(set(sequence_ids))
        return original_forward(token_ids, positions, sequence_ids, slots, cache, **kwargs)

    model.forward_paged = record_forward  # type: ignore[method-assign]
    engine = LLMEngine(
        model,
        ByteTokenizer(),
        cache_config=CacheConfig(2, 64),
        scheduler_config=SchedulerConfig(max_batch_size=2, max_tokens_per_step=32),
        scheduler_mode="static",
        prefill_chunk_size=16,
    )
    first_token = asyncio.Event()

    def on_first_token(_: int) -> None:
        first_token.set()

    first_task = asyncio.create_task(
        engine.generate(
            list(range(1, 25)),
            GenerationConfig(max_new_tokens=4),
            on_first_token,
            request_id="first",
        )
    )
    await first_token.wait()
    late_task = asyncio.create_task(
        engine.generate([9, 10], GenerationConfig(max_new_tokens=2), request_id="late")
    )
    await asyncio.gather(first_task, late_task)
    await engine.close()

    assert {"first", "late"} not in calls


@pytest.mark.asyncio
async def test_streaming_callbacks_emit_each_token_once() -> None:
    engine = LLMEngine(
        make_model(),
        ByteTokenizer(),
        cache_config=CacheConfig(2, 32),
        scheduler_config=SchedulerConfig(max_batch_size=2),
        prefill_chunk_size=16,
    )
    streamed: list[int] = []
    output = await engine.generate(
        list(range(1, 20)),
        GenerationConfig(max_new_tokens=5),
        streamed.append,
        request_id="stream",
    )
    await engine.close()

    assert streamed == output
    assert len(streamed) == 5


@pytest.mark.asyncio
async def test_chunked_prefill_matches_token_at_a_time_generation() -> None:
    prompt = list(range(1, 24))
    config = GenerationConfig(max_new_tokens=4, seed=29)
    token_engine = LLMEngine(
        make_model(),
        ByteTokenizer(),
        cache_config=CacheConfig(2, 64),
        scheduler_config=SchedulerConfig(max_batch_size=1),
        prefill_chunk_size=1,
    )
    chunk_engine = LLMEngine(
        make_model(),
        ByteTokenizer(),
        cache_config=CacheConfig(2, 64),
        scheduler_config=SchedulerConfig(max_batch_size=1),
        prefill_chunk_size=16,
    )
    token_output = await token_engine.generate(prompt, config, request_id="token")
    chunk_output = await chunk_engine.generate(prompt, config, request_id="chunk")
    await token_engine.close()
    await chunk_engine.close()
    assert chunk_output == token_output


@pytest.mark.asyncio
async def test_engine_failure_clears_cache_metadata() -> None:
    engine = LLMEngine(
        make_model(),
        ByteTokenizer(),
        cache_config=CacheConfig(2, 64),
        scheduler_config=SchedulerConfig(max_batch_size=1),
        prefill_chunk_size=16,
    )
    with pytest.raises(CacheFullError):
        await engine.generate(
            list(range(1, 66)), GenerationConfig(max_new_tokens=1), request_id="too-long"
        )
    assert engine.cache.allocator.used_blocks == 0
    assert torch.all(engine.cache.device_block_tables == -1)
    assert torch.all(engine.cache.device_context_lengths == 0)
    await engine.close()


@pytest.mark.asyncio
async def test_preemption_replay_does_not_duplicate_streaming_output() -> None:
    engine = LLMEngine(
        make_model(),
        ByteTokenizer(),
        cache_config=CacheConfig(2, 4),
        scheduler_config=SchedulerConfig(max_batch_size=2, max_tokens_per_step=4),
        prefill_chunk_size=2,
    )
    preemptions = 0
    original_preempt = engine.scheduler.preempt

    def record_preempt(request: RequestState) -> None:
        nonlocal preemptions
        preemptions += 1
        original_preempt(request)

    engine.scheduler.preempt = record_preempt  # type: ignore[method-assign]
    streamed = {"a": [], "b": []}
    outputs = await asyncio.gather(
        engine.generate(
            [4, 5, 6, 7],
            GenerationConfig(max_new_tokens=2),
            streamed["a"].append,
            request_id="a",
        ),
        engine.generate(
            [8, 9, 10, 11],
            GenerationConfig(max_new_tokens=2),
            streamed["b"].append,
            request_id="b",
        ),
    )
    assert torch.all(engine.cache.device_block_tables == -1)
    assert torch.all(engine.cache.device_context_lengths == 0)
    await engine.close()

    assert preemptions > 0
    assert streamed["a"] == outputs[0]
    assert streamed["b"] == outputs[1]
    assert [len(streamed["a"]), len(streamed["b"])] == [2, 2]


def test_scheduler_policies_and_static_barrier() -> None:
    scheduler = IterationScheduler(2, policy="sjf", mode="static")
    long = RequestState("long", [1], 20)
    short = RequestState("short", [1], 2)
    later = RequestState("later", [1], 1)
    scheduler.add(long)
    scheduler.add(short)
    assert [item.request_id for item in scheduler.admit()] == ["short", "long"]
    scheduler.add(later)
    assert scheduler.admit() == []
    scheduler.finish(short)
    assert scheduler.admit() == []
