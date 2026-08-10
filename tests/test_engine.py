from __future__ import annotations

import asyncio

import pytest
import torch

from llmserve.config import CacheConfig, ModelConfig, SchedulerConfig
from llmserve.engine import GenerationConfig, LLMEngine
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
