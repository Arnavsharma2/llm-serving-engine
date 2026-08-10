from __future__ import annotations

import httpx
import pytest
import torch

from llmserve.config import CacheConfig, ModelConfig
from llmserve.engine import LLMEngine
from llmserve.model import Transformer
from llmserve.service import create_app
from llmserve.tokenizer import ByteTokenizer


@pytest.mark.asyncio
async def test_openai_completion_endpoint() -> None:
    torch.manual_seed(9)
    tokenizer = ByteTokenizer()
    model = Transformer(
        ModelConfig(
            vocab_size=260,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    engine = LLMEngine(model, tokenizer, cache_config=CacheConfig(2, 16))
    app = create_app(engine)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/completions",
                json={"model": "test", "prompt": "hi", "max_tokens": 2},
            )
    await engine.close()
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert payload["usage"]["completion_tokens"] == 2
