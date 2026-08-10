from __future__ import annotations

import torch

from llmserve.config import ModelConfig
from llmserve.model import Transformer
from llmserve.speculative import SpeculativeDecoder


def test_identical_models_accept_every_draft_token() -> None:
    torch.manual_seed(5)
    model = Transformer(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    result = SpeculativeDecoder(model, model, speculation_tokens=3).generate([4, 7], 8)
    assert result.acceptance_rate == 1.0
    assert len(result.token_ids) == 8
    assert result.target_passes < len(result.token_ids)
