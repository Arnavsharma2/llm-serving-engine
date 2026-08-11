from __future__ import annotations

import pytest
import torch
from torch import nn

from llmserve.config import ModelConfig
from llmserve.model import Transformer
from llmserve.quantization import QuantizedLinear, _pack_int4, _unpack_int4, quantize_model


def test_int4_pack_round_trip() -> None:
    values = torch.tensor([[-7, -1, 0, 3, 7]], dtype=torch.int8)
    packed, columns = _pack_int4(values)
    assert packed.numel() == 3
    assert torch.equal(_unpack_int4(packed, columns), values)


def test_per_channel_quantization_preserves_linear_shape_and_reasonable_error() -> None:
    torch.manual_seed(2)
    layer = nn.Linear(17, 9)
    inputs = torch.randn(4, 17)
    expected = layer(inputs)
    int8 = QuantizedLinear(layer, 8)(inputs)
    int4 = QuantizedLinear(layer, 4)(inputs)
    assert int8.shape == expected.shape
    assert (int8 - expected).abs().mean() < (int4 - expected).abs().mean()


def test_quantize_model_replaces_nested_linears() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 2)))
    quantized = quantize_model(model, 8)
    assert isinstance(quantized[0], QuantizedLinear)
    assert isinstance(quantized[1][0], QuantizedLinear)
    assert isinstance(model[0], nn.Linear)


def test_int8_quantizer_handles_zero_rows_and_extreme_values() -> None:
    layer = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-1.0e6, -1.0, 1.0, 1.0e6],
                    [-float("inf"), 0.0, float("inf"), float("nan")],
                ]
            )
        )
        # The contract supports finite weights; sanitize the deliberately extreme row.
        layer.weight[2].nan_to_num_(nan=0.0, posinf=3.4e38, neginf=-3.4e38)
    quantized = QuantizedLinear(layer, 8)
    reconstructed = quantized.dequantized_weight(torch.float32)
    assert quantized.scale[0] == 1
    assert torch.equal(reconstructed[0], torch.zeros(4))
    assert torch.isfinite(reconstructed).all()
    assert quantized.quantized_weight.min() >= -127
    assert quantized.quantized_weight.max() <= 127


def test_int8_packed_layout_is_k_major_contiguous_and_reconstructs() -> None:
    layer = nn.Linear(7, 5, bias=False)
    quantized = QuantizedLinear(layer, 8)
    expected = (
        quantized.quantized_weight.t().float() * quantized.scale[:, None]
    )
    assert quantized.quantized_weight.shape == (7, 5)
    assert quantized.quantized_weight.is_contiguous()
    torch.testing.assert_close(quantized.dequantized_weight(torch.float32), expected)


def test_triton_backend_falls_back_to_reference_on_cpu() -> None:
    torch.manual_seed(13)
    source = nn.Linear(9, 6)
    reference = QuantizedLinear(source, 8, backend="reference")
    fused = QuantizedLinear(source, 8, backend="triton")
    inputs = torch.randn(2, 3, 9)
    torch.testing.assert_close(fused(inputs), reference(inputs))


def test_triton_backend_can_fail_closed_when_fallback_is_disabled() -> None:
    layer = QuantizedLinear(
        nn.Linear(4, 3), 8, backend="triton", fallback_to_reference=False
    )
    with pytest.raises(RuntimeError, match="requires FP16 CUDA"):
        layer(torch.randn(2, 4))


def test_transformer_quantization_preserves_tied_embedding_head() -> None:
    model = Transformer(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=True,
        )
    )
    quantized = quantize_model(model, 8, backend="triton")
    assert isinstance(quantized.lm_head, nn.Linear)
    assert quantized.lm_head.weight.data_ptr() == quantized.embed_tokens.weight.data_ptr()
    assert all(
        isinstance(module, QuantizedLinear)
        for layer in quantized.layers
        for module in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        )
    )


def test_inplace_conversion_does_not_retain_converted_fp_weights() -> None:
    model = nn.Sequential(nn.Linear(8, 7), nn.Linear(7, 3))
    original_pointers = {parameter.data_ptr() for parameter in model.parameters()}
    quantized = quantize_model(model, 8, backend="triton", inplace=True)
    live_pointers = {
        tensor.data_ptr()
        for tensor in list(quantized.parameters()) + list(quantized.buffers())
        if tensor is not None
    }
    assert not original_pointers & live_pointers
    assert not list(quantized.parameters())
