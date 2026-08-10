from __future__ import annotations

import torch
from torch import nn

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
