from __future__ import annotations

import pytest
import torch
from torch import nn

from llmserve.quantization import QuantizedLinear
from llmserve.triton_kernels import is_triton_int8_linear_available

pytestmark = pytest.mark.skipif(
    not is_triton_int8_linear_available(),
    reason="fused INT8 linear requires Triton and a CUDA GPU with compute capability 7.5+",
)


@pytest.mark.parametrize(
    ("in_features", "out_features"),
    [
        (1536, 1536),  # Q and attention output projections
        (1536, 256),  # K and V projections for Qwen GQA
        (1536, 8960),  # MLP gate and up projections
        (8960, 1536),  # MLP down projection
        (1536, 151936),  # untied LM head shape
    ],
)
@pytest.mark.parametrize("rows", [1, 17, 128])
def test_fused_int8_matches_reference_for_qwen_shapes(
    in_features: int, out_features: int, rows: int
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(101 + rows + out_features)
    source = nn.Linear(in_features, out_features, bias=out_features <= 1536).cuda().half()
    with torch.no_grad():
        source.weight.normal_(generator=generator)
        if source.bias is not None:
            source.bias.normal_(generator=generator)
    layer = QuantizedLinear(source, 8, backend="triton").cuda().half()
    inputs = torch.randn(
        rows,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    expected = torch.nn.functional.linear(
        inputs, layer.dequantized_weight(inputs.dtype), layer.bias
    )
    actual = layer(inputs)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 15, 16, 31, 32, 63, 64, 65])
def test_fused_int8_handles_decode_batches_and_tile_boundaries(rows: int) -> None:
    torch.manual_seed(131)
    source = nn.Linear(95, 79).cuda().half()
    layer = QuantizedLinear(source, 8, backend="triton").cuda().half()
    inputs = torch.randn(rows, 95, device="cuda", dtype=torch.float16)
    expected = torch.nn.functional.linear(
        inputs, layer.dequantized_weight(inputs.dtype), layer.bias
    )
    torch.testing.assert_close(layer(inputs), expected, atol=2e-2, rtol=2e-2)


def test_fused_int8_supports_batched_sequence_activations() -> None:
    torch.manual_seed(149)
    layer = QuantizedLinear(nn.Linear(96, 65).cuda().half(), 8, backend="triton")
    inputs = torch.randn(4, 8, 96, device="cuda", dtype=torch.float16)
    expected = torch.nn.functional.linear(
        inputs, layer.dequantized_weight(inputs.dtype), layer.bias
    )
    torch.testing.assert_close(layer(inputs), expected, atol=2e-2, rtol=2e-2)


def test_fused_forward_never_calls_full_weight_dequantization(monkeypatch) -> None:
    layer = QuantizedLinear(nn.Linear(96, 65).cuda().half(), 8, backend="triton")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("full dequantization was called")

    monkeypatch.setattr(layer, "dequantized_weight", forbidden)
    output = layer(torch.randn(8, 96, device="cuda", dtype=torch.float16))
    assert output.shape == (8, 65)
