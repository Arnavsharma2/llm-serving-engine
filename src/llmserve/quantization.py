from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _pack_int4(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    original_columns = values.shape[1]
    unsigned = (values.to(torch.int16) + 8).to(torch.uint8)
    if original_columns % 2:
        unsigned = F.pad(unsigned, (0, 1), value=8)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    return packed, original_columns


def _unpack_int4(packed: torch.Tensor, columns: int) -> torch.Tensor:
    output = torch.empty(
        packed.shape[0], packed.shape[1] * 2, dtype=torch.int8, device=packed.device
    )
    output[:, 0::2] = (packed & 0x0F).to(torch.int8) - 8
    output[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    return output[:, :columns]


class QuantizedLinear(nn.Module):
    """Simple symmetric per-output-channel weight-only INT8/INT4 linear layer."""

    def __init__(self, source: nn.Linear, bits: int) -> None:
        super().__init__()
        if bits not in {4, 8}:
            raise ValueError("bits must be 4 or 8")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bits = bits
        limit = 127 if bits == 8 else 7
        weight = source.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / limit
        quantized = torch.round(weight / scale).clamp(-limit, limit).to(torch.int8)
        self.register_buffer("scale", scale)
        if bits == 8:
            self.register_buffer("quantized_weight", quantized)
            self.original_columns = self.in_features
        else:
            packed, columns = _pack_int4(quantized)
            self.register_buffer("quantized_weight", packed)
            self.original_columns = columns
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        values = (
            self.quantized_weight
            if self.bits == 8
            else _unpack_int4(self.quantized_weight, self.original_columns)
        )
        return (values.float() * self.scale).to(dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.dequantized_weight(inputs.dtype), self.bias)

    @property
    def storage_bytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size()
            for buffer in self.buffers()
            if buffer is not None
        )


def quantize_model(model: nn.Module, bits: int, *, inplace: bool = False) -> nn.Module:
    """Replace every Linear recursively; deliberately does not claim GPTQ/AWQ calibration."""

    result = model if inplace else copy.deepcopy(model)

    def replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child, bits))
            else:
                replace(child)

    replace(result)
    return result


def model_storage_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        pointer = tensor.data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += tensor.numel() * tensor.element_size()
    return total


@dataclass(frozen=True)
class QuantizationResult:
    bits: int
    perplexity: float
    tokens_per_second: float
    model_size_bytes: int
