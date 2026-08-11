from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from llmserve.triton_kernels import fused_int8_linear, is_triton_int8_linear_available


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
    """Symmetric per-output-channel weight-only linear layer.

    INT8 uses signed values in ``[-127, 127]``, round-to-nearest-even, one FP32
    scale per output row, and a contiguous ``[in_features, out_features]`` storage
    layout. Zero rows use scale 1 and therefore reconstruct exactly to zero. The
    reference backend reconstructs ``[out_features, in_features]`` immediately
    before ``F.linear``. The Triton backend reads the packed INT8 layout directly,
    dequantizes individual tiles to FP16, accumulates in FP32, and emits FP16.

    INT4 remains the pre-existing reference-only implementation.
    """

    def __init__(
        self,
        source: nn.Linear,
        bits: int,
        *,
        backend: str = "reference",
        fallback_to_reference: bool = True,
    ) -> None:
        super().__init__()
        if bits not in {4, 8}:
            raise ValueError("bits must be 4 or 8")
        if backend not in {"reference", "triton"}:
            raise ValueError("backend must be reference or triton")
        if bits != 8 and backend != "reference":
            raise ValueError("the Triton backend supports INT8 only")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bits = bits
        self.backend = backend
        self.fallback_to_reference = fallback_to_reference
        limit = 127 if bits == 8 else 7
        weight = source.weight.detach().float()
        if bits == 8:
            maximum = weight.abs().amax(dim=1)
            scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / limit)
            quantized = (
                torch.round(weight / scale[:, None]).clamp(-limit, limit).to(torch.int8)
            )
            self.register_buffer("scale", scale.contiguous())
            # K-major storage makes each Triton B tile contiguous along output channels.
            self.register_buffer("quantized_weight", quantized.t().contiguous())
            self.original_columns = self.in_features
        else:
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / limit
            quantized = torch.round(weight / scale).clamp(-limit, limit).to(torch.int8)
            self.register_buffer("scale", scale)
            packed, columns = _pack_int4(quantized)
            self.register_buffer("quantized_weight", packed)
            self.original_columns = columns
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def _apply(self, fn, recurse: bool = True):
        # Module.to(dtype=FP16) must not lower the precision of the quantization contract.
        result = super()._apply(fn, recurse=recurse)
        if self.bits == 8:
            self.scale = self.scale.float()
        return result

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.bits == 8:
            values = self.quantized_weight.t()
            return (values.float() * self.scale[:, None]).to(dtype)
        else:
            values = _unpack_int4(self.quantized_weight, self.original_columns)
            return (values.float() * self.scale).to(dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.backend == "triton":
            if is_triton_int8_linear_available(inputs.device) and inputs.dtype == torch.float16:
                return fused_int8_linear(
                    inputs, self.quantized_weight, self.scale, self.bias
                )
            if not self.fallback_to_reference:
                raise RuntimeError(
                    "Triton INT8 was selected but requires FP16 CUDA activations and "
                    "compute capability 7.5+"
                )
        return F.linear(inputs, self.dequantized_weight(inputs.dtype), self.bias)

    @property
    def storage_bytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size()
            for buffer in self.buffers()
            if buffer is not None
        )


def quantize_model(
    model: nn.Module,
    bits: int,
    *,
    backend: str = "reference",
    inplace: bool = False,
    fallback_to_reference: bool = True,
) -> nn.Module:
    """Replace every Linear recursively; deliberately does not claim GPTQ/AWQ calibration."""

    result = model if inplace else copy.deepcopy(model)
    tied_lm_head = None
    if (
        getattr(getattr(result, "config", None), "tie_word_embeddings", False)
        and isinstance(getattr(result, "lm_head", None), nn.Linear)
        and isinstance(getattr(result, "embed_tokens", None), nn.Embedding)
        and result.lm_head.weight.data_ptr() == result.embed_tokens.weight.data_ptr()
    ):
        # An INT8 output head cannot share physical storage with an FP16 embedding table.
        tied_lm_head = result.lm_head

    def replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                if child is tied_lm_head:
                    continue
                setattr(
                    module,
                    name,
                    QuantizedLinear(
                        child,
                        bits,
                        backend=backend,
                        fallback_to_reference=fallback_to_reference,
                    ),
                )
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
