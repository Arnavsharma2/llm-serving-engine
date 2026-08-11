from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is optional so reference and CPU paths keep working.
    triton = None
    tl = None


def is_triton_int8_linear_available(device: torch.device | str | None = None) -> bool:
    """Return whether the fused W8A16 kernel can run on the requested device."""

    if triton is None or not torch.cuda.is_available():
        return False
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(resolved)
    return (major, minor) >= (7, 5)


if triton is not None:

    @triton.jit
    def _w8a16_linear_kernel(
        inputs,
        weights,
        scales,
        bias,
        output,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        program = tl.program_id(0)
        programs_m = tl.cdiv(M, BLOCK_M)
        program_m = program % programs_m
        program_n = program // programs_m

        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        channel_scales = tl.load(scales + offsets_n, mask=offsets_n < N, other=0.0)

        for start_k in range(0, tl.cdiv(K, BLOCK_K)):
            current_k = start_k * BLOCK_K + offsets_k
            activations = tl.load(
                inputs + offsets_m[:, None] * stride_am + current_k[None, :] * stride_ak,
                mask=(offsets_m[:, None] < M) & (current_k[None, :] < K),
                other=0.0,
            )
            quantized = tl.load(
                weights + current_k[:, None] * stride_wk + offsets_n[None, :] * stride_wn,
                mask=(current_k[:, None] < K) & (offsets_n[None, :] < N),
                other=0,
            )
            dequantized = (quantized.to(tl.float32) * channel_scales[None, :]).to(
                tl.float16
            )
            accumulator += tl.dot(activations, dequantized, out_dtype=tl.float32)

        if HAS_BIAS:
            channel_bias = tl.load(bias + offsets_n, mask=offsets_n < N, other=0.0)
            accumulator += channel_bias[None, :]
        tl.store(
            output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


def _kernel_config(rows: int) -> tuple[int, int, int, int]:
    """T4-oriented tiles for decode, chunked prefill, and larger prefill matrices."""

    if rows <= 16:
        return 16, 64, 32, 4
    if rows <= 64:
        return 32, 64, 32, 4
    return 64, 64, 32, 4


def fused_int8_linear(
    inputs: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a fused W8A16 linear without materializing a dequantized weight matrix.

    ``packed_weight`` is the contiguous ``[in_features, out_features]`` transpose of
    the signed INT8 weight. Dequantization occurs only for the tile held by a Triton
    program. Activations and output are FP16 and the dot accumulator is FP32.
    """

    if not is_triton_int8_linear_available(inputs.device):
        raise RuntimeError(
            "fused INT8 linear requires Triton and a CUDA GPU with compute capability 7.5+"
        )
    if inputs.dtype != torch.float16:
        raise TypeError("fused INT8 linear requires FP16 activations")
    if packed_weight.dtype != torch.int8:
        raise TypeError("fused INT8 linear requires signed INT8 weights")
    if scales.dtype not in {torch.float16, torch.float32}:
        raise TypeError("fused INT8 linear scales must be FP16 or FP32")
    if inputs.ndim < 1 or packed_weight.ndim != 2 or scales.ndim != 1:
        raise ValueError("invalid fused INT8 linear tensor ranks")
    in_features, out_features = packed_weight.shape
    if inputs.shape[-1] != in_features or scales.shape != (out_features,):
        raise ValueError("input, packed weight, and scale shapes are incompatible")
    if bias is not None and (bias.shape != (out_features,) or bias.dtype != inputs.dtype):
        raise ValueError("bias must match the output width and activation dtype")
    tensors = [packed_weight, scales] + ([] if bias is None else [bias])
    if any(tensor.device != inputs.device for tensor in tensors):
        raise ValueError("all fused INT8 linear tensors must be on the same device")
    if not packed_weight.is_contiguous() or not scales.is_contiguous():
        raise ValueError("packed weight and scales must be contiguous")

    contiguous_inputs = inputs.contiguous()
    flattened = contiguous_inputs.view(-1, in_features)
    rows = flattened.shape[0]
    output = torch.empty((rows, out_features), device=inputs.device, dtype=inputs.dtype)
    block_m, block_n, block_k, num_warps = _kernel_config(rows)
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(out_features, block_n),)
    _w8a16_linear_kernel[grid](
        flattened,
        packed_weight,
        scales,
        bias if bias is not None else scales,
        output,
        M=rows,
        N=out_features,
        K=in_features,
        stride_am=flattened.stride(0),
        stride_ak=flattened.stride(1),
        stride_wk=packed_weight.stride(0),
        stride_wn=packed_weight.stride(1),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
        num_stages=2,
    )
    return output.view(*inputs.shape[:-1], out_features)
