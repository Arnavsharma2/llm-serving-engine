from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

import llmserve.engine as engine_module
from llmserve.benchmark import BenchmarkHarness, LoadProfile
from llmserve.benchmark.workload import LengthDistribution
from llmserve.checkpoint import load_qwen2
from llmserve.config import CacheConfig, SchedulerConfig
from llmserve.engine import GenerationConfig, LLMEngine
from llmserve.model import MLP, Attention
from llmserve.quantization import QuantizedLinear, model_storage_bytes, quantize_model
from llmserve.scheduler import IterationScheduler
from llmserve.tokenizer import HuggingFaceTokenizer


def fixed_profile(vocabulary_size: int, seed: int) -> LoadProfile:
    return LoadProfile(
        requests=8,
        arrival_rate=10_000.0,
        prompt=LengthDistribution(16, 0.0, 16, 16),
        output=LengthDistribution(8, 0.0, 8, 8),
        vocabulary_size=vocabulary_size,
        seed=seed,
    )


def write_requests(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--output", default="artifacts/checkpoint4-t4-pre-edit-profile.json"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the Checkpoint 4 reference profile requires CUDA")

    torch.manual_seed(args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = load_qwen2(args.model, device="cpu", dtype=torch.float16)
    fp16_cpu_load_s = time.perf_counter() - load_started
    quantization_started = time.perf_counter()
    model = quantize_model(model, 8, inplace=True)
    quantization_s = time.perf_counter() - quantization_started
    transfer_started = time.perf_counter()
    model = model.cuda().eval()
    torch.cuda.synchronize()
    gpu_transfer_s = time.perf_counter() - transfer_started
    tokenizer = HuggingFaceTokenizer(args.model)
    steady_allocated = torch.cuda.memory_allocated()
    steady_reserved = torch.cuda.memory_reserved()
    load_peak_allocated = torch.cuda.max_memory_allocated()

    def make_engine() -> LLMEngine:
        return LLMEngine(
            model,
            tokenizer,
            cache_config=CacheConfig(block_size=16, num_blocks=args.num_blocks),
            scheduler_config=SchedulerConfig(max_batch_size=8, max_tokens_per_step=2048),
            scheduler_mode="continuous",
            paged_attention_backend="triton",
            prefill_chunk_size=16,
            cache_device_metadata=True,
            collect_iteration_metrics=True,
            device="cuda",
            dtype=torch.float16,
        )

    warmup = make_engine()
    try:
        await warmup.generate(
            [11, 22],
            GenerationConfig(max_new_tokens=1, seed=args.seed),
            request_id="checkpoint4-reference-warmup",
        )
        torch.cuda.synchronize()
    finally:
        await warmup.close()
    torch.cuda.reset_peak_memory_stats()

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    cpu_ms: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    def timed(stage, original):
        def wrapped(*wrapped_args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*wrapped_args, **kwargs)
            end.record()
            events[stage].append((start, end))
            return result

        return wrapped

    original_dequant = QuantizedLinear.dequantized_weight
    original_linear = QuantizedLinear.forward
    original_attention = Attention.forward_paged
    original_mlp = MLP.forward
    original_sample = engine_module.sample_tokens
    original_admit = IterationScheduler.admit
    QuantizedLinear.dequantized_weight = timed("dequantization", original_dequant)
    QuantizedLinear.forward = timed("linear", original_linear)
    Attention.forward_paged = timed("attention", original_attention)
    MLP.forward = timed("mlp", original_mlp)

    def sample(*sample_args, **kwargs):
        started = time.perf_counter()
        result = original_sample(*sample_args, **kwargs)
        cpu_ms["sampling"] += (time.perf_counter() - started) * 1000
        counts["sampling_calls"] += 1
        return result

    def admit(self, *admit_args, **kwargs):
        started = time.perf_counter()
        result = original_admit(self, *admit_args, **kwargs)
        cpu_ms["scheduler"] += (time.perf_counter() - started) * 1000
        counts["scheduler_calls"] += 1
        return result

    engine_module.sample_tokens = sample
    IterationScheduler.admit = admit
    engine = make_engine()
    harness = BenchmarkHarness(fixed_profile(model.config.vocab_size, args.seed))

    async def generate(spec, callback):
        return await engine.generate(
            list(spec.prompt_token_ids),
            GenerationConfig(max_new_tokens=spec.max_new_tokens, seed=args.seed),
            callback,
            request_id=spec.request_id,
        )

    try:
        report = await harness.run("reference-int8-pre-edit", generate)
        torch.cuda.synchronize()
        iteration_stats = engine.iteration_stats
    finally:
        await engine.close()
        QuantizedLinear.dequantized_weight = original_dequant
        QuantizedLinear.forward = original_linear
        Attention.forward_paged = original_attention
        MLP.forward = original_mlp
        engine_module.sample_tokens = original_sample
        IterationScheduler.admit = original_admit

    stage_ms = {
        stage: sum(start.elapsed_time(end) for start, end in pairs)
        for stage, pairs in events.items()
    }
    source_revision = args.source_revision or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    checkpoint3_path = Path("artifacts/checkpoint3-t4.json")
    checkpoint3_fp16 = None
    if checkpoint3_path.exists():
        checkpoint3 = json.loads(checkpoint3_path.read_text())
        checkpoint3_fp16 = next(
            (
                row
                for row in checkpoint3["aggregates"]
                if row["experiment"] == "metadata-cached-16x8-b8-continuous-fcfs"
            ),
            None,
        )
    requests = [request.to_dict() for request in report.requests]
    artifact = {
        "phase": "checkpoint4-pre-edit-reference-int8-profile",
        "source_revision": source_revision,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model": args.model,
        },
        "command": " ".join([sys.executable, *sys.argv]),
        "workload": {
            "seed": args.seed,
            "requests": 8,
            "prompt_tokens": 16,
            "output_tokens": 8,
            "scheduler": "continuous",
            "prefill_chunk_size": 16,
            "paged_attention_backend": "triton",
        },
        "implementation_confirmation": {
            "creates_full_dequantized_weight_before_gemm": True,
            "code_path": "QuantizedLinear.forward -> dequantized_weight -> F.linear",
            "full_dequantized_bytes_per_model_iteration": sum(
                module.out_features * module.in_features * 2
                for module in model.modules()
                if isinstance(module, QuantizedLinear)
            ),
        },
        "loading": {
            "fp16_cpu_load_s": fp16_cpu_load_s,
            "reference_int8_quantization_s": quantization_s,
            "gpu_transfer_s": gpu_transfer_s,
        },
        "memory": {
            "model_storage_bytes": model_storage_bytes(model),
            "steady_torch_allocated_bytes": steady_allocated,
            "steady_torch_reserved_bytes": steady_reserved,
            "load_peak_torch_allocated_bytes": load_peak_allocated,
            "inference_peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nvml_peak_bytes": report.gpu_memory_peak_bytes,
        },
        "timing": {
            "dequantization_cuda_ms": stage_ms["dequantization"],
            "quantized_linear_total_cuda_ms": stage_ms["linear"],
            "gemm_bias_estimated_cuda_ms": stage_ms["linear"]
            - stage_ms["dequantization"],
            "attention_total_cuda_ms": stage_ms["attention"],
            "mlp_total_cuda_ms": stage_ms["mlp"],
            "sampling_cpu_ms": cpu_ms["sampling"],
            "scheduler_cpu_ms": cpu_ms["scheduler"],
            "iteration_stats": iteration_stats,
            "event_counts": {stage: len(pairs) for stage, pairs in events.items()},
            "call_counts": counts,
            "note": "Attention and MLP totals overlap their contained linear totals.",
        },
        "reference_int8_summary": report.summary,
        "checkpoint3_fp16_identical_trace_aggregate": checkpoint3_fp16,
        "reference_int8_requests": requests,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    requests_path = output.with_name(output.stem + "-requests.csv")
    write_requests(requests_path, requests)
    checksum_path = output.with_suffix(".sha256")
    checksum_path.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (output, requests_path)
        )
    )
    print("CHECKPOINT4_PRE_EDIT_PROFILE", json.dumps(artifact, sort_keys=True), flush=True)
    print("CHECKPOINT4_PRE_EDIT_CHECKSUMS", checksum_path.read_text(), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
