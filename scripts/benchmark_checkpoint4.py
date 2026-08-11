from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

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

BACKENDS = ("fp16", "reference-int8", "triton-int8")
MAJOR_SHAPES = (
    ("q-o-proj", 1536, 1536),
    ("k-v-proj", 1536, 256),
    ("gate-up-proj", 1536, 8960),
    ("down-proj", 8960, 1536),
    ("lm-head", 1536, 151936),
)


@dataclass(frozen=True)
class Experiment:
    workload: str
    batch_size: int
    chunk_size: int


def fixed_profile(
    prompt_tokens: int,
    output_tokens: int,
    requests: int,
    vocabulary_size: int,
    seed: int,
) -> LoadProfile:
    return LoadProfile(
        requests=requests,
        arrival_rate=10_000.0,
        prompt=LengthDistribution(prompt_tokens, 0.0, prompt_tokens, prompt_tokens),
        output=LengthDistribution(output_tokens, 0.0, output_tokens, output_tokens),
        vocabulary_size=vocabulary_size,
        seed=seed,
    )


def profile_for(experiment: Experiment, vocabulary_size: int, seed: int) -> LoadProfile:
    if experiment.workload == "16x8":
        return fixed_profile(16, 8, experiment.batch_size, vocabulary_size, seed)
    if experiment.workload == "128x32":
        return fixed_profile(128, 32, experiment.batch_size, vocabulary_size, seed)
    return LoadProfile(
        requests=max(8, experiment.batch_size * 3),
        arrival_rate=100.0,
        prompt=LengthDistribution(64, 0.9, 8, 256),
        output=LengthDistribution(16, 0.6, 4, 48),
        vocabulary_size=vocabulary_size,
        seed=seed,
    )


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def build_model(model_id: str, backend: str) -> tuple[nn.Module, dict[str, float | int]]:
    release_cuda()
    load_started = time.perf_counter()
    model = load_qwen2(model_id, device="cpu", dtype=torch.float16)
    fp16_load_s = time.perf_counter() - load_started
    quantization_s = 0.0
    if backend != "fp16":
        quant_started = time.perf_counter()
        model = quantize_model(
            model,
            8,
            backend="reference" if backend == "reference-int8" else "triton",
            inplace=True,
            fallback_to_reference=False,
        )
        quantization_s = time.perf_counter() - quant_started
    torch.cuda.reset_peak_memory_stats()
    transfer_started = time.perf_counter()
    model = model.cuda().eval()
    torch.cuda.synchronize()
    transfer_s = time.perf_counter() - transfer_started
    return model, {
        "fp16_cpu_load_s": fp16_load_s,
        "quantization_s": quantization_s,
        "gpu_transfer_s": transfer_s,
        "model_storage_bytes": model_storage_bytes(model),
        "steady_allocated_bytes": torch.cuda.memory_allocated(),
        "steady_reserved_bytes": torch.cuda.memory_reserved(),
        "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def make_engine(model, tokenizer, experiment: Experiment, num_blocks: int) -> LLMEngine:
    return LLMEngine(
        model,
        tokenizer,
        cache_config=CacheConfig(block_size=16, num_blocks=num_blocks),
        scheduler_config=SchedulerConfig(
            max_batch_size=experiment.batch_size,
            max_tokens_per_step=max(2048, experiment.batch_size * experiment.chunk_size),
        ),
        scheduler_mode="continuous",
        paged_attention_backend="triton",
        prefill_chunk_size=experiment.chunk_size,
        cache_device_metadata=True,
        collect_iteration_metrics=True,
        device="cuda",
        dtype=torch.float16,
    )


async def warmup(model, tokenizer, experiment: Experiment, num_blocks: int, seed: int) -> None:
    engine = make_engine(model, tokenizer, experiment, num_blocks)
    try:
        await engine.generate(
            [11, 22],
            GenerationConfig(max_new_tokens=1, seed=seed),
            request_id=f"warm-{experiment.workload}-{experiment.batch_size}",
        )
        torch.cuda.synchronize()
    finally:
        await engine.close()


async def run_experiment(
    model,
    tokenizer,
    backend: str,
    experiment: Experiment,
    profile: LoadProfile,
    repeat: int,
    num_blocks: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    engine = make_engine(model, tokenizer, experiment, num_blocks)
    torch.cuda.reset_peak_memory_stats()
    steady_allocated = torch.cuda.memory_allocated()
    steady_reserved = torch.cuda.memory_reserved()
    harness = BenchmarkHarness(profile)

    async def generate(spec, callback):
        return await engine.generate(
            list(spec.prompt_token_ids),
            GenerationConfig(max_new_tokens=spec.max_new_tokens, seed=seed),
            callback,
            request_id=spec.request_id,
        )

    try:
        report = await harness.run(f"{backend}-{experiment.workload}", generate)
        torch.cuda.synchronize()
        engine_stats = engine.iteration_stats
        summary = {
            **report.summary,
            **asdict(experiment),
            "backend": backend,
            "repeat": repeat,
            "steady_allocated_bytes": steady_allocated,
            "steady_reserved_bytes": steady_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            **engine_stats,
        }
        request_rows = [
            {
                **request.to_dict(),
                "backend": backend,
                "workload": experiment.workload,
                "batch_size": experiment.batch_size,
                "repeat": repeat,
            }
            for request in report.requests
        ]
        return summary, request_rows
    finally:
        await engine.close()


async def component_profile(
    model, tokenizer, backend: str, num_blocks: int, seed: int
) -> dict[str, object]:
    experiment = Experiment("16x8", 8, 16)
    await warmup(model, tokenizer, experiment, num_blocks, seed)
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "linear": [],
        "dequantization": [],
        "attention": [],
        "mlp": [],
    }
    cpu_ms = {"sampling": 0.0, "scheduler": 0.0}

    def timed(stage, original):
        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            events[stage].append((start, end))
            return result

        return wrapped

    linear_class = nn.Linear if backend == "fp16" else QuantizedLinear
    original_linear = linear_class.forward
    original_attention = Attention.forward_paged
    original_mlp = MLP.forward
    original_dequant = QuantizedLinear.dequantized_weight
    original_sample = engine_module.sample_tokens
    original_admit = IterationScheduler.admit
    linear_class.forward = timed("linear", original_linear)
    Attention.forward_paged = timed("attention", original_attention)
    MLP.forward = timed("mlp", original_mlp)
    if backend == "reference-int8":
        QuantizedLinear.dequantized_weight = timed("dequantization", original_dequant)

    def sample(*args, **kwargs):
        started = time.perf_counter()
        result = original_sample(*args, **kwargs)
        cpu_ms["sampling"] += (time.perf_counter() - started) * 1000
        return result

    def admit(self, *args, **kwargs):
        started = time.perf_counter()
        result = original_admit(self, *args, **kwargs)
        cpu_ms["scheduler"] += (time.perf_counter() - started) * 1000
        return result

    engine_module.sample_tokens = sample
    IterationScheduler.admit = admit
    engine = make_engine(model, tokenizer, experiment, num_blocks)
    harness = BenchmarkHarness(profile_for(experiment, model.config.vocab_size, seed))

    async def generate(spec, callback):
        return await engine.generate(
            list(spec.prompt_token_ids),
            GenerationConfig(max_new_tokens=spec.max_new_tokens, seed=seed),
            callback,
            request_id=spec.request_id,
        )

    try:
        report = await harness.run(f"profile-{backend}", generate)
        torch.cuda.synchronize()
    finally:
        await engine.close()
        linear_class.forward = original_linear
        Attention.forward_paged = original_attention
        MLP.forward = original_mlp
        QuantizedLinear.dequantized_weight = original_dequant
        engine_module.sample_tokens = original_sample
        IterationScheduler.admit = original_admit
    return {
        "backend": backend,
        "cuda_ms": {
            stage: sum(start.elapsed_time(end) for start, end in pairs)
            for stage, pairs in events.items()
        },
        "event_counts": {stage: len(pairs) for stage, pairs in events.items()},
        "sampling_cpu_ms": cpu_ms["sampling"],
        "scheduler_cpu_ms": cpu_ms["scheduler"],
        "summary": report.summary,
        "note": "Attention and MLP totals overlap their contained linear totals.",
    }


def microbenchmark(
    backend: str,
    shape_name: str,
    in_features: int,
    out_features: int,
    rows: int,
    repetitions: int,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    torch.manual_seed(seed + rows + in_features + out_features)
    source = nn.Linear(in_features, out_features, bias=out_features <= 1536).cuda().half()
    candidate: nn.Module = source
    if backend != "fp16":
        candidate = QuantizedLinear(
            source,
            8,
            backend="reference" if backend == "reference-int8" else "triton",
            fallback_to_reference=False,
        ).cuda().half()
        del source
    inputs = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)

    for _ in range(5):
        candidate(inputs)
    torch.cuda.synchronize()
    output = []
    for repeat in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            candidate(inputs)
        end.record()
        end.synchronize()
        output.append(
            {
                "backend": backend,
                "shape": shape_name,
                "in_features": in_features,
                "out_features": out_features,
                "rows": rows,
                "repeat": repeat,
                "iterations": iterations,
                "latency_ms": start.elapsed_time(end) / iterations,
            }
        )
    del candidate, inputs
    release_cuda()
    return output


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "throughput_tokens_per_s",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "ttft_ms_p99",
        "tpot_ms_p50",
        "tpot_ms_p95",
        "tpot_ms_p99",
        "e2e_ms_p50",
        "e2e_ms_p95",
        "e2e_ms_p99",
        "gpu_memory_peak_gb",
        "gpu_utilization_mean_pct",
        "steady_allocated_bytes",
        "steady_reserved_bytes",
        "peak_allocated_bytes",
        "prefill_cuda_ms",
        "decode_cuda_ms",
        "mixed_cuda_ms",
    )
    output = []
    keys = sorted(
        {
            (str(row["backend"]), str(row["workload"]), int(row["batch_size"]))
            for row in rows
        }
    )
    for backend, workload, batch_size in keys:
        selected = [
            row
            for row in rows
            if row["backend"] == backend
            and row["workload"] == workload
            and row["batch_size"] == batch_size
        ]
        result: dict[str, object] = {
            "backend": backend,
            "workload": workload,
            "batch_size": batch_size,
        }
        for metric in metrics:
            result[f"{metric}_median"] = statistics.median(
                float(row[metric]) for row in selected
            )
        result["requests_total"] = sum(int(row["requests"]) for row in selected)
        result["successful_requests_total"] = sum(
            int(row["successful_requests"]) for row in selected
        )
        result["preemptions_total"] = sum(int(row["preemptions"]) for row in selected)
        result["recomputed_tokens_total"] = sum(
            int(row["recomputed_tokens"]) for row in selected
        )
        output.append(result)
    return output


def aggregate_micro(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted(
        {
            (str(row["backend"]), str(row["shape"]), int(row["rows"]))
            for row in rows
        }
    )
    for backend, shape, activation_rows in keys:
        selected = [
            float(row["latency_ms"])
            for row in rows
            if row["backend"] == backend
            and row["shape"] == shape
            and row["rows"] == activation_rows
        ]
        output.append(
            {
                "backend": backend,
                "shape": shape,
                "rows": activation_rows,
                "latency_ms_median": statistics.median(selected),
                "latency_ms_min": min(selected),
                "latency_ms_max": max(selected),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(paths: list[Path], destination: Path) -> None:
    destination.write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths)
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--micro-iterations", type=int, default=10)
    parser.add_argument("--micro-rows", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--output", default="artifacts/checkpoint4-t4.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--skip-micro", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 5:
        raise ValueError("Checkpoint 4 requires at least five repetitions")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5):
        raise RuntimeError("Checkpoint 4 requires a CUDA GPU with compute capability 7.5+")

    torch.manual_seed(args.seed)
    tokenizer = HuggingFaceTokenizer(args.model)
    experiments = [
        Experiment(workload, batch_size, 16 if workload != "128x32" else 32)
        for batch_size in args.batch_sizes
        for workload in ("16x8", "128x32", "right-skewed")
    ]
    output = Path(args.output)
    csv_path = output.with_suffix(".csv")
    requests_path = output.with_name(output.stem + "-requests.csv")
    micro_path = output.with_name(output.stem + "-micro.csv")
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": args.model,
        "source_revision": args.source_revision
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    loads: list[dict[str, object]] = []
    component_profiles: list[dict[str, object]] = []
    micro_rows: list[dict[str, object]] = []
    if args.resume and output.exists():
        existing = json.loads(output.read_text())
        for key in ("gpu", "compute_capability", "torch", "cuda", "model"):
            if existing["environment"].get(key) != environment[key]:
                raise ValueError(f"cannot resume with a different {key}")
        rows = existing.get("rows", [])
        request_rows = existing.get("requests", [])
        loads = existing.get("loads", [])
        component_profiles = existing.get("component_profiles", [])
        micro_rows = existing.get("micro_rows", [])
        print(
            "CHECKPOINT4_RESUME",
            len(rows),
            "end-to-end rows and",
            len(micro_rows),
            "micro rows",
            flush=True,
        )

    def persist() -> None:
        payload = {
            "environment": environment,
            "config": vars(args),
            "loads": loads,
            "component_profiles": component_profiles,
            "aggregates": aggregate(rows),
            "micro_aggregates": aggregate_micro(micro_rows),
            "rows": rows,
            "micro_rows": micro_rows,
            "requests": request_rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        write_csv(csv_path, rows)
        write_csv(requests_path, request_rows)
        write_csv(micro_path, micro_rows)

    completed = {
        (
            str(row["backend"]),
            str(row["workload"]),
            int(row["batch_size"]),
            int(row["repeat"]),
        )
        for row in rows
    }
    for backend in args.backends:
        if args.skip_end_to_end:
            break
        missing = [
            (experiment, repeat)
            for experiment in experiments
            for repeat in range(args.repetitions)
            if (backend, experiment.workload, experiment.batch_size, repeat) not in completed
        ]
        already_profiled = any(row["backend"] == backend for row in component_profiles)
        if not missing and already_profiled:
            continue
        model, load_metrics = build_model(args.model, backend)
        loads = [row for row in loads if row["backend"] != backend]
        loads.append({"backend": backend, **load_metrics})
        persist()
        for experiment in experiments:
            missing_repeats = [
                repeat
                for repeat in range(args.repetitions)
                if (backend, experiment.workload, experiment.batch_size, repeat)
                not in completed
            ]
            if not missing_repeats:
                continue
            profile = profile_for(experiment, model.config.vocab_size, args.seed)
            await warmup(model, tokenizer, experiment, args.num_blocks, args.seed)
            for repeat in missing_repeats:
                row, requests = await run_experiment(
                    model,
                    tokenizer,
                    backend,
                    experiment,
                    profile,
                    repeat,
                    args.num_blocks,
                    args.seed,
                )
                rows.append(row)
                request_rows.extend(requests)
                print("CHECKPOINT4_ROW", json.dumps(row, sort_keys=True), flush=True)
                persist()
        if not already_profiled:
            component_profiles.append(
                await component_profile(model, tokenizer, backend, args.num_blocks, args.seed)
            )
            persist()
        del model
        release_cuda()

    completed_micro = {
        (str(row["backend"]), str(row["shape"]), int(row["rows"]), int(row["repeat"]))
        for row in micro_rows
    }
    for shape_name, in_features, out_features in (() if args.skip_micro else MAJOR_SHAPES):
        for activation_rows in args.micro_rows:
            for backend in args.backends:
                if all(
                    (backend, shape_name, activation_rows, repeat) in completed_micro
                    for repeat in range(args.repetitions)
                ):
                    continue
                measured = microbenchmark(
                    backend,
                    shape_name,
                    in_features,
                    out_features,
                    activation_rows,
                    args.repetitions,
                    args.micro_iterations,
                    args.seed,
                )
                micro_rows.extend(measured)
                print("CHECKPOINT4_MICRO", json.dumps(measured[-1], sort_keys=True), flush=True)
                persist()

    persist()
    checksum_path = output.with_suffix(".sha256")
    write_checksums([output, csv_path, requests_path, micro_path], checksum_path)
    print("CHECKPOINT4_CHECKSUMS", checksum_path.read_text(), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
