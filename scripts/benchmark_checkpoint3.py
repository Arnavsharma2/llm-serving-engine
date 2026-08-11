from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from llmserve.benchmark import BenchmarkHarness, LoadProfile
from llmserve.benchmark.workload import LengthDistribution
from llmserve.checkpoint import load_qwen2
from llmserve.config import CacheConfig, SchedulerConfig
from llmserve.engine import GenerationConfig, LLMEngine
from llmserve.tokenizer import HuggingFaceTokenizer


@dataclass(frozen=True)
class Experiment:
    comparison: str
    variant: str
    profile_name: str
    batch_size: int
    scheduler_mode: str
    chunk_size: int
    cache_device_metadata: bool
    policy: str = "fcfs"

    @property
    def name(self) -> str:
        return (
            f"{self.comparison}-{self.variant}-{self.profile_name}-"
            f"b{self.batch_size}-{self.scheduler_mode}-{self.policy}"
        )


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


def right_skewed_profile(requests: int, vocabulary_size: int, seed: int) -> LoadProfile:
    return LoadProfile(
        requests=requests,
        arrival_rate=100.0,
        prompt=LengthDistribution(64, 0.9, 8, 256),
        output=LengthDistribution(16, 0.6, 4, 48),
        vocabulary_size=vocabulary_size,
        seed=seed,
    )


def build_experiments(batch_sizes: list[int]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for batch_size in batch_sizes:
        experiments.extend(
            [
                Experiment(
                    "metadata",
                    "rebuild",
                    "16x8",
                    batch_size,
                    "continuous",
                    1,
                    False,
                ),
                Experiment("metadata", "cached", "16x8", batch_size, "continuous", 1, True),
                Experiment("prefill", "token", "128x32", batch_size, "continuous", 1, True),
                Experiment("prefill", "chunk16", "128x32", batch_size, "continuous", 16, True),
                Experiment("prefill", "chunk32", "128x32", batch_size, "continuous", 32, True),
                Experiment("batching", "static", "right-skewed", batch_size, "static", 16, True),
                Experiment(
                    "batching",
                    "continuous",
                    "right-skewed",
                    batch_size,
                    "continuous",
                    16,
                    True,
                ),
            ]
        )
    return experiments


def profile_for(experiment: Experiment, vocabulary_size: int, seed: int) -> LoadProfile:
    if experiment.profile_name == "16x8":
        return fixed_profile(16, 8, experiment.batch_size, vocabulary_size, seed)
    if experiment.profile_name == "128x32":
        return fixed_profile(128, 32, experiment.batch_size, vocabulary_size, seed)
    requests = max(8, experiment.batch_size * 3)
    return right_skewed_profile(requests, vocabulary_size, seed)


async def warmup(model, tokenizer, experiment: Experiment, seed: int, num_blocks: int) -> None:
    engine = LLMEngine(
        model,
        tokenizer,
        cache_config=CacheConfig(block_size=16, num_blocks=num_blocks),
        scheduler_config=SchedulerConfig(
            max_batch_size=experiment.batch_size,
            max_tokens_per_step=max(2048, experiment.batch_size * experiment.chunk_size),
            policy=experiment.policy,
        ),
        scheduler_mode=experiment.scheduler_mode,
        paged_attention_backend="triton",
        prefill_chunk_size=experiment.chunk_size,
        cache_device_metadata=experiment.cache_device_metadata,
        device="cuda",
        dtype=torch.float16,
    )
    try:
        await engine.generate(
            [11, 22],
            GenerationConfig(max_new_tokens=1, seed=seed),
            request_id=f"warmup-{experiment.name}",
        )
        torch.cuda.synchronize()
    finally:
        await engine.close()


async def run_experiment(
    model,
    tokenizer,
    experiment: Experiment,
    load_profile: LoadProfile,
    repeat: int,
    seed: int,
    num_blocks: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    engine = LLMEngine(
        model,
        tokenizer,
        cache_config=CacheConfig(block_size=16, num_blocks=num_blocks),
        scheduler_config=SchedulerConfig(
            max_batch_size=experiment.batch_size,
            max_tokens_per_step=max(2048, experiment.batch_size * experiment.chunk_size),
            policy=experiment.policy,
        ),
        scheduler_mode=experiment.scheduler_mode,
        paged_attention_backend="triton",
        prefill_chunk_size=experiment.chunk_size,
        cache_device_metadata=experiment.cache_device_metadata,
        collect_iteration_metrics=True,
        device="cuda",
        dtype=torch.float16,
    )
    harness = BenchmarkHarness(load_profile)

    async def generate(spec, callback):
        return await engine.generate(
            list(spec.prompt_token_ids),
            GenerationConfig(max_new_tokens=spec.max_new_tokens, seed=seed),
            callback,
            request_id=spec.request_id,
        )

    try:
        report = await harness.run(experiment.name, generate)
        report.cache = engine.cache_stats
        engine_stats = engine.iteration_stats
        report.metadata["engine"] = engine_stats
        summary = {
            **report.summary,
            **asdict(experiment),
            "experiment": experiment.name,
            "repeat": repeat,
            **engine_stats,
        }
        successful = [request for request in report.requests if request.error is None]
        summary["ttft_ms_max"] = max(
            (request.ttft_ms or 0.0 for request in successful), default=0.0
        )
        summary["e2e_ms_max"] = max((request.e2e_ms or 0.0 for request in successful), default=0.0)
        requests = [
            {
                **request.to_dict(),
                "experiment": experiment.name,
                "comparison": experiment.comparison,
                "variant": experiment.variant,
                "profile_name": experiment.profile_name,
                "batch_size": experiment.batch_size,
                "repeat": repeat,
            }
            for request in report.requests
        ]
        return summary, requests
    finally:
        await engine.close()


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
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
        "ttft_ms_max",
        "e2e_ms_max",
        "gpu_memory_peak_gb",
        "gpu_utilization_mean_pct",
        "metadata_construction_ms",
        "prefill_cuda_ms",
        "decode_cuda_ms",
        "mixed_cuda_ms",
    ]
    aggregates = []
    for experiment in sorted({str(row["experiment"]) for row in rows}):
        selected = [row for row in rows if row["experiment"] == experiment]
        first = selected[0]
        aggregate: dict[str, object] = {
            key: first[key]
            for key in (
                "experiment",
                "comparison",
                "variant",
                "profile_name",
                "batch_size",
                "scheduler_mode",
                "chunk_size",
                "cache_device_metadata",
                "policy",
            )
        }
        for metric in metrics:
            aggregate[f"{metric}_median"] = statistics.median(
                float(row[metric]) for row in selected
            )
        aggregate["preemptions_total"] = sum(int(row["preemptions"]) for row in selected)
        aggregate["recomputed_tokens_total"] = sum(
            int(row["recomputed_tokens"]) for row in selected
        )
        aggregate["successful_requests_total"] = sum(
            int(row["successful_requests"]) for row in selected
        )
        aggregate["requests_total"] = sum(int(row["requests"]) for row in selected)
        aggregates.append(aggregate)
    return aggregates


async def operator_profile(model, tokenizer, seed: int, num_blocks: int) -> dict[str, object]:
    experiment = Experiment("profile", "cached-chunk16", "16x8", 8, "continuous", 16, True)
    await warmup(model, tokenizer, experiment, seed, num_blocks)
    engine = LLMEngine(
        model,
        tokenizer,
        cache_config=CacheConfig(block_size=16, num_blocks=num_blocks),
        scheduler_config=SchedulerConfig(max_batch_size=8, max_tokens_per_step=2048),
        scheduler_mode="continuous",
        paged_attention_backend="triton",
        prefill_chunk_size=16,
        device="cuda",
        dtype=torch.float16,
    )
    prompts = [
        [
            4 + ((seed + row * 104729 + column * 1009) % (model.config.vocab_size - 4))
            for column in range(16)
        ]
        for row in range(8)
    ]
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=False,
        ) as measured:
            await asyncio.gather(
                *[
                    engine.generate(
                        prompt,
                        GenerationConfig(max_new_tokens=8, seed=seed),
                        request_id=f"operator-{row}",
                    )
                    for row, prompt in enumerate(prompts)
                ]
            )
            torch.cuda.synchronize()
    finally:
        await engine.close()
    allocation_names = {
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::zeros",
        "aten::zeros_like",
        "aten::full",
        "aten::tensor",
        "aten::arange",
        "aten::new_empty",
    }
    tracked = allocation_names | {
        "aten::to",
        "aten::_to_copy",
        "aten::copy_",
        "aten::item",
        "aten::_local_scalar_dense",
    }
    operators = {
        average.key: {
            "calls": average.count,
            "self_cpu_ms": average.self_cpu_time_total / 1000,
            "self_cuda_ms": getattr(average, "self_cuda_time_total", 0.0) / 1000,
        }
        for average in measured.key_averages()
        if average.key in tracked
    }
    return {
        "allocation_operator_calls": sum(
            row["calls"] for name, row in operators.items() if name in allocation_names
        ),
        "operators": operators,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = sorted({column for row in rows for column in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoint(
    output: Path,
    environment: dict[str, object],
    config: dict[str, object],
    rows: list[dict[str, object]],
    requests: list[dict[str, object]],
    operator_counts: dict[str, object] | None,
) -> None:
    aggregates = aggregate_rows(rows)
    payload = {
        "environment": environment,
        "config": config,
        "aggregates": aggregates,
        "rows": rows,
        "requests": requests,
        "operator_profile": operator_counts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(output.with_suffix(".csv"), rows)
    write_csv(output.with_name(output.stem + "-requests.csv"), requests)


def write_checksums(paths: list[Path], destination: Path) -> None:
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    destination.write_text("\n".join(lines) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--output", default="artifacts/checkpoint3-t4.json")
    parser.add_argument("--skip-operator-profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if args.repetitions < 5:
        raise ValueError("Checkpoint 3 requires at least five repetitions")
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint 3 benchmark requires CUDA")

    torch.manual_seed(args.seed)
    model = load_qwen2(args.model, device="cuda", dtype=torch.float16)
    tokenizer = HuggingFaceTokenizer(args.model)
    experiments = build_experiments(args.batch_sizes)
    revision = (
        args.source_revision
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": args.model,
        "source_revision": revision,
    }
    config = vars(args)
    rows: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []
    output = Path(args.output)

    if args.resume and output.exists():
        existing = json.loads(output.read_text())
        rows = existing["rows"]
        requests = existing["requests"]
        existing_environment = existing.get("environment", {})
        for key in ("gpu", "torch", "cuda", "model"):
            if existing_environment.get(key) != environment[key]:
                raise ValueError(
                    f"Cannot resume with different {key}: "
                    f"{existing_environment.get(key)!r} != {environment[key]!r}"
                )
        print(f"CHECKPOINT3_RESUME {len(rows)} rows", flush=True)

    completed = {(str(row["experiment"]), int(row["repeat"])) for row in rows}

    for experiment in experiments:
        missing_repeats = [
            repeat
            for repeat in range(args.repetitions)
            if (experiment.name, repeat) not in completed
        ]
        if not missing_repeats:
            continue
        load_profile = profile_for(experiment, model.config.vocab_size, args.seed)
        await warmup(model, tokenizer, experiment, args.seed, args.num_blocks)
        for repeat in missing_repeats:
            summary, request_rows = await run_experiment(
                model,
                tokenizer,
                experiment,
                load_profile,
                repeat,
                args.seed,
                args.num_blocks,
            )
            rows.append(summary)
            requests.extend(request_rows)
            print("CHECKPOINT3_ROW", json.dumps(summary, sort_keys=True), flush=True)
            write_checkpoint(output, environment, config, rows, requests, None)

    operator_counts = None
    if not args.skip_operator_profile:
        operator_counts = await operator_profile(model, tokenizer, args.seed, args.num_blocks)
    write_checkpoint(output, environment, config, rows, requests, operator_counts)
    artifacts = [
        output,
        output.with_suffix(".csv"),
        output.with_name(output.stem + "-requests.csv"),
    ]
    checksum_path = output.with_suffix(".sha256")
    write_checksums(artifacts, checksum_path)
    print("CHECKPOINT3_AGGREGATES", json.dumps(aggregate_rows(rows), sort_keys=True), flush=True)
    print("CHECKPOINT3_CHECKSUMS", checksum_path.read_text(), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
