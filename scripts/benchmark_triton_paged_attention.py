from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from pathlib import Path

import torch

from llmserve.benchmark.gpu import GPUMonitor
from llmserve.checkpoint import load_qwen2
from llmserve.config import CacheConfig, SchedulerConfig
from llmserve.engine import GenerationConfig, LLMEngine
from llmserve.kv_cache import PagedKVCache
from llmserve.tokenizer import HuggingFaceTokenizer
from llmserve.triton_kernels import is_triton_paged_attention_available


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": _percentile(values, 0.95),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


@torch.inference_mode()
def _microbenchmark(
    model,
    *,
    backend: str,
    batch_size: int,
    sequence_length: int,
    repeats: int,
    iterations: int,
) -> list[dict[str, float | int | str]]:
    attention = model.layers[0].self_attn
    cache = PagedKVCache(
        model.config,
        CacheConfig(block_size=16, num_blocks=max(64, batch_size * 16)),
        device="cuda",
        dtype=torch.float16,
    )
    sequence_ids = [f"micro-{index}" for index in range(batch_size)]
    slots = []
    for sequence_id in sequence_ids:
        cache.create(sequence_id)
        slot = None
        for _ in range(sequence_length):
            slot = cache.reserve_token(sequence_id)
        assert slot is not None
        slots.append(slot)
    cache.storage[0].normal_(mean=0.0, std=0.02)
    block_tables, sequence_lengths = cache.block_table_tensors(sequence_ids)
    generator = torch.Generator(device="cuda").manual_seed(101 + batch_size)
    hidden = torch.randn(
        batch_size,
        1,
        model.config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    positions = torch.full((batch_size,), sequence_length - 1, device="cuda", dtype=torch.long)

    def run_once() -> None:
        attention.forward_paged(
            hidden,
            positions,
            sequence_ids,
            slots,
            cache,
            backend=backend,
            block_tables=block_tables if backend == "triton" else None,
            sequence_lengths=sequence_lengths if backend == "triton" else None,
        )

    for _ in range(5):
        run_once()
    torch.cuda.synchronize()

    rows = []
    for repeat in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with GPUMonitor(interval_s=0.01) as gpu:
            start.record()
            for _ in range(iterations):
                run_once()
            end.record()
            end.synchronize()
        elapsed_ms = start.elapsed_time(end) / iterations
        rows.append(
            {
                "kind": "micro",
                "backend": backend,
                "batch_size": batch_size,
                "repeat": repeat,
                "sequence_length": sequence_length,
                "attention_ms": elapsed_ms,
                "gpu_utilization_pct": gpu.mean_utilization_pct,
            }
        )
    return rows


async def _warmup_engine(model, tokenizer, backend: str) -> None:
    engine = LLMEngine(
        model,
        tokenizer,
        cache_config=CacheConfig(block_size=16, num_blocks=64),
        scheduler_config=SchedulerConfig(max_batch_size=1),
        paged_attention_backend=backend,
        device="cuda",
        dtype=torch.float16,
    )
    await engine.generate(
        [11, 22], GenerationConfig(max_new_tokens=1, seed=7), request_id=f"warm-{backend}"
    )
    await engine.close()


async def _end_to_end_run(
    model,
    tokenizer,
    *,
    backend: str,
    batch_size: int,
    repeat: int,
    prompts: list[list[int]],
    output_tokens: int,
) -> dict[str, float | int | str]:
    engine = LLMEngine(
        model,
        tokenizer,
        cache_config=CacheConfig(block_size=16, num_blocks=256),
        scheduler_config=SchedulerConfig(max_batch_size=batch_size),
        paged_attention_backend=backend,
        device="cuda",
        dtype=torch.float16,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    with GPUMonitor(interval_s=0.02) as gpu:
        outputs = await asyncio.gather(
            *[
                engine.generate(
                    prompt,
                    GenerationConfig(max_new_tokens=output_tokens, seed=7),
                    request_id=f"{backend}-b{batch_size}-r{repeat}-{index}",
                )
                for index, prompt in enumerate(prompts[:batch_size])
            ]
        )
        torch.cuda.synchronize()
    duration_s = time.perf_counter() - start
    await engine.close()
    generated_tokens = sum(len(output) for output in outputs)
    return {
        "kind": "end_to_end",
        "backend": backend,
        "batch_size": batch_size,
        "repeat": repeat,
        "prompt_tokens": len(prompts[0]),
        "output_tokens_per_request": output_tokens,
        "duration_s": duration_s,
        "tokens_per_second": generated_tokens / duration_s,
        "gpu_utilization_pct": gpu.mean_utilization_pct,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--micro-iterations", type=int, default=20)
    parser.add_argument("--micro-sequence-length", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/checkpoint2-t4.json")
    args = parser.parse_args()

    if args.repeats < 5:
        raise ValueError("at least five repetitions are required for performance reporting")
    if not is_triton_paged_attention_available("cuda"):
        raise RuntimeError("a CUDA GPU with Triton and compute capability 7.5+ is required")

    torch.manual_seed(args.seed)
    model = load_qwen2(args.model, device="cuda", dtype=torch.float16)
    tokenizer = HuggingFaceTokenizer(args.model)
    rng = random.Random(args.seed)
    prompts = [
        [rng.randrange(4, model.config.vocab_size) for _ in range(args.prompt_tokens)]
        for _ in range(max(args.batch_sizes))
    ]

    for backend in ("pytorch", "triton"):
        await _warmup_engine(model, tokenizer, backend)
    rows: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for backend in ("pytorch", "triton"):
            rows.extend(
                _microbenchmark(
                    model,
                    backend=backend,
                    batch_size=batch_size,
                    sequence_length=args.micro_sequence_length,
                    repeats=args.repeats,
                    iterations=args.micro_iterations,
                )
            )
            for repeat in range(args.repeats):
                row = await _end_to_end_run(
                    model,
                    tokenizer,
                    backend=backend,
                    batch_size=batch_size,
                    repeat=repeat,
                    prompts=prompts,
                    output_tokens=args.output_tokens,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)

    summaries = []
    for kind in ("micro", "end_to_end"):
        metric = "attention_ms" if kind == "micro" else "tokens_per_second"
        for batch_size in args.batch_sizes:
            for backend in ("pytorch", "triton"):
                selected = [
                    float(row[metric])
                    for row in rows
                    if row["kind"] == kind
                    and row["batch_size"] == batch_size
                    and row["backend"] == backend
                ]
                utilization = [
                    float(row["gpu_utilization_pct"])
                    for row in rows
                    if row["kind"] == kind
                    and row["batch_size"] == batch_size
                    and row["backend"] == backend
                ]
                summaries.append(
                    {
                        "kind": kind,
                        "backend": backend,
                        "batch_size": batch_size,
                        "metric": metric,
                        **_summary(selected),
                        "gpu_utilization_median_pct": statistics.median(utilization),
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model": args.model,
        },
        "config": vars(args),
        "summaries": summaries,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(output.with_suffix(".csv"), rows)
    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
