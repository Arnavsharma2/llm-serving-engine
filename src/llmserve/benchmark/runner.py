from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable

from llmserve.benchmark.gpu import GPUMonitor
from llmserve.benchmark.metrics import BenchmarkReport, RequestMetrics
from llmserve.benchmark.workload import LoadProfile, RequestSpec, generate_workload

TokenCallback = Callable[[int], Awaitable[None] | None]
GenerateFunction = Callable[[RequestSpec, TokenCallback], Awaitable[list[int]]]


class BenchmarkHarness:
    """Deterministic, open-loop benchmark driver used by every engine configuration."""

    def __init__(self, profile: LoadProfile) -> None:
        self.profile = profile
        self.workload = generate_workload(profile)

    async def run(self, name: str, generate: GenerateFunction) -> BenchmarkReport:
        loop = asyncio.get_running_loop()
        benchmark_start = loop.time()
        rows: list[RequestMetrics] = []

        async def execute(spec: RequestSpec) -> None:
            delay = benchmark_start + spec.arrival_offset_s - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            row = RequestMetrics(
                request_id=spec.request_id,
                arrival_s=loop.time(),
                prompt_tokens=spec.prompt_length,
                requested_output_tokens=spec.max_new_tokens,
                admitted_s=loop.time(),
            )
            rows.append(row)

            async def on_token(_: int) -> None:
                now = loop.time()
                if row.first_token_s is None:
                    row.first_token_s = now
                row.output_tokens += 1

            try:
                result = generate(spec, on_token)
                if not inspect.isawaitable(result):
                    raise TypeError("generate must be async")
                await result
                row.finished_s = loop.time()
            except Exception as error:  # Preserve failures in the result instead of hiding them.
                row.error = f"{type(error).__name__}: {error}"
                row.finished_s = loop.time()

        with GPUMonitor() as gpu:
            tasks = [asyncio.create_task(execute(spec)) for spec in self.workload]
            await asyncio.gather(*tasks)
        finished = loop.time()
        rows.sort(key=lambda row: row.request_id)
        return BenchmarkReport(
            config_name=name,
            started_s=benchmark_start,
            finished_s=finished,
            requests=rows,
            gpu_memory_peak_bytes=gpu.peak_memory_bytes,
            gpu_utilization_mean_pct=gpu.mean_utilization_pct,
            metadata={"profile": self.profile.to_dict(), "wall_clock_unix_s": time.time()},
        )
