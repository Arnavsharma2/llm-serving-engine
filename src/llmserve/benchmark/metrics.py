from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@dataclass
class RequestMetrics:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    requested_output_tokens: int
    admitted_s: float | None = None
    first_token_s: float | None = None
    finished_s: float | None = None
    output_tokens: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_s is None:
            return None
        return (self.first_token_s - self.arrival_s) * 1_000

    @property
    def tpot_ms(self) -> float | None:
        if self.first_token_s is None or self.finished_s is None or self.output_tokens < 2:
            return None
        return (self.finished_s - self.first_token_s) * 1_000 / (self.output_tokens - 1)

    @property
    def e2e_ms(self) -> float | None:
        if self.finished_s is None:
            return None
        return (self.finished_s - self.arrival_s) * 1_000

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(ttft_ms=self.ttft_ms, tpot_ms=self.tpot_ms, e2e_ms=self.e2e_ms)
        return result


@dataclass
class BenchmarkReport:
    config_name: str
    started_s: float
    finished_s: float
    requests: list[RequestMetrics]
    gpu_memory_peak_bytes: int = 0
    gpu_utilization_mean_pct: float = 0.0
    cache: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, object]:
        successful = [item for item in self.requests if item.error is None]
        ttft = [item.ttft_ms for item in successful if item.ttft_ms is not None]
        tpot = [item.tpot_ms for item in successful if item.tpot_ms is not None]
        e2e = [item.e2e_ms for item in successful if item.e2e_ms is not None]
        duration = max(self.finished_s - self.started_s, 1e-9)
        output_tokens = sum(item.output_tokens for item in successful)
        result: dict[str, object] = {
            "config": self.config_name,
            "requests": len(self.requests),
            "successful_requests": len(successful),
            "duration_s": duration,
            "throughput_tokens_per_s": output_tokens / duration,
            "gpu_memory_peak_gb": self.gpu_memory_peak_bytes / 1024**3,
            "gpu_utilization_mean_pct": self.gpu_utilization_mean_pct,
        }
        for name, values in (("ttft_ms", ttft), ("tpot_ms", tpot), ("e2e_ms", e2e)):
            result[f"{name}_p50"] = percentile(values, 0.50)
            result[f"{name}_p95"] = percentile(values, 0.95)
            result[f"{name}_p99"] = percentile(values, 0.99)
        result.update({f"cache_{key}": value for key, value in self.cache.items()})
        return result

    def write_json(self, path: str | Path) -> None:
        payload = {
            "summary": self.summary,
            "metadata": self.metadata,
            "requests": [item.to_dict() for item in self.requests],
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    def write_csv(self, path: str | Path) -> None:
        rows = [item.to_dict() for item in self.requests]
        if not rows:
            return
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def reports_table(reports: Iterable[BenchmarkReport]) -> list[dict[str, object]]:
    return [report.summary for report in reports]
