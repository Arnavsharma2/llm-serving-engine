from __future__ import annotations

import pytest

from llmserve.benchmark.metrics import BenchmarkReport, RequestMetrics, percentile
from llmserve.benchmark.workload import LengthDistribution, LoadProfile, generate_workload


def test_workload_is_deterministic_and_skewed() -> None:
    profile = LoadProfile(
        requests=20,
        arrival_rate=5,
        prompt=LengthDistribution(32, 0.8, 4, 256),
        output=LengthDistribution(16, 0.5, 2, 64),
        vocabulary_size=260,
        seed=19,
    )
    first = generate_workload(profile)
    second = generate_workload(profile)
    assert first == second
    assert all(
        left.arrival_offset_s <= right.arrival_offset_s
        for left, right in zip(first, first[1:], strict=False)
    )
    assert len({item.prompt_length for item in first}) > 5
    assert len({item.max_new_tokens for item in first}) > 3


def test_report_computes_latency_and_throughput() -> None:
    row = RequestMetrics(
        "r1",
        arrival_s=1.0,
        prompt_tokens=5,
        requested_output_tokens=3,
        first_token_s=1.1,
        finished_s=1.3,
        output_tokens=3,
    )
    report = BenchmarkReport("test", 1.0, 2.0, [row])
    assert row.ttft_ms == pytest.approx(100)
    assert round(row.tpot_ms) == 100
    assert report.summary["throughput_tokens_per_s"] == 3
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
