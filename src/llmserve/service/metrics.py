from __future__ import annotations

import time

try:
    from prometheus_client import Counter, Gauge, Histogram

    REQUESTS = Counter("llm_requests_total", "Completion requests", ["status", "stream"])
    OUTPUT_TOKENS = Counter("llm_output_tokens_total", "Generated output tokens")
    ACTIVE = Gauge("llm_active_requests", "Requests currently generating")
    TTFT = Histogram(
        "llm_time_to_first_token_seconds",
        "Arrival to first output token",
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    E2E = Histogram(
        "llm_request_duration_seconds",
        "End-to-end completion latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    )
except ImportError:  # The core engine remains importable without serving extras.
    REQUESTS = OUTPUT_TOKENS = ACTIVE = TTFT = E2E = None


class RequestObserver:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.first_token: float | None = None
        if ACTIVE:
            ACTIVE.inc()

    def token(self) -> None:
        if self.first_token is None:
            self.first_token = time.perf_counter()
            if TTFT:
                TTFT.observe(self.first_token - self.started)
        if OUTPUT_TOKENS:
            OUTPUT_TOKENS.inc()

    def finish(self, status: str, stream: bool) -> None:
        if ACTIVE:
            ACTIVE.dec()
        if E2E:
            E2E.observe(time.perf_counter() - self.started)
        if REQUESTS:
            REQUESTS.labels(status=status, stream=str(stream).lower()).inc()
