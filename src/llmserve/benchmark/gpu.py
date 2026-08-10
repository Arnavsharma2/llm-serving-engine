from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSnapshot:
    memory_used_bytes: int
    utilization_pct: float


class GPUMonitor:
    """NVML sampler with a zero-valued fallback for CPU and non-NVIDIA systems."""

    def __init__(self, interval_s: float = 0.1, device_index: int = 0) -> None:
        self.interval_s = interval_s
        self.device_index = device_index
        self.samples: list[GPUSnapshot] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def __enter__(self) -> GPUMonitor:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except (ImportError, RuntimeError):
            self._nvml = None
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 2 + 0.1)
        if self._nvml:
            self._nvml.nvmlShutdown()

    def _sample_loop(self) -> None:
        assert self._nvml is not None
        while not self._stop.is_set():
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            self.samples.append(GPUSnapshot(memory.used, float(utilization.gpu)))
            self._stop.wait(self.interval_s)

    @property
    def peak_memory_bytes(self) -> int:
        return max((sample.memory_used_bytes for sample in self.samples), default=0)

    @property
    def mean_utilization_pct(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.utilization_pct for sample in self.samples) / len(self.samples)
