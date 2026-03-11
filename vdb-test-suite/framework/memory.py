from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass(slots=True)
class MemorySample:
    timestamp: float
    rss_mb: float
    vms_mb: float


@dataclass
class MemoryTracker:
    pid: int
    interval_s: float = 1.0
    samples: list[MemorySample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _collect(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop.is_set():
            try:
                mem = proc.memory_info()
                self.samples.append(MemorySample(
                    timestamp=time.perf_counter(),
                    rss_mb=mem.rss / (1024 * 1024),
                    vms_mb=mem.vms / (1024 * 1024),
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self.interval_s)

    @property
    def peak_rss_mb(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.rss_mb for s in self.samples)

    @property
    def rss_drift_mb(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].rss_mb - self.samples[0].rss_mb


def snapshot_rss_mb(pid: int) -> float:
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0
