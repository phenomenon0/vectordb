from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class TimedResult:
    elapsed_s: float = 0.0


@contextmanager
def timed():
    result = TimedResult()
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = time.perf_counter() - t0


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"
