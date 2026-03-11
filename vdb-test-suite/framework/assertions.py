from __future__ import annotations
from typing import Any


class ThresholdViolation(AssertionError):
    def __init__(self, metric: str, value: float, threshold: float, direction: str):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        super().__init__(
            f"{metric}: {value:.4f} {direction} threshold {threshold:.4f}"
        )


def assert_recall_above(recall: float, minimum: float, label: str = "recall") -> None:
    if recall < minimum:
        raise ThresholdViolation(label, recall, minimum, "below")


def assert_latency_below(latency_ms: float, maximum: float, label: str = "p95_ms") -> None:
    if latency_ms > maximum:
        raise ThresholdViolation(label, latency_ms, maximum, "above")


def assert_count_exact(actual: int, expected: int, label: str = "count") -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def assert_no_regression(
    current: float,
    baseline: float,
    max_regression_pct: float,
    label: str = "metric",
    higher_is_better: bool = True,
) -> None:
    if baseline == 0:
        return
    if higher_is_better:
        drop_pct = (baseline - current) / baseline * 100
    else:
        drop_pct = (current - baseline) / baseline * 100
    if drop_pct > max_regression_pct:
        raise ThresholdViolation(
            f"{label}_regression_pct", drop_pct, max_regression_pct, "above"
        )


def assert_ids_absent(search_ids: list[int], deleted_ids: set[int], label: str = "search") -> None:
    leaked = set(search_ids) & deleted_ids
    if leaked:
        raise AssertionError(f"{label}: deleted IDs leaked into results: {leaked}")
