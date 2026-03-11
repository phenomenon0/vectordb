from __future__ import annotations

import numpy as np


def compute_recall(retrieved: list[int], gt_row: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    truth = set(int(x) for x in gt_row[:k])
    got = set(int(x) for x in retrieved[:k])
    return len(truth & got) / len(truth) if truth else 0.0


def percentile_ms(samples_ms: list[float], p: float) -> float:
    if not samples_ms:
        return 0.0
    return round(float(np.percentile(samples_ms, p)), 2)


def bytes_per_vector(total_bytes: int, n_vectors: int) -> float:
    if n_vectors <= 0:
        return 0.0
    return total_bytes / n_vectors
