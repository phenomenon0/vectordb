from __future__ import annotations
from pathlib import Path


def dir_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def dir_size_mb(path: str | Path) -> float:
    return dir_size_bytes(path) / (1024 * 1024)


def bytes_per_vector(dir_path: str | Path, n_vectors: int) -> float:
    if n_vectors <= 0:
        return 0.0
    return dir_size_bytes(dir_path) / n_vectors
