from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class LoadedDataset:
    name: str
    dim: int
    base: np.ndarray
    queries: np.ndarray
    gt: np.ndarray


@dataclass(slots=True)
class DatasetManifest:
    name: str
    dim: int
    base_path: Path
    query_path: Path
    gt_path: Path
    n_base: int = 0


def read_fvecs(path: Path, max_n: int | None = None) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        first_dim = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_dim) == 0:
            return np.empty((0, 0), dtype=np.float32)
        dim = int(first_dim[0])
        f.seek(0)
        data = np.fromfile(f, dtype=np.int32)

    row_width = dim + 1
    if len(data) % row_width != 0:
        raise ValueError(f"Invalid fvecs file shape for {path}")
    rows = data.reshape(-1, row_width)
    dims = rows[:, 0]
    if not np.all(dims == dim):
        raise ValueError(f"Non-uniform dimension in {path}")
    vecs = rows[:, 1:].view(np.float32)
    if max_n is not None:
        vecs = vecs[:max_n]
    return np.ascontiguousarray(vecs, dtype=np.float32)


def compute_ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    base_norm = base / np.clip(np.linalg.norm(base, axis=1, keepdims=True), 1e-12, None)
    query_norm = queries / np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    sims = query_norm @ base_norm.T
    idx = np.argpartition(-sims, kth=min(k - 1, sims.shape[1] - 1), axis=1)[:, :k]
    row_ids = np.arange(len(queries))[:, None]
    vals = sims[row_ids, idx]
    order = np.argsort(-vals, axis=1)
    return idx[row_ids, order]


def load_dataset(manifest: DatasetManifest) -> LoadedDataset:
    base = read_fvecs(manifest.base_path, max_n=manifest.n_base if manifest.n_base > 0 else None)
    queries = read_fvecs(manifest.query_path)
    gt = np.load(manifest.gt_path)

    if manifest.n_base > 0 and len(base) > manifest.n_base:
        base = base[:manifest.n_base]

    if gt.size and int(gt.max()) >= len(base):
        gt = compute_ground_truth(base, queries, k=min(100, len(base)))

    if base.shape[1] != manifest.dim or queries.shape[1] != manifest.dim:
        raise ValueError(
            f"Dimension mismatch for {manifest.name}: expected {manifest.dim}, "
            f"got base={base.shape[1]} queries={queries.shape[1]}"
        )

    return LoadedDataset(
        name=manifest.name,
        dim=manifest.dim,
        base=base,
        queries=queries,
        gt=gt,
    )
