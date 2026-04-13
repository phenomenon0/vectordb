#!/usr/bin/env python3
"""Download and prepare real-world benchmark datasets for HNSW recall validation.

Datasets:
  - SIFT-10K/100K: 128d visual descriptors (ANN benchmark standard)
  - GloVe-25d/100d: Word embeddings from Stanford NLP
  - GIST-960d: High-dimensional visual descriptors (subset)

All datasets are normalized for cosine similarity and stored in .fvecs format.
Ground truth (brute-force top-100) is precomputed and cached.

Usage:
    python benchmarks/download_datasets.py              # Download all
    python benchmarks/download_datasets.py --dataset sift glove
"""

import argparse
import os
import struct
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

CACHE_DIR = Path.home() / ".vectordb_bench" / "dataset"


# ═══════════════════════════════════════════════════════════════════
# .fvecs format I/O
# ═══════════════════════════════════════════════════════════════════


def read_fvecs(path: str | Path, max_n: int | None = None) -> np.ndarray:
    """Read .fvecs format: each vector prefixed by int32 dimension."""
    vecs = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if not buf:
                break
            d = struct.unpack("<i", buf)[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32).copy()
            vecs.append(vec)
            if max_n and len(vecs) >= max_n:
                break
    return np.array(vecs, dtype=np.float32)


def read_ivecs(path: str | Path, max_n: int | None = None) -> np.ndarray:
    """Read .ivecs format: each vector prefixed by int32 dimension."""
    vecs = []
    with open(path, "rb") as f:
        while True:
            buf = f.read(4)
            if not buf:
                break
            d = struct.unpack("<i", buf)[0]
            vec = np.frombuffer(f.read(d * 4), dtype=np.int32).copy()
            vecs.append(vec)
            if max_n and len(vecs) >= max_n:
                break
    return np.array(vecs, dtype=np.int32)


def write_fvecs(path: str | Path, vecs: np.ndarray) -> None:
    """Write .fvecs format."""
    vecs = vecs.astype(np.float32)
    with open(path, "wb") as f:
        for vec in vecs:
            f.write(struct.pack("<i", len(vec)))
            f.write(vec.tobytes())


def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def compute_ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    """Brute-force cosine similarity ground truth (top-k indices)."""
    n_queries = len(queries)
    gt = np.zeros((n_queries, k), dtype=np.int32)
    print(f"  Computing brute-force ground truth ({n_queries} queries, k={k})...")
    for i in range(n_queries):
        sims = base @ queries[i]
        if k >= len(sims):
            top_k = np.argsort(-sims)[:k]
        else:
            top_k = np.argpartition(sims, -k)[-k:]
            top_k = top_k[np.argsort(-sims[top_k])]
        gt[i, :len(top_k)] = top_k
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n_queries} queries done")
    return gt


# ═══════════════════════════════════════════════════════════════════
# Download helpers
# ═══════════════════════════════════════════════════════════════════


def _download(url: str, dest: Path) -> None:
    """Download a file with progress."""
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} ...")

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size if total_size > 0 else 0
        print(f"\r  {pct:.1f}%", end="", flush=True)

    urlretrieve(url, dest, reporthook=_progress)
    print()


# ═══════════════════════════════════════════════════════════════════
# SIFT dataset
# ═══════════════════════════════════════════════════════════════════


def download_sift() -> None:
    """Download SIFT-10K (siftsmall) and SIFT-100K (first 100K of sift1M)."""
    sift_dir = CACHE_DIR / "sift"
    sift_dir.mkdir(parents=True, exist_ok=True)

    # ── SIFT-10K (siftsmall) ──
    small_dir = sift_dir / "siftsmall"
    if not (small_dir / "siftsmall_base.fvecs").exists():
        tarpath = sift_dir / "siftsmall.tar.gz"
        _download("ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz", tarpath)
        print("  Extracting siftsmall...")
        with tarfile.open(tarpath, "r:gz") as tar:
            tar.extractall(sift_dir)
    else:
        print("  SIFT-10K already extracted")

    # Normalize and save
    base_10k_path = sift_dir / "siftsmall_base_norm.fvecs"
    query_10k_path = sift_dir / "siftsmall_query_norm.fvecs"
    gt_10k_path = sift_dir / "siftsmall_gt100.npy"

    if not base_10k_path.exists():
        print("  Processing SIFT-10K...")
        base = read_fvecs(small_dir / "siftsmall_base.fvecs")
        queries = read_fvecs(small_dir / "siftsmall_query.fvecs")
        base = normalize_vectors(base)
        queries = normalize_vectors(queries)
        write_fvecs(base_10k_path, base)
        write_fvecs(query_10k_path, queries)
        gt = compute_ground_truth(base, queries, k=100)
        np.save(gt_10k_path, gt)
        print(f"  SIFT-10K ready: {len(base)} base, {len(queries)} queries")
    else:
        print("  SIFT-10K already processed")

    # ── SIFT-100K (from sift1M) ──
    base_100k_path = sift_dir / "sift_base_100k_norm.fvecs"
    query_100k_path = sift_dir / "sift_query_norm.fvecs"
    gt_100k_path = sift_dir / "sift_100k_gt100.npy"

    if not base_100k_path.exists():
        # Download full sift1M
        sift1m_dir = sift_dir / "sift"
        if not (sift1m_dir / "sift_base.fvecs").exists():
            tarpath = sift_dir / "sift.tar.gz"
            _download("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", tarpath)
            print("  Extracting sift1M (this may take a while)...")
            with tarfile.open(tarpath, "r:gz") as tar:
                tar.extractall(sift_dir)

        print("  Processing SIFT-100K (first 100K from sift1M)...")
        base = read_fvecs(sift1m_dir / "sift_base.fvecs", max_n=100_000)
        queries = read_fvecs(sift1m_dir / "sift_query.fvecs")
        base = normalize_vectors(base)
        queries = normalize_vectors(queries)
        write_fvecs(base_100k_path, base)
        write_fvecs(query_100k_path, queries)
        gt = compute_ground_truth(base, queries, k=100)
        np.save(gt_100k_path, gt)
        print(f"  SIFT-100K ready: {len(base)} base, {len(queries)} queries")
    else:
        print("  SIFT-100K already processed")


# ═══════════════════════════════════════════════════════════════════
# GloVe dataset
# ═══════════════════════════════════════════════════════════════════


def download_glove() -> None:
    """Download GloVe 6B word vectors (25d, 100d)."""
    glove_dir = CACHE_DIR / "glove"
    glove_dir.mkdir(parents=True, exist_ok=True)

    zippath = glove_dir / "glove.6B.zip"
    txt_50 = glove_dir / "glove.6B.50d.txt"
    txt_100 = glove_dir / "glove.6B.100d.txt"

    if not txt_50.exists() or not txt_100.exists():
        _download("https://nlp.stanford.edu/data/glove.6B.zip", zippath)
        print("  Extracting GloVe...")
        with zipfile.ZipFile(zippath, "r") as zf:
            for name in ["glove.6B.50d.txt", "glove.6B.100d.txt"]:
                if not (glove_dir / name).exists():
                    zf.extract(name, glove_dir)
    else:
        print("  GloVe text files already extracted")

    # Convert each dimension variant
    for dim in [50, 100]:
        _convert_glove(glove_dir, dim)


def _convert_glove(glove_dir: Path, dim: int) -> None:
    """Convert GloVe text format to normalized .fvecs + compute ground truth."""
    txt_path = glove_dir / f"glove.6B.{dim}d.txt"
    base_path = glove_dir / f"glove_{dim}d_base_norm.fvecs"
    query_path = glove_dir / f"glove_{dim}d_query_norm.fvecs"
    gt_path = glove_dir / f"glove_{dim}d_gt100.npy"

    if base_path.exists():
        print(f"  GloVe-{dim}d already processed")
        return

    print(f"  Converting GloVe-{dim}d text → fvecs...")
    vecs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vecs.append(vec)
    vecs = np.array(vecs, dtype=np.float32)
    print(f"  Loaded {len(vecs)} words")

    vecs = normalize_vectors(vecs)

    # Use first 1000 as queries, rest as base
    n_queries = 1000
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(vecs))
    query_idx = indices[:n_queries]
    base_idx = indices[n_queries:]

    queries = vecs[query_idx]
    base = vecs[base_idx]

    write_fvecs(base_path, base)
    write_fvecs(query_path, queries)

    # Ground truth (use min(100, n_base) for small sets)
    k = min(100, len(base))
    gt = compute_ground_truth(base, queries, k=k)
    np.save(gt_path, gt)
    print(f"  GloVe-{dim}d ready: {len(base)} base, {len(queries)} queries")


# ═══════════════════════════════════════════════════════════════════
# Code embeddings (already on disk)
# ═══════════════════════════════════════════════════════════════════


def prepare_code_embeddings() -> None:
    """Prepare code embeddings from existing live benchmark data."""
    import json

    src = Path(__file__).resolve().parent / "competitive" / "live" / "data"
    dest = CACHE_DIR / "code"
    dest.mkdir(parents=True, exist_ok=True)

    base_path = dest / "code_1536d_base_norm.fvecs"
    query_path = dest / "code_1536d_query_norm.fvecs"
    gt_path = dest / "code_1536d_gt100.npy"

    if base_path.exists():
        print("  Code embeddings already processed")
        return

    corpus_file = src / "collection_a.jsonl"
    gt_file = src / "ground_truth.jsonl"
    queries_file = src / "queries.jsonl"

    if not corpus_file.exists():
        print("  Code embeddings source not found, skipping")
        return

    print("  Loading code embeddings from collection_a.jsonl...")
    vecs = []
    with open(corpus_file) as f:
        for line in f:
            doc = json.loads(line)
            emb = doc.get("embedding")
            if emb:
                vecs.append(np.array(emb, dtype=np.float32))
    base = np.array(vecs, dtype=np.float32)
    base = normalize_vectors(base)

    print("  Loading code queries...")
    q_vecs = []
    with open(queries_file) as f:
        for line in f:
            doc = json.loads(line)
            emb = doc.get("embedding")
            if emb:
                q_vecs.append(np.array(emb, dtype=np.float32))
    # Use first 50 queries (matches ground_truth.jsonl)
    queries = np.array(q_vecs[:50], dtype=np.float32)
    queries = normalize_vectors(queries)

    write_fvecs(base_path, base)
    write_fvecs(query_path, queries)

    # Compute our own ground truth (brute-force)
    k = min(100, len(base))
    gt = compute_ground_truth(base, queries, k=k)
    np.save(gt_path, gt)
    print(f"  Code embeddings ready: {len(base)} base, {len(queries)} queries")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


ALL_DATASETS = ["sift", "glove", "code"]


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--dataset", nargs="*", default=ALL_DATASETS,
                        choices=ALL_DATASETS, help="Datasets to download")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    for ds in args.dataset:
        print(f"{'=' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 60}")
        if ds == "sift":
            download_sift()
        elif ds == "glove":
            download_glove()
        elif ds == "code":
            prepare_code_embeddings()
        print()

    print("Done! All datasets cached to:", CACHE_DIR)


if __name__ == "__main__":
    main()
