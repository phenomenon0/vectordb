#!/usr/bin/env python3
"""Generate synthetic realistic-dimension corpus with metadata payloads."""
from __future__ import annotations
import argparse
import json
import random
import struct
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "realistic"

DOC_TYPES = ["report", "email", "code", "note", "spec", "readme", "changelog"]
LANGUAGES = ["en", "es", "fr", "de", "zh", "ja", "ko"]
TEAMS = list(range(1, 21))


def write_fvecs(path: Path, vecs: np.ndarray) -> None:
    n, dim = vecs.shape
    with path.open("wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", dim))
            f.write(vecs[i].tobytes())
    print(f"  Wrote {path} ({n} vectors, dim={dim})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build realistic synthetic corpus")
    parser.add_argument("--dim", default=1536, type=int)
    parser.add_argument("--n-base", default=100000, type=int)
    parser.add_argument("--n-queries", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    prefix = "text" if args.dim <= 768 else "code"

    # Base vectors
    print("Generating base vectors...")
    base = rng.normal(size=(args.n_base, args.dim)).astype(np.float32)
    base /= np.clip(np.linalg.norm(base, axis=1, keepdims=True), 1e-12, None)
    write_fvecs(CACHE_DIR / f"{prefix}_base_{args.dim}.fvecs", base)

    # Query vectors
    print("Generating query vectors...")
    queries = rng.normal(size=(args.n_queries, args.dim)).astype(np.float32)
    queries /= np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    write_fvecs(CACHE_DIR / f"{prefix}_query_{args.dim}.fvecs", queries)

    # Ground truth
    print("Computing ground truth (cosine)...")
    sims = queries @ base.T
    k = min(100, args.n_base)
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    row_ids = np.arange(len(queries))[:, None]
    vals = sims[row_ids, idx]
    order = np.argsort(-vals, axis=1)
    gt = idx[row_ids, order]
    gt_path = CACHE_DIR / f"{prefix}_gt100.npy"
    np.save(gt_path, gt)
    print(f"  Wrote {gt_path} (shape {gt.shape})")

    # Payloads
    print("Generating payloads...")
    payload_path = CACHE_DIR / f"{prefix}_payloads.jsonl"
    with payload_path.open("w") as f:
        for i in range(args.n_base):
            payload = {
                "tenant_id": random.choice(TEAMS),
                "doc_type": random.choice(DOC_TYPES),
                "language": random.choice(LANGUAGES),
                "word_count": random.randint(50, 5000),
                "created_epoch": 1700000000 + random.randint(0, 10000000),
            }
            f.write(json.dumps(payload) + "\n")
    print(f"  Wrote {payload_path}")

    print("\nDone. Realistic corpus in:", CACHE_DIR)


if __name__ == "__main__":
    main()
