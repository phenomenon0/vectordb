#!/usr/bin/env python3
"""Download standard ANN benchmark datasets to the local cache."""
from __future__ import annotations
import gzip
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"

DATASETS = {
    "sift": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "archive": "sift.tar.gz",
        "files": {
            "sift_base.fvecs": "sift/sift_base.fvecs",
            "sift_query.fvecs": "sift/sift_query.fvecs",
            "sift_groundtruth.ivecs": "sift/sift_groundtruth.ivecs",
        },
    },
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already cached: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved to {dest}")


def extract_tar_gz(archive: Path, dest_dir: Path) -> None:
    print(f"  Extracting {archive}...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)


def ivecs_to_gt_npy(ivecs_path: Path, out_path: Path, k: int = 100) -> None:
    """Convert ivecs ground truth to numpy array."""
    with ivecs_path.open("rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    dim = data[0]
    rows = data.reshape(-1, dim + 1)
    gt = rows[:, 1:]
    gt_k = gt[:, :k] if gt.shape[1] >= k else gt
    np.save(out_path, gt_k)
    print(f"  Ground truth saved: {out_path} (shape {gt_k.shape})")


def main() -> None:
    for name, info in DATASETS.items():
        print(f"\nDataset: {name}")
        dataset_dir = CACHE_DIR / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        archive_path = dataset_dir / info["archive"]
        download_file(info["url"], archive_path)

        # Check if already extracted
        first_file = list(info["files"].values())[0]
        if not (CACHE_DIR / first_file).exists():
            extract_tar_gz(archive_path, CACHE_DIR)

        # Convert ground truth
        gt_ivecs = dataset_dir / "sift_groundtruth.ivecs"
        gt_npy = dataset_dir / "sift_gt100.npy"
        if gt_ivecs.exists() and not gt_npy.exists():
            ivecs_to_gt_npy(gt_ivecs, gt_npy)

    print("\nDone. Datasets cached in:", CACHE_DIR)


if __name__ == "__main__":
    main()
