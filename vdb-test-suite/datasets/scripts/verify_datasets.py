#!/usr/bin/env python3
"""Verify cached datasets: check dimensions, shapes, ground truth validity."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import yaml

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"
SUITE_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(SUITE_ROOT))
from framework.datasets import read_fvecs


def verify_manifest(manifest_path: Path) -> bool:
    with manifest_path.open() as f:
        m = yaml.safe_load(f)

    name = m["name"]
    dim = m["dim"]
    base_path = SUITE_ROOT / m["base_path"]
    query_path = SUITE_ROOT / m["query_path"]
    gt_path = SUITE_ROOT / m["gt_path"]

    print(f"\n{'='*40}")
    print(f"Dataset: {name} (dim={dim})")
    ok = True

    for label, path in [("base", base_path), ("query", query_path)]:
        if not path.exists():
            print(f"  [{label}] MISSING: {path}")
            ok = False
            continue
        vecs = read_fvecs(path)
        print(f"  [{label}] shape={vecs.shape}, dtype={vecs.dtype}")
        if vecs.shape[1] != dim:
            print(f"  [{label}] ERROR: expected dim={dim}, got {vecs.shape[1]}")
            ok = False

    if not gt_path.exists():
        print(f"  [gt] MISSING: {gt_path}")
        ok = False
    else:
        gt = np.load(gt_path)
        print(f"  [gt] shape={gt.shape}, dtype={gt.dtype}")
        if base_path.exists():
            n_base = read_fvecs(base_path).shape[0]
            if gt.size and int(gt.max()) >= n_base:
                print(f"  [gt] WARNING: max index {gt.max()} >= n_base {n_base}")

    status = "OK" if ok else "ERRORS"
    print(f"  Status: {status}")
    return ok


def main() -> None:
    manifests = sorted(MANIFEST_DIR.glob("*.yaml"))
    if not manifests:
        print("No manifests found in", MANIFEST_DIR)
        return

    all_ok = True
    for m in manifests:
        if not verify_manifest(m):
            all_ok = False

    print(f"\n{'='*40}")
    if all_ok:
        print("All datasets verified OK")
    else:
        print("Some datasets have issues")
        sys.exit(1)


if __name__ == "__main__":
    main()
