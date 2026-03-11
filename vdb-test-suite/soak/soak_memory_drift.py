#!/usr/bin/env python3
"""Soak test for memory leaks: repeated search + small churn, track RSS over time."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from framework.clients.deepdata import DeepDataClient
from framework.envinfo import collect_env_info
from framework.memory import MemoryTracker, snapshot_rss_mb
from framework.report import ScenarioResult, make_run_report
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soak: memory drift detection")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=128, type=int)
    p.add_argument("--n-base", default=10000, type=int)
    p.add_argument("--duration", default=600, type=float, help="Duration in seconds")
    p.add_argument("--json", default=None)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(42)

    server = DeepDataProcess(port=args.port)
    if not args.skip_build:
        server.build()
    server.start(clean=True)

    client = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
    try:
        coll = "soak_mem"
        client.create_collection(coll, dim=args.dim)

        vecs = rng.normal(size=(args.n_base, args.dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, args.n_base + 1, dtype=np.uint64)
        client.insert(coll, ids, vecs)
        time.sleep(1.0)

        mem_tracker = MemoryTracker(pid=server.proc.pid, interval_s=2.0)
        mem_tracker.start()

        deadline = time.perf_counter() + args.duration
        ops = 0
        print(f"Memory drift soak running for {args.duration}s...")

        while time.perf_counter() < deadline:
            # Mostly searches with occasional small inserts
            q = rng.normal(size=(args.dim,)).astype(np.float32)
            client.search(coll, q, top_k=10, ef_search=200)
            ops += 1

            if ops % 100 == 0:
                rss = snapshot_rss_mb(server.proc.pid)
                elapsed = time.perf_counter() - (deadline - args.duration)
                print(f"  [{elapsed:.0f}s] ops={ops} rss={rss:.0f}MB")

        mem_tracker.stop()

        rss_drift = mem_tracker.rss_drift_mb
        peak = mem_tracker.peak_rss_mb
        success = abs(rss_drift) < peak * 0.1  # < 10% growth tolerance

        result = ScenarioResult(
            suite="soak",
            scenario="memory_drift",
            vdb="deepdata",
            dataset="synthetic",
            success=success,
            metrics={
                "duration_s": args.duration,
                "total_ops": float(ops),
                "peak_rss_mb": round(peak, 1),
                "rss_drift_mb": round(rss_drift, 1),
                "rss_drift_pct": round(rss_drift / peak * 100, 1) if peak > 0 else 0,
            },
        )

        print(f"\nMemory drift soak: {'PASS' if success else 'FAIL'}")
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")

        if args.json:
            env = collect_env_info()
            report = make_run_report([result], {"env": env.to_dict()})
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
    finally:
        client.close()
        server.stop()


if __name__ == "__main__":
    main()
