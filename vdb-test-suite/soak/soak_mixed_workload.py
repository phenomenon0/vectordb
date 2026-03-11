#!/usr/bin/env python3
"""Long-running mixed workload soak: detect crashes, memory leaks, latency drift."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from framework.clients.deepdata import DeepDataClient
from framework.envinfo import collect_env_info
from framework.memory import MemoryTracker, snapshot_rss_mb
from framework.metrics import compute_recall, percentile_ms
from framework.report import ScenarioResult, make_run_report
from framework.server.deepdata_process import DeepDataProcess
from framework.workload import MixedWorkloadConfig, OpType, WorkloadDriver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soak: mixed workload stability test")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=128, type=int)
    p.add_argument("--n-seed", default=10000, type=int)
    p.add_argument("--duration", default=1800, type=float, help="Duration in seconds")
    p.add_argument("--sample-interval", default=5, type=float, help="Metric sample interval")
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
        coll = "soak_mixed"
        client.delete_collection(coll)
        client.create_collection(coll, dim=args.dim)

        vecs = rng.normal(size=(args.n_seed, args.dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, args.n_seed + 1, dtype=np.uint64)
        client.insert(coll, ids, vecs)
        time.sleep(1.0)

        mix = MixedWorkloadConfig(
            search_pct=70, insert_pct=20, delete_pct=10, dim=args.dim,
        )
        driver = WorkloadDriver(client=client, collection=coll, config=mix, rng=rng)
        driver.seed_ids(list(range(1, args.n_seed + 1)))
        driver._next_id = args.n_seed + 1

        # Start memory tracking
        mem_tracker = MemoryTracker(pid=server.proc.pid, interval_s=args.sample_interval)
        mem_tracker.start()

        # Run the soak
        rss_samples = []
        latency_windows: list[list[float]] = []
        window_lats: list[float] = []
        deadline = time.perf_counter() + args.duration
        window_start = time.perf_counter()
        total_ops = 0
        total_errors = 0

        print(f"Soak running for {args.duration}s...")
        while time.perf_counter() < deadline:
            result = driver.run_one()
            total_ops += 1
            if not result.success:
                total_errors += 1
            if result.op == OpType.SEARCH:
                window_lats.append(result.latency_ms)

            # Sample every interval
            if time.perf_counter() - window_start >= args.sample_interval:
                if window_lats:
                    latency_windows.append(window_lats)
                    p95 = percentile_ms(window_lats, 95)
                    rss = snapshot_rss_mb(server.proc.pid)
                    rss_samples.append(rss)
                    elapsed = time.perf_counter() - (deadline - args.duration)
                    print(f"  [{elapsed:.0f}s] ops={total_ops} errors={total_errors} "
                          f"search_p95={p95:.1f}ms rss={rss:.0f}MB")
                window_lats = []
                window_start = time.perf_counter()

        mem_tracker.stop()

        # Analyze results
        all_search_lats = [l for w in latency_windows for l in w]
        early_p95 = percentile_ms(latency_windows[0], 95) if latency_windows else 0
        late_p95 = percentile_ms(latency_windows[-1], 95) if latency_windows else 0

        rss_drift = 0.0
        if len(rss_samples) >= 2:
            rss_drift = rss_samples[-1] - rss_samples[0]

        latency_drift_pct = 0.0
        if early_p95 > 0:
            latency_drift_pct = (late_p95 - early_p95) / early_p95 * 100

        crashed = server.proc.poll() is not None

        success = (
            not crashed
            and total_errors == 0
            and latency_drift_pct < 25.0
        )

        result = ScenarioResult(
            suite="soak",
            scenario="mixed_workload",
            vdb="deepdata",
            dataset="synthetic",
            success=success,
            metrics={
                "duration_s": args.duration,
                "total_ops": float(total_ops),
                "total_errors": float(total_errors),
                "search_p50_ms": percentile_ms(all_search_lats, 50),
                "search_p95_ms": percentile_ms(all_search_lats, 95),
                "search_p99_ms": percentile_ms(all_search_lats, 99),
                "early_p95_ms": early_p95,
                "late_p95_ms": late_p95,
                "latency_drift_pct": round(latency_drift_pct, 1),
                "peak_rss_mb": round(mem_tracker.peak_rss_mb, 1),
                "rss_drift_mb": round(rss_drift, 1),
                "crashed": 1.0 if crashed else 0.0,
            },
        )

        print(f"\nSoak complete: {'PASS' if success else 'FAIL'}")
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
