#!/usr/bin/env python3
"""Pure ingest benchmark: measure insert throughput at various batch sizes and collection states."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from framework.clients.deepdata import DeepDataClient
from framework.config import RunConfig
from framework.disk import dir_size_mb
from framework.envinfo import collect_env_info
from framework.memory import snapshot_rss_mb
from framework.metrics import percentile_ms
from framework.report import ScenarioResult, make_run_report
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest throughput benchmark")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=128, type=int)
    p.add_argument("--n-vectors", default=100000, type=int)
    p.add_argument("--batch-sizes", default="100,500,1000,5000,10000", help="Comma-separated")
    p.add_argument("--json", default=None)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def bench_ingest(
    client: DeepDataClient,
    collection: str,
    dim: int,
    n_vectors: int,
    batch_size: int,
    server_pid: int,
    data_dir: str,
) -> ScenarioResult:
    rng = np.random.default_rng(42)
    try:
        client.delete_collection(collection)
        client.create_collection(collection, dim=dim)

        vecs = rng.normal(size=(n_vectors, dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

        batch_latencies = []
        total_inserted = 0
        t_start = time.perf_counter()

        for start in range(0, n_vectors, batch_size):
            end = min(start + batch_size, n_vectors)
            batch_vecs = vecs[start:end]
            batch_ids = np.arange(start + 1, end + 1, dtype=np.uint64)

            t0 = time.perf_counter()
            client.insert(collection, batch_ids, batch_vecs)
            batch_latencies.append((time.perf_counter() - t0) * 1000.0)
            total_inserted += len(batch_ids)

        total_s = time.perf_counter() - t_start
        rss = snapshot_rss_mb(server_pid)
        disk = dir_size_mb(data_dir)

        return ScenarioResult(
            suite="bench",
            scenario="ingest",
            vdb="deepdata",
            dataset=f"synthetic_{dim}d",
            success=True,
            metrics={
                "n_vectors": float(n_vectors),
                "batch_size": float(batch_size),
                "dim": float(dim),
                "total_time_s": round(total_s, 2),
                "vectors_per_sec": round(n_vectors / total_s, 1),
                "batch_p50_ms": percentile_ms(batch_latencies, 50),
                "batch_p95_ms": percentile_ms(batch_latencies, 95),
                "rss_mb": round(rss, 1),
                "disk_mb": round(disk, 1),
            },
        )
    except Exception as e:
        return ScenarioResult(
            suite="bench",
            scenario="ingest",
            vdb="deepdata",
            dataset=f"synthetic_{dim}d",
            success=False,
            error=str(e),
        )
    finally:
        try:
            client.delete_collection(collection)
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    server = DeepDataProcess(port=args.port)
    if not args.skip_build:
        server.build()
    server.start(clean=True)

    client = DeepDataClient(base_url=f"http://127.0.0.1:{args.port}")
    results = []
    try:
        for bs in batch_sizes:
            print(f"Benchmarking batch_size={bs}...")
            r = bench_ingest(
                client, "ingest_bench", args.dim, args.n_vectors, bs,
                server.proc.pid, str(server.data_dir),
            )
            results.append(r)
            if r.success:
                m = r.metrics
                print(f"  {m['vectors_per_sec']:.0f} vec/s, "
                      f"batch p95={m['batch_p95_ms']:.1f}ms, "
                      f"RSS={m['rss_mb']:.0f}MB, disk={m['disk_mb']:.0f}MB")
            else:
                print(f"  FAILED: {r.error}")

        if args.json:
            env = collect_env_info()
            report = make_run_report(results, {"env": env.to_dict()})
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
    finally:
        client.close()
        server.stop()


if __name__ == "__main__":
    main()
