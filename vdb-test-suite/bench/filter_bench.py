#!/usr/bin/env python3
"""Benchmark vector search under metadata filters at various selectivities."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from framework.clients.deepdata import DeepDataClient
from framework.envinfo import collect_env_info
from framework.metrics import percentile_ms
from framework.report import ScenarioResult, make_run_report
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filtered search benchmark")
    p.add_argument("--port", default=8080, type=int)
    p.add_argument("--dim", default=128, type=int)
    p.add_argument("--n-vectors", default=50000, type=int)
    p.add_argument("--n-queries", default=200, type=int)
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
    results = []
    try:
        coll = "filter_bench"
        client.delete_collection(coll)
        client.create_collection(coll, dim=args.dim)

        vecs = rng.normal(size=(args.n_vectors, args.dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, args.n_vectors + 1, dtype=np.uint64)
        client.insert(coll, ids, vecs)
        time.sleep(1.0)

        queries = rng.normal(size=(args.n_queries, args.dim)).astype(np.float32)

        # Unfiltered baseline
        lats = []
        for q in queries:
            t0 = time.perf_counter()
            client.search(coll, q, top_k=10, ef_search=200)
            lats.append((time.perf_counter() - t0) * 1000.0)
        results.append(ScenarioResult(
            suite="bench", scenario="filter_none", vdb="deepdata",
            dataset="synthetic", success=True,
            metrics={"p50_ms": percentile_ms(lats, 50), "p95_ms": percentile_ms(lats, 95)},
        ))
        print(f"No filter:    p50={percentile_ms(lats, 50):.2f}ms  p95={percentile_ms(lats, 95):.2f}ms")

        # NOTE: Filtered search requires metadata-aware insert.
        # Once payload insert is supported, add low/medium/high selectivity filters here.
        print("(Filtered benchmarks require payload insert — skipped for now)")

        if args.json:
            env = collect_env_info()
            report = make_run_report(results, {"env": env.to_dict()})
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                json.dump(report.to_dict(), f, indent=2)
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        client.close()
        server.stop()


if __name__ == "__main__":
    main()
