#!/usr/bin/env python3
from __future__ import annotations
import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from framework.clients.deepdata import DeepDataClient
from framework.config import RunConfig
from framework.datasets import DatasetManifest, load_dataset
from framework.metrics import compute_recall, percentile_ms
from framework.report import ScenarioResult
from framework.runner import ScenarioRunner
from framework.server.deepdata_process import DeepDataProcess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANN recall benchmark for DeepData")
    parser.add_argument("--base", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--n-base", default=0, type=int)
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--n-search", default=200, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--json", default=None)
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(port=args.port, n_search=args.n_search, warmup=args.warmup)
    manifest = DatasetManifest(
        name=args.dataset_name,
        dim=args.dim,
        base_path=Path(args.base),
        query_path=Path(args.queries),
        gt_path=Path(args.gt),
        n_base=args.n_base,
    )
    ds = load_dataset(manifest)

    server = DeepDataProcess(port=cfg.port)
    if not args.skip_build:
        server.build()
    server.start()
    runner = ScenarioRunner(config=asdict(cfg))

    def scenario() -> ScenarioResult:
        client = DeepDataClient(base_url=f"http://127.0.0.1:{cfg.port}")
        coll = "bench"
        n_search = min(len(ds.queries), cfg.n_search)
        warmup = min(cfg.warmup, n_search)
        try:
            client.delete_collection(coll)
            client.create_collection(coll, dim=ds.dim)
            ids = np.arange(len(ds.base), dtype=np.uint64)
            t0 = time.perf_counter()
            client.insert(coll, ids, ds.base)
            insert_s = time.perf_counter() - t0
            time.sleep(cfg.settle_time_s)

            latencies = []
            recalls10 = []
            recalls100 = []

            for i in range(warmup):
                client.search(coll, ds.queries[i], top_k=100, ef_search=200)

            for i in range(warmup, n_search):
                t1 = time.perf_counter()
                res = client.search(coll, ds.queries[i], top_k=100, ef_search=200)
                latencies.append((time.perf_counter() - t1) * 1000.0)
                recalls10.append(compute_recall(res.ids, ds.gt[i], 10))
                recalls100.append(compute_recall(res.ids, ds.gt[i], 100))

            return ScenarioResult(
                suite="bench",
                scenario="ann_recall",
                vdb="deepdata",
                dataset=ds.name,
                success=True,
                metrics={
                    "n_vectors": float(len(ds.base)),
                    "n_queries": float(n_search),
                    "insert_qps": round(len(ds.base) / insert_s, 1),
                    "recall_at_10": round(float(np.mean(recalls10)), 4) if recalls10 else 0.0,
                    "recall_at_100": round(float(np.mean(recalls100)), 4) if recalls100 else 0.0,
                    "search_p50_ms": percentile_ms(latencies, 50),
                    "search_p95_ms": percentile_ms(latencies, 95),
                    "search_p99_ms": percentile_ms(latencies, 99),
                },
            )
        except Exception as e:
            return ScenarioResult(
                suite="bench",
                scenario="ann_recall",
                vdb="deepdata",
                dataset=ds.name,
                success=False,
                error=str(e),
            )
        finally:
            client.delete_collection(coll)
            client.close()

    result = runner.run(scenario)
    print(result)
    if args.json:
        runner.save_json(args.json)
    server.stop()


if __name__ == "__main__":
    main()
