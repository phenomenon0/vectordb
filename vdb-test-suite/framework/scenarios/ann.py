from __future__ import annotations
import time

import numpy as np

from framework.clients.base import VDBClient
from framework.datasets import LoadedDataset
from framework.metrics import compute_recall, percentile_ms
from framework.memory import snapshot_rss_mb
from framework.disk import dir_size_mb
from framework.report import ScenarioResult


def run_ann_recall_scenario(
    client: VDBClient,
    dataset: LoadedDataset,
    collection: str = "bench",
    n_search: int = 200,
    warmup: int = 10,
    ef_search: int = 200,
    settle_time_s: float = 1.0,
    server_pid: int | None = None,
    data_dir: str | None = None,
) -> ScenarioResult:
    try:
        client.delete_collection(collection)
        client.create_collection(collection, dim=dataset.dim)

        ids = np.arange(1, len(dataset.base) + 1, dtype=np.uint64)
        t0 = time.perf_counter()
        client.insert(collection, ids, dataset.base)
        insert_s = time.perf_counter() - t0
        time.sleep(settle_time_s)

        rss_mb = snapshot_rss_mb(server_pid) if server_pid else 0.0
        disk_mb = dir_size_mb(data_dir) if data_dir else 0.0

        n_search = min(len(dataset.queries), n_search)
        warmup = min(warmup, n_search)
        latencies: list[float] = []
        recalls10: list[float] = []
        recalls100: list[float] = []

        for i in range(warmup):
            client.search(collection, dataset.queries[i], top_k=100, ef_search=ef_search)

        for i in range(warmup, n_search):
            t1 = time.perf_counter()
            res = client.search(collection, dataset.queries[i], top_k=100, ef_search=ef_search)
            latencies.append((time.perf_counter() - t1) * 1000.0)
            recalls10.append(compute_recall(res.ids, dataset.gt[i], 10))
            recalls100.append(compute_recall(res.ids, dataset.gt[i], 100))

        return ScenarioResult(
            suite="bench",
            scenario="ann_recall",
            vdb=client.name,
            dataset=dataset.name,
            success=True,
            metrics={
                "n_vectors": float(len(dataset.base)),
                "n_queries": float(n_search - warmup),
                "insert_qps": round(len(dataset.base) / insert_s, 1),
                "recall_at_10": round(float(np.mean(recalls10)), 4) if recalls10 else 0.0,
                "recall_at_100": round(float(np.mean(recalls100)), 4) if recalls100 else 0.0,
                "search_p50_ms": percentile_ms(latencies, 50),
                "search_p95_ms": percentile_ms(latencies, 95),
                "search_p99_ms": percentile_ms(latencies, 99),
                "rss_mb": round(rss_mb, 1),
                "disk_mb": round(disk_mb, 1),
            },
        )
    except Exception as e:
        return ScenarioResult(
            suite="bench",
            scenario="ann_recall",
            vdb=client.name,
            dataset=dataset.name,
            success=False,
            error=str(e),
        )
    finally:
        try:
            client.delete_collection(collection)
        except Exception:
            pass


def run_param_sweep_scenario(
    client: VDBClient,
    dataset: LoadedDataset,
    server_builder,
    m_values: list[int],
    efc_values: list[int],
    efs_values: list[int],
    n_search: int = 100,
    warmup: int = 10,
) -> list[ScenarioResult]:
    results = []
    for m in m_values:
        for efc in efc_values:
            server_builder.stop()
            server_builder.start(hnsw_m=m, ef_construction=efc, clean=True)
            client_new = type(client)(base_url=f"http://127.0.0.1:{server_builder.port}")
            try:
                coll = "sweep"
                client_new.delete_collection(coll)
                client_new.create_collection(coll, dim=dataset.dim)
                ids = np.arange(1, len(dataset.base) + 1, dtype=np.uint64)
                t0 = time.perf_counter()
                client_new.insert(coll, ids, dataset.base)
                build_s = time.perf_counter() - t0
                time.sleep(1.0)

                rss_mb = snapshot_rss_mb(server_builder.proc.pid) if server_builder.proc else 0.0

                for efs in efs_values:
                    latencies = []
                    recalls10 = []
                    n = min(len(dataset.queries), n_search)
                    w = min(warmup, n)

                    for i in range(w):
                        client_new.search(coll, dataset.queries[i], top_k=10, ef_search=efs)
                    for i in range(w, n):
                        t1 = time.perf_counter()
                        res = client_new.search(coll, dataset.queries[i], top_k=10, ef_search=efs)
                        latencies.append((time.perf_counter() - t1) * 1000.0)
                        recalls10.append(compute_recall(res.ids, dataset.gt[i], 10))

                    results.append(ScenarioResult(
                        suite="bench",
                        scenario="param_sweep",
                        vdb=client.name,
                        dataset=dataset.name,
                        success=True,
                        metrics={
                            "M": float(m),
                            "ef_construction": float(efc),
                            "ef_search": float(efs),
                            "recall_at_10": round(float(np.mean(recalls10)), 4) if recalls10 else 0.0,
                            "search_p95_ms": percentile_ms(latencies, 95),
                            "build_time_s": round(build_s, 2),
                            "rss_mb": round(rss_mb, 1),
                        },
                    ))
                client_new.delete_collection(coll)
            finally:
                client_new.close()
    return results
