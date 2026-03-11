from __future__ import annotations
import time

import numpy as np

from framework.clients.base import VDBClient
from framework.metrics import percentile_ms
from framework.report import ScenarioResult
from framework.workload import MixedWorkloadConfig, OpType, WorkloadDriver


def run_mixed_workload_scenario(
    client: VDBClient,
    collection: str,
    dim: int = 128,
    n_seed: int = 10000,
    duration_s: float = 30.0,
    concurrency: int = 1,
    mix: MixedWorkloadConfig | None = None,
    seed: int = 42,
) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    if mix is None:
        mix = MixedWorkloadConfig(dim=dim)
    else:
        mix.dim = dim

    try:
        client.delete_collection(collection)
        client.create_collection(collection, dim=dim)

        # Seed initial data
        vecs = rng.normal(size=(n_seed, dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, n_seed + 1, dtype=np.uint64)
        client.insert(collection, ids, vecs)
        time.sleep(1.0)

        driver = WorkloadDriver(client=client, collection=collection, config=mix, rng=rng)
        driver.seed_ids(list(range(1, n_seed + 1)))
        driver._next_id = n_seed + 1

        results = driver.run_duration(duration_s, concurrency=concurrency)

        search_lat = [r.latency_ms for r in results if r.op == OpType.SEARCH and r.success]
        insert_lat = [r.latency_ms for r in results if r.op == OpType.INSERT and r.success]
        delete_lat = [r.latency_ms for r in results if r.op == OpType.DELETE and r.success]
        errors = [r for r in results if not r.success]

        total_ops = len(results)
        total_s = duration_s

        return ScenarioResult(
            suite="bench",
            scenario="mixed_workload",
            vdb=client.name,
            dataset="synthetic",
            success=len(errors) == 0,
            metrics={
                "total_ops": float(total_ops),
                "ops_per_sec": round(total_ops / total_s, 1),
                "search_count": float(len(search_lat)),
                "search_p50_ms": percentile_ms(search_lat, 50),
                "search_p95_ms": percentile_ms(search_lat, 95),
                "search_p99_ms": percentile_ms(search_lat, 99),
                "insert_count": float(len(insert_lat)),
                "insert_p95_ms": percentile_ms(insert_lat, 95),
                "delete_count": float(len(delete_lat)),
                "delete_p95_ms": percentile_ms(delete_lat, 95),
                "error_count": float(len(errors)),
            },
            details={"errors": [r.error for r in errors[:10]]},
        )
    except Exception as e:
        return ScenarioResult(
            suite="bench",
            scenario="mixed_workload",
            vdb=client.name,
            dataset="synthetic",
            success=False,
            error=str(e),
        )
    finally:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
