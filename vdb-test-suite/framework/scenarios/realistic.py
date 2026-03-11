from __future__ import annotations
import random
import time

import numpy as np

from framework.clients.base import VDBClient
from framework.metrics import percentile_ms
from framework.report import ScenarioResult


def run_realistic_app_scenario(
    client: VDBClient,
    collection: str,
    dim: int = 1536,
    n_base: int = 50000,
    n_queries: int = 500,
    top_k_values: list[int] | None = None,
    warmup: int = 20,
    ef_search: int = 200,
    seed: int = 42,
) -> ScenarioResult:
    """
    Simulate a realistic RAG/search application:
    - High-dimensional embeddings (768 or 1536)
    - Mixed top_k values
    - Repeated queries (cache hit simulation)
    - Query bursts
    """
    rng = np.random.default_rng(seed)
    if top_k_values is None:
        top_k_values = [10, 20, 50]

    try:
        client.delete_collection(collection)
        client.create_collection(collection, dim=dim)

        # Generate base vectors
        vecs = rng.normal(size=(n_base, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.clip(norms, 1e-12, None)
        ids = np.arange(1, n_base + 1, dtype=np.uint64)

        t0 = time.perf_counter()
        client.insert(collection, ids, vecs)
        ingest_s = time.perf_counter() - t0
        time.sleep(1.0)

        # Build query pool (some repeated to simulate cache patterns)
        query_pool = rng.normal(size=(n_queries, dim)).astype(np.float32)
        query_pool /= np.clip(np.linalg.norm(query_pool, axis=1, keepdims=True), 1e-12, None)

        # Warmup
        for i in range(min(warmup, len(query_pool))):
            client.search(collection, query_pool[i], top_k=10, ef_search=ef_search)

        # Run mixed queries
        latencies_by_k: dict[int, list[float]] = {k: [] for k in top_k_values}

        for i in range(warmup, len(query_pool)):
            top_k = random.choice(top_k_values)
            # 20% chance of repeating a recent query
            if i > warmup + 5 and random.random() < 0.2:
                qi = random.randint(warmup, i - 1)
            else:
                qi = i
            t1 = time.perf_counter()
            client.search(collection, query_pool[qi], top_k=top_k, ef_search=ef_search)
            latencies_by_k[top_k].append((time.perf_counter() - t1) * 1000.0)

        metrics: dict[str, float] = {
            "n_base": float(n_base),
            "dim": float(dim),
            "ingest_qps": round(n_base / ingest_s, 1),
        }
        for k, lats in latencies_by_k.items():
            if lats:
                metrics[f"top{k}_p50_ms"] = percentile_ms(lats, 50)
                metrics[f"top{k}_p95_ms"] = percentile_ms(lats, 95)
                metrics[f"top{k}_count"] = float(len(lats))

        return ScenarioResult(
            suite="bench",
            scenario="realistic_app",
            vdb=client.name,
            dataset="synthetic_realistic",
            success=True,
            metrics=metrics,
        )
    except Exception as e:
        return ScenarioResult(
            suite="bench",
            scenario="realistic_app",
            vdb=client.name,
            dataset="synthetic_realistic",
            success=False,
            error=str(e),
        )
    finally:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
