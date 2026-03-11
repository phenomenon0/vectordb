from __future__ import annotations
import time

import numpy as np

from framework.clients.base import VDBClient
from framework.report import ScenarioResult


def run_persistence_scenario(
    client_factory,
    server,
    collection: str,
    dim: int = 128,
    n_vectors: int = 500,
    seed: int = 42,
) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    try:
        client = client_factory()
        try:
            client.delete_collection(collection)
            client.create_collection(collection, dim=dim)

            vecs = rng.normal(size=(n_vectors, dim)).astype(np.float32)
            vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
            ids = np.arange(1, n_vectors + 1, dtype=np.uint64)

            client.insert(collection, ids, vecs)
            before_count = client.count(collection)
            q = vecs[0]
            before_search = client.search(collection, q, top_k=10).ids
        finally:
            client.close()

        # Restart
        server.stop()
        server.start()
        time.sleep(0.5)

        client2 = client_factory()
        try:
            after_count = client2.count(collection)
            after_search = client2.search(collection, q, top_k=10).ids

            count_match = after_count == before_count
            top5_match = after_search[:5] == before_search[:5]

            return ScenarioResult(
                suite="tests",
                scenario="persistence",
                vdb="deepdata",
                dataset="synthetic",
                success=count_match and top5_match,
                metrics={
                    "before_count": float(before_count),
                    "after_count": float(after_count),
                    "count_match": 1.0 if count_match else 0.0,
                    "top5_match": 1.0 if top5_match else 0.0,
                },
            )
        finally:
            try:
                client2.delete_collection(collection)
            except Exception:
                pass
            client2.close()
    except Exception as e:
        return ScenarioResult(
            suite="tests",
            scenario="persistence",
            vdb="deepdata",
            dataset="synthetic",
            success=False,
            error=str(e),
        )
