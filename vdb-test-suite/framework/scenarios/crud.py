from __future__ import annotations

import numpy as np

from framework.clients.base import VDBClient
from framework.report import ScenarioResult


def run_crud_scenario(
    client: VDBClient,
    collection: str,
    dim: int = 128,
    n_vectors: int = 1000,
    n_delete: int = 100,
    seed: int = 42,
) -> ScenarioResult:
    rng = np.random.default_rng(seed)
    try:
        client.delete_collection(collection)
        client.create_collection(collection, dim=dim)

        vecs = rng.normal(size=(n_vectors, dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, n_vectors + 1, dtype=np.uint64)

        client.insert(collection, ids, vecs)
        count_after_insert = client.count(collection)
        assert count_after_insert == n_vectors, f"count {count_after_insert} != {n_vectors}"

        to_delete = ids[:n_delete]
        client.delete_ids(collection, to_delete)
        count_after_delete = client.count(collection)
        expected = n_vectors - n_delete
        assert count_after_delete == expected, f"count {count_after_delete} != {expected}"

        q = vecs[n_delete]  # first non-deleted vector
        res = client.search(collection, q, top_k=10)
        deleted_set = set(int(x) for x in to_delete)
        leaked = set(res.ids) & deleted_set
        assert not leaked, f"deleted IDs leaked into search: {leaked}"

        return ScenarioResult(
            suite="tests",
            scenario="crud",
            vdb=client.name,
            dataset="synthetic",
            success=True,
            metrics={
                "n_inserted": float(n_vectors),
                "n_deleted": float(n_delete),
                "count_correct": 1.0,
                "no_deleted_leak": 1.0,
            },
        )
    except Exception as e:
        return ScenarioResult(
            suite="tests",
            scenario="crud",
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
