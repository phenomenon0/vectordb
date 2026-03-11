from __future__ import annotations

import numpy as np

from framework.clients.base import VDBClient
from framework.report import ScenarioResult


def run_filter_scenario(
    client: VDBClient,
    collection: str,
    dim: int = 128,
    n_vectors: int = 5000,
    seed: int = 42,
) -> ScenarioResult:
    """Test search with metadata filters at various selectivities."""
    rng = np.random.default_rng(seed)
    try:
        client.delete_collection(collection)
        client.create_collection(
            collection,
            dim=dim,
            metadata_schema={"tenant_id": "int", "doc_type": "string"},
        )

        vecs = rng.normal(size=(n_vectors, dim)).astype(np.float32)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        ids = np.arange(1, n_vectors + 1, dtype=np.uint64)

        # Insert with payloads - requires payload-aware insert
        tenants = (rng.integers(0, 10, size=n_vectors)).tolist()
        doc_types = [["report", "email", "code", "note"][i % 4] for i in range(n_vectors)]
        payloads = [{"tenant_id": t, "doc_type": d} for t, d in zip(tenants, doc_types)]

        client.insert(collection, ids, vecs, payloads=payloads)

        q = rng.normal(size=(dim,)).astype(np.float32)

        # Low selectivity filter (10% match)
        res_low = client.search(collection, q, top_k=10, filters={"tenant_id": {"$eq": 0}})
        # Impossible filter
        res_impossible = client.search(collection, q, top_k=10, filters={"tenant_id": {"$eq": 999}})

        return ScenarioResult(
            suite="bench",
            scenario="filter_bench",
            vdb=client.name,
            dataset="synthetic",
            success=True,
            metrics={
                "low_selectivity_results": float(len(res_low.ids)),
                "impossible_filter_results": float(len(res_impossible.ids)),
            },
        )
    except Exception as e:
        return ScenarioResult(
            suite="bench",
            scenario="filter_bench",
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
