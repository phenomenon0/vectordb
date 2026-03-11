from __future__ import annotations

import numpy as np
import pytest


def test_vector_search_with_filter_only_returns_matching_docs(client, collection_name, rng):
    dim = 16
    n = 40

    client.create_collection(collection_name, dim=dim)

    # Generate vectors and normalize
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    ids = np.arange(1, n + 1, dtype=np.uint64)
    payloads = []
    for i in range(n):
        payloads.append({
            "tenant_id": int(i % 4),
            "doc_type": "report" if i % 2 == 0 else "memo",
        })

    client.insert(collection_name, ids, vecs, payloads=payloads)

    # Search with filter: tenant_id == 0
    query = vecs[0]
    res = client.search(collection_name, query, top_k=10, filters={"tenant_id": {"$eq": 0}})
    assert len(res.ids) > 0
    # All returned docs should have tenant_id == 0 (i.e. IDs 1, 5, 9, ...)
    for rid in res.ids:
        assert (rid - 1) % 4 == 0, f"Doc {rid} should have tenant_id=0 but doesn't"

    # Search with impossible filter — should return empty
    res2 = client.search(collection_name, query, top_k=10, filters={"tenant_id": {"$eq": 999}})
    assert len(res2.ids) == 0
