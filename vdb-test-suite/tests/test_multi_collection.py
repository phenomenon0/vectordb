from __future__ import annotations

import numpy as np


def test_multiple_collections_isolated(client, rng):
    """Data in one collection must not appear in another."""
    dim = 16
    n = 50
    collections = [f"multi_coll_{i}_{np.random.randint(100000)}" for i in range(3)]

    try:
        for i, coll in enumerate(collections):
            client.create_collection(coll, dim=dim)
            vecs = rng.normal(size=(n, dim)).astype(np.float32)
            vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
            # Use different ID ranges per collection
            ids = np.arange(i * 1000 + 1, i * 1000 + n + 1, dtype=np.uint64)
            client.insert(coll, ids, vecs)

        # Search each collection and verify IDs belong to that collection's range
        for i, coll in enumerate(collections):
            q = rng.normal(size=(dim,)).astype(np.float32)
            res = client.search(coll, q, top_k=10, ef_search=200)
            low = i * 1000 + 1
            high = i * 1000 + n
            for rid in res.ids:
                assert low <= rid <= high, (
                    f"Collection {coll}: got ID {rid} outside range [{low}, {high}]"
                )
    finally:
        for coll in collections:
            try:
                client.delete_collection(coll)
            except Exception:
                pass


def test_collection_count_independent(client, rng):
    """Inserting into one collection should not change another's count."""
    dim = 16
    coll_a = f"count_a_{np.random.randint(100000)}"
    coll_b = f"count_b_{np.random.randint(100000)}"

    try:
        client.create_collection(coll_a, dim=dim)
        client.create_collection(coll_b, dim=dim)

        vecs_a = rng.normal(size=(20, dim)).astype(np.float32)
        client.insert(coll_a, np.arange(1, 21, dtype=np.uint64), vecs_a)

        vecs_b = rng.normal(size=(50, dim)).astype(np.float32)
        client.insert(coll_b, np.arange(1, 51, dtype=np.uint64), vecs_b)

        assert client.count(coll_a) == 20
        assert client.count(coll_b) == 50
    finally:
        for coll in (coll_a, coll_b):
            try:
                client.delete_collection(coll)
            except Exception:
                pass
