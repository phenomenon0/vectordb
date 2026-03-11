from __future__ import annotations
import threading

import numpy as np
import pytest


def test_concurrent_search_does_not_crash(client, collection_name, small_vectors):
    """Parallel searches should not wedge or crash the server."""
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)

    errors: list[str] = []

    def search_worker(idx: int) -> None:
        try:
            q = small_vectors[idx % len(small_vectors)]
            res = client.search(collection_name, q, top_k=5, ef_search=200)
            assert len(res.ids) > 0
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=search_worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent search errors: {errors}"


def test_concurrent_insert_and_search(client, collection_name, rng):
    """Insert + search running concurrently should not produce errors."""
    dim = 16
    client.create_collection(collection_name, dim=dim)

    # Seed some initial data
    seed_vecs = rng.normal(size=(50, dim)).astype(np.float32)
    seed_vecs /= np.clip(np.linalg.norm(seed_vecs, axis=1, keepdims=True), 1e-12, None)
    client.insert(collection_name, np.arange(1, 51, dtype=np.uint64), seed_vecs)

    errors: list[str] = []
    insert_count = 0
    lock = threading.Lock()

    def inserter() -> None:
        nonlocal insert_count
        for i in range(10):
            try:
                vec = rng.normal(size=(1, dim)).astype(np.float32)
                doc_id = 1000 + i
                with lock:
                    insert_count += 1
                client.insert(
                    collection_name,
                    np.array([doc_id], dtype=np.uint64),
                    vec,
                )
            except Exception as e:
                errors.append(f"insert: {e}")

    def searcher() -> None:
        for _ in range(20):
            try:
                q = rng.normal(size=(dim,)).astype(np.float32)
                client.search(collection_name, q, top_k=5, ef_search=200)
            except Exception as e:
                errors.append(f"search: {e}")

    threads = [
        threading.Thread(target=inserter),
        threading.Thread(target=searcher),
        threading.Thread(target=searcher),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent errors: {errors}"
