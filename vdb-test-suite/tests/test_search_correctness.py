from __future__ import annotations

import numpy as np


def _brute_force_topk(base: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Cosine similarity brute force search."""
    base_norm = base / np.clip(np.linalg.norm(base, axis=1, keepdims=True), 1e-12, None)
    q_norm = query / np.clip(np.linalg.norm(query), 1e-12, None)
    sims = base_norm @ q_norm
    return list(np.argsort(-sims)[:k])


def test_self_nearest(client, collection_name, small_vectors):
    """Each vector should be its own nearest neighbor."""
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)

    for i in range(min(10, len(small_vectors))):
        res = client.search(collection_name, small_vectors[i], top_k=1, ef_search=200)
        assert res.ids[0] == i + 1, f"Vector {i+1} not its own nearest: got {res.ids[0]}"


def test_topk_ordering(client, collection_name, small_vectors):
    """Results should be ordered by descending similarity."""
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)

    res = client.search(collection_name, small_vectors[0], top_k=10, ef_search=200)
    if res.scores is not None and len(res.scores) >= 2:
        for j in range(len(res.scores) - 1):
            assert res.scores[j] >= res.scores[j + 1], (
                f"Scores not ordered: {res.scores[j]} < {res.scores[j + 1]}"
            )


def test_repeated_search_stable(client, collection_name, small_vectors):
    """Same query should return same results."""
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)

    q = small_vectors[0]
    res1 = client.search(collection_name, q, top_k=10, ef_search=200)
    res2 = client.search(collection_name, q, top_k=10, ef_search=200)
    assert res1.ids == res2.ids, f"Unstable results: {res1.ids} vs {res2.ids}"


def test_brute_force_agreement_small(client, collection_name, rng):
    """On a small corpus, ANN should agree with brute force top-5."""
    dim = 16
    n = 50
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    client.create_collection(collection_name, dim=dim)
    ids = np.arange(1, n + 1, dtype=np.uint64)
    client.insert(collection_name, ids, vecs)

    q = rng.normal(size=(dim,)).astype(np.float32)
    bf_ids = [x + 1 for x in _brute_force_topk(vecs, q, 5)]  # 1-based
    res = client.search(collection_name, q, top_k=5, ef_search=200)

    overlap = len(set(res.ids[:5]) & set(bf_ids))
    assert overlap >= 4, (
        f"ANN disagrees with brute force: overlap={overlap}/5, "
        f"ANN={res.ids[:5]}, BF={bf_ids}"
    )
