from __future__ import annotations

import numpy as np
import pytest


def test_wrong_dimension_insert_fails(client, collection_name, rng):
    client.create_collection(collection_name, dim=8)
    ids = np.arange(10, dtype=np.uint64)
    wrong = rng.normal(size=(10, 16)).astype(np.float32)
    with pytest.raises(Exception):
        client.insert(collection_name, ids, wrong)


def test_search_unknown_collection_fails(client, rng):
    q = rng.normal(size=(16,)).astype(np.float32)
    with pytest.raises(Exception):
        client.search("does_not_exist", q, top_k=5)


def test_empty_query_rejected(client, collection_name):
    client.create_collection(collection_name, dim=16)
    with pytest.raises(Exception):
        client.search(collection_name, [], top_k=5)


def test_negative_topk_rejected(client, collection_name, rng):
    client.create_collection(collection_name, dim=16)
    q = rng.normal(size=(16,)).astype(np.float32)
    with pytest.raises(Exception):
        client.search(collection_name, q, top_k=-1)
