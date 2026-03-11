from __future__ import annotations

import numpy as np
import pytest


def test_insert_count_and_delete_cycle(client, collection_name, small_vectors):
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)
    assert client.count(collection_name) == len(small_vectors)

    to_delete = ids[:10]
    client.delete_ids(collection_name, to_delete)
    assert client.count(collection_name) == len(small_vectors) - len(to_delete)


def test_reinsert_after_delete(client, collection_name, small_vectors):
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)

    removed = ids[:5]
    client.delete_ids(collection_name, removed)
    assert client.count(collection_name) == len(small_vectors) - len(removed)

    client.insert(collection_name, removed, small_vectors[:5])
    assert client.count(collection_name) == len(small_vectors)
