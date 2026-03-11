from __future__ import annotations

import numpy as np
import pytest

from framework.clients.deepdata import DeepDataClient


def test_data_survives_restart(deepdata_server, client, collection_name, small_vectors):
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)
    before = client.count(collection_name)
    query = small_vectors[0]
    before_search = client.search(collection_name, query, top_k=10).ids

    client.close()
    deepdata_server.stop()
    deepdata_server.start()

    c2 = DeepDataClient(base_url=f"http://127.0.0.1:{deepdata_server.port}")
    try:
        after = c2.count(collection_name)
        after_search = c2.search(collection_name, query, top_k=10).ids
        assert after == before
        assert after_search[:5] == before_search[:5]
    finally:
        c2.delete_collection(collection_name)
        c2.close()
