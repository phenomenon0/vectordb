from __future__ import annotations
import os
import signal

import numpy as np
import pytest

from framework.clients.deepdata import DeepDataClient


def test_graceful_restart_preserves_data(deepdata_server, client, collection_name, small_vectors):
    """Data should survive a graceful SIGTERM + restart."""
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)
    before_count = client.count(collection_name)
    client.close()

    deepdata_server.stop()
    deepdata_server.start()

    c2 = DeepDataClient(base_url=f"http://127.0.0.1:{deepdata_server.port}")
    try:
        assert c2.collection_exists(collection_name)
        assert c2.count(collection_name) == before_count
    finally:
        try:
            c2.delete_collection(collection_name)
        except Exception:
            pass
        c2.close()


def test_crash_recovery_service_healthy(deepdata_server, client, collection_name, small_vectors):
    """After SIGKILL, server should restart and serve requests."""
    client.create_collection(collection_name, dim=16)
    ids = np.arange(1, len(small_vectors) + 1, dtype=np.uint64)
    client.insert(collection_name, ids, small_vectors)
    client.close()

    # SIGKILL
    if deepdata_server.proc and deepdata_server.proc.poll() is None:
        try:
            os.killpg(deepdata_server.proc.pid, signal.SIGKILL)
            deepdata_server.proc.wait(timeout=5)
        except Exception:
            pass

    deepdata_server.start()

    c2 = DeepDataClient(base_url=f"http://127.0.0.1:{deepdata_server.port}")
    try:
        # Service should at least be up and responding
        # Collection may or may not survive, but service must not be wedged
        new_coll = f"post_crash_{collection_name}"
        c2.create_collection(new_coll, dim=16)
        assert c2.collection_exists(new_coll)
        c2.delete_collection(new_coll)
    finally:
        c2.close()


def test_writes_work_after_restart(deepdata_server, client, collection_name, small_vectors):
    """After restart, new writes and queries must work."""
    client.close()

    deepdata_server.stop()
    deepdata_server.start(clean=True)

    c2 = DeepDataClient(base_url=f"http://127.0.0.1:{deepdata_server.port}")
    try:
        c2.create_collection(collection_name, dim=16)
        ids = np.arange(1, 21, dtype=np.uint64)
        c2.insert(collection_name, ids, small_vectors[:20])
        assert c2.count(collection_name) == 20

        res = c2.search(collection_name, small_vectors[0], top_k=5, ef_search=200)
        assert len(res.ids) > 0
    finally:
        try:
            c2.delete_collection(collection_name)
        except Exception:
            pass
        c2.close()
