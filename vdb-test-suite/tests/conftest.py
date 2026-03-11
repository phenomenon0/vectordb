from __future__ import annotations
import os
import uuid

import numpy as np
import pytest

from framework.clients.deepdata import DeepDataClient
from framework.server.deepdata_process import DeepDataProcess


@pytest.fixture(scope="session")
def deepdata_server():
    if os.environ.get("DEEPDATA_TESTS", "0") != "1":
        pytest.skip("Set DEEPDATA_TESTS=1 to run integration tests")
    server = DeepDataProcess(port=int(os.environ.get("DEEPDATA_PORT", "8080")))
    if os.environ.get("DEEPDATA_SKIP_BUILD", "0") != "1":
        server.build()
    server.start(clean=True)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture()
def client(deepdata_server):
    c = DeepDataClient(base_url=f"http://127.0.0.1:{deepdata_server.port}")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture()
def collection_name():
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def small_vectors(rng):
    vecs = rng.normal(size=(100, 16)).astype(np.float32)
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    return vecs
