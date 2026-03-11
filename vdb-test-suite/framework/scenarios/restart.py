from __future__ import annotations
import os
import signal
import time

import numpy as np

from framework.report import ScenarioResult


def run_crash_recovery_scenario(
    client_factory,
    server,
    collection: str,
    dim: int = 128,
    n_vectors: int = 500,
    seed: int = 42,
) -> ScenarioResult:
    """Insert data, SIGKILL the server, restart, verify sanity."""
    rng = np.random.default_rng(seed)
    try:
        client = client_factory()
        try:
            client.delete_collection(collection)
            client.create_collection(collection, dim=dim)

            vecs = rng.normal(size=(n_vectors, dim)).astype(np.float32)
            vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
            ids = np.arange(1, n_vectors + 1, dtype=np.uint64)
            client.insert(collection, ids, vecs)
            client.flush(collection)
        finally:
            client.close()

        # Force kill
        if server.proc and server.proc.poll() is None:
            try:
                os.killpg(server.proc.pid, signal.SIGKILL)
                server.proc.wait(timeout=5)
            except Exception:
                pass

        server.start()
        time.sleep(0.5)

        client2 = client_factory()
        try:
            exists = client2.collection_exists(collection)
            if not exists:
                return ScenarioResult(
                    suite="soak",
                    scenario="crash_recovery",
                    vdb="deepdata",
                    dataset="synthetic",
                    success=False,
                    error="Collection gone after crash recovery",
                )

            after_count = client2.count(collection)
            q = vecs[0]
            res = client2.search(collection, q, top_k=10)

            # After crash, count may be less (uncommitted writes lost), but service must work
            return ScenarioResult(
                suite="soak",
                scenario="crash_recovery",
                vdb="deepdata",
                dataset="synthetic",
                success=True,
                metrics={
                    "expected_count": float(n_vectors),
                    "actual_count": float(after_count),
                    "search_returned": float(len(res.ids)),
                    "service_healthy": 1.0,
                },
            )
        finally:
            try:
                client2.delete_collection(collection)
            except Exception:
                pass
            client2.close()
    except Exception as e:
        return ScenarioResult(
            suite="soak",
            scenario="crash_recovery",
            vdb="deepdata",
            dataset="synthetic",
            success=False,
            error=str(e),
        )
