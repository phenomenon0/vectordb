from __future__ import annotations

import numpy as np
import pytest


def test_tenant_isolation(client, rng):
    """Data inserted under tenant A must not appear in tenant B queries."""
    dim = 16
    n = 30
    coll = "shared_coll"

    try:
        client.tenant_create_collection("alpha", coll, dim=dim)
        client.tenant_create_collection("beta", coll, dim=dim)

        # Insert distinct vectors per tenant
        vecs_a = rng.normal(size=(n, dim)).astype(np.float32)
        vecs_a /= np.clip(np.linalg.norm(vecs_a, axis=1, keepdims=True), 1e-12, None)
        alpha_ids = []
        for i in range(n):
            doc_id = client.tenant_insert("alpha", coll, vecs_a[i])
            alpha_ids.append(doc_id)

        vecs_b = rng.normal(size=(n, dim)).astype(np.float32)
        vecs_b /= np.clip(np.linalg.norm(vecs_b, axis=1, keepdims=True), 1e-12, None)
        beta_ids = []
        for i in range(n):
            doc_id = client.tenant_insert("beta", coll, vecs_b[i])
            beta_ids.append(doc_id)

        alpha_set = set(alpha_ids)
        beta_set = set(beta_ids)

        # Query tenant alpha — should only see alpha's IDs
        res_a = client.tenant_search("alpha", coll, vecs_b[0], top_k=10)
        for rid in res_a.ids:
            assert rid in alpha_set, f"Beta ID {rid} leaked into alpha results"

        # Query tenant beta — should only see beta's IDs
        res_b = client.tenant_search("beta", coll, vecs_a[0], top_k=10)
        for rid in res_b.ids:
            assert rid in beta_set, f"Alpha ID {rid} leaked into beta results"
    finally:
        for tid in ("alpha", "beta"):
            try:
                client.tenant_delete_collection(tid, coll)
            except Exception:
                pass
