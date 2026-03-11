from __future__ import annotations
import struct
import time
from typing import Any

import httpx
import numpy as np

from framework.clients.base import SearchResult


class DeepDataClient:
    name = "deepdata"

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def close(self) -> None:
        self.client.close()

    def create_collection(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        metadata_schema: dict | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "Name": name,
            "Fields": [{"Name": "embedding", "Type": 0, "Dim": dim}],
        }
        if metadata_schema:
            payload["MetadataSchema"] = metadata_schema
        resp = self.client.post("/v2/collections", json=payload)
        resp.raise_for_status()

    def delete_collection(self, name: str) -> None:
        resp = self.client.delete(f"/v2/collections/{name}")
        if resp.status_code not in (200, 204, 400, 404):
            resp.raise_for_status()

    def collection_exists(self, name: str) -> bool:
        resp = self.client.get(f"/v2/collections/{name}")
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def count(self, name: str) -> int:
        resp = self.client.get(f"/v2/collections/{name}/stats")
        resp.raise_for_status()
        data = resp.json()
        for key in ("doc_count", "count", "Count", "num_documents", "NumDocuments"):
            if key in data:
                return int(data[key])
        raise KeyError(f"Could not find count field in stats response: {data}")

    def _build_binary_import_payload(self, ids, vectors) -> bytes:
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        ids_arr = np.asarray(ids, dtype=np.uint64)
        if len(ids_arr) != len(vecs):
            raise ValueError("ids and vectors must have same length")
        count, dim = vecs.shape
        rec_dtype = np.dtype([("id", "<u8"), ("vec", "<f4", (dim,))], align=False)
        recs = np.empty(count, dtype=rec_dtype)
        recs["id"] = ids_arr
        recs["vec"] = np.ascontiguousarray(vecs)
        header = struct.pack("<II", count, dim)
        return header + recs.tobytes()

    def insert(self, name: str, ids, vectors, payloads=None) -> None:
        if payloads is not None:
            return self.insert_json(name, ids, vectors, payloads)
        payload = self._build_binary_import_payload(ids, vectors)
        resp = self.client.post(
            f"/v2/import?collection={name}&field=embedding",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()

    def insert_json(self, name: str, ids, vectors, payloads=None) -> None:
        """Insert via JSON API (supports metadata payloads)."""
        vecs = np.asarray(vectors, dtype=np.float32)
        ids_arr = np.asarray(ids, dtype=np.uint64)
        docs = []
        for i in range(len(ids_arr)):
            doc: dict[str, Any] = {
                "collection": name,
                "vectors": {"embedding": vecs[i].tolist()},
            }
            if payloads and i < len(payloads) and payloads[i]:
                doc["metadata"] = payloads[i]
            docs.append(doc)
        resp = self.client.post("/v2/insert/batch", json={"collection": name, "docs": docs})
        resp.raise_for_status()

    def upsert(self, name: str, ids, vectors, payloads=None) -> None:
        self.insert(name, ids, vectors, payloads=payloads)

    def delete_ids(self, name: str, ids) -> None:
        for doc_id in ids:
            resp = self.client.post(
                "/v2/delete",
                json={"collection": name, "doc_id": int(doc_id)},
            )
            resp.raise_for_status()

    def get_by_ids(self, name: str, ids) -> list[dict]:
        resp = self.client.post(
            f"/v2/collections/{name}/get",
            json={"ids": [int(x) for x in ids]},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("documents", data.get("Documents", []))

    def search(
        self,
        name: str,
        vector,
        top_k: int,
        filters: dict | None = None,
        ef_search: int | None = None,
    ) -> SearchResult:
        payload: dict[str, Any] = {
            "collection": name,
            "queries": {"embedding": np.asarray(vector, dtype=np.float32).tolist()},
            "top_k": int(top_k),
        }
        if ef_search is not None:
            payload["ef_search"] = int(ef_search)
        if filters is not None:
            payload["filters"] = filters
        resp = self.client.post("/v2/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("documents", data.get("Documents", []))
        ids = [int(d.get("id", d.get("ID", 0))) for d in docs]
        scores = []
        for d in docs:
            score = d.get("score", d.get("Score"))
            if score is not None:
                scores.append(float(score))
        return SearchResult(ids=ids, scores=scores or None, raw=data)

    def flush(self, name: str | None = None) -> None:
        # Starter no-op. Replace with real flush endpoint if DeepData exposes one.
        time.sleep(0.05)

    # ── v3 Tenant API ──────────────────────────────────────────────

    def tenant_create_collection(self, tenant_id: str, name: str, dim: int) -> None:
        payload = {
            "Name": name,
            "Fields": [{"Name": "embedding", "Type": 0, "Dim": dim}],
        }
        resp = self.client.post(f"/v3/tenants/{tenant_id}/collections", json=payload)
        resp.raise_for_status()

    def tenant_delete_collection(self, tenant_id: str, name: str) -> None:
        resp = self.client.delete(f"/v3/tenants/{tenant_id}/collections/{name}")
        if resp.status_code not in (200, 204, 400, 404):
            resp.raise_for_status()

    def tenant_insert(self, tenant_id: str, collection: str, vector, metadata: dict | None = None) -> int:
        """Insert a single doc into a tenant collection. Returns the assigned doc ID."""
        vec = np.asarray(vector, dtype=np.float32).tolist()
        body: dict[str, Any] = {"vectors": {"embedding": vec}}
        if metadata:
            body["metadata"] = metadata
        resp = self.client.post(
            f"/v3/tenants/{tenant_id}/collections/{collection}/docs",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return int(data.get("doc_id", data.get("id", data.get("ID", 0))))

    def tenant_search(self, tenant_id: str, collection: str, vector, top_k: int) -> SearchResult:
        vec = np.asarray(vector, dtype=np.float32).tolist()
        body = {"queries": {"embedding": vec}, "top_k": top_k}
        resp = self.client.post(
            f"/v3/tenants/{tenant_id}/collections/{collection}/search",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("documents", data.get("Documents", []))
        ids = [int(d.get("id", d.get("ID", 0))) for d in docs]
        scores = []
        for d in docs:
            score = d.get("score", d.get("Score"))
            if score is not None:
                scores.append(float(score))
        return SearchResult(ids=ids, scores=scores or None, raw=data)

    def tenant_delete_doc(self, tenant_id: str, collection: str, doc_id: int) -> None:
        resp = self.client.request(
            "DELETE",
            f"/v3/tenants/{tenant_id}/collections/{collection}/docs",
            json={"doc_id": doc_id},
        )
        resp.raise_for_status()
