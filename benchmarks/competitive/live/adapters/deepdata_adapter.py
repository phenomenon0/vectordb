"""DeepData benchmark adapter using raw HTTP for v2 API."""

import httpx

from .base import SearchResult, VDBAdapter


class DeepDataAdapter(VDBAdapter):
    name = "deepdata"

    def __init__(self, url: str = "http://localhost:8080"):
        self._url = url
        self._http = httpx.Client(base_url=url, timeout=60.0)
        # Maps internal uint64 doc_id → our string chunk id
        self._id_map: dict[int, str] = {}

    def create_collection(
        self, name: str, dim: int, hnsw_m: int = 16, ef_construction: int = 200
    ) -> None:
        # Delete if exists
        try:
            self._http.delete(f"/v2/collections/{name}")
        except Exception:
            pass

        # v2 schema uses Go struct field names (no json tags)
        schema = {
            "Name": name,
            "Fields": [
                {"Name": "embedding", "Type": 0, "Dim": dim}
            ],
        }
        resp = self._http.post("/v2/collections", json=schema)
        resp.raise_for_status()
        self._id_map.clear()

    def insert_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        import time
        for i in range(len(ids)):
            meta = {**metadata[i], "chunk_id": ids[i]}
            payload = {
                "collection": collection,
                "doc": texts[i],
                "vectors": {"embedding": vectors[i]},
                "metadata": meta,
            }
            # Retry on 429 with exponential backoff
            for attempt in range(10):
                resp = self._http.post("/v2/insert", json=payload)
                if resp.status_code == 429:
                    time.sleep(0.5 * (2 ** min(attempt, 4)))
                    continue
                resp.raise_for_status()
                data = resp.json()
                internal_id = data.get("id")
                if internal_id is not None:
                    self._id_map[internal_id] = ids[i]
                break
            else:
                raise RuntimeError(f"Failed to insert doc {i} after 10 retries (429)")

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 100,
        ef_search: int = 128,
        meta_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        payload = {
            "collection": collection,
            "queries": {"embedding": vector},
            "top_k": top_k,
        }
        if meta_filter:
            # v2 filter format: {"key": {"$eq": "value"}}
            filters = {}
            for k, v in meta_filter.items():
                filters[k] = {"$eq": v}
            payload["filters"] = filters

        resp = self._http.post("/v2/search", json=payload)
        resp.raise_for_status()
        data = resp.json()

        results = []
        documents = data.get("documents", [])
        scores = data.get("scores", [])
        for j, doc in enumerate(documents):
            internal_id = doc.get("id", doc.get("ID", 0))
            chunk_id = self._id_map.get(internal_id, str(internal_id))
            score = scores[j] if j < len(scores) else 0.0
            results.append(SearchResult(
                id=chunk_id,
                score=score,
                metadata=doc.get("metadata", doc.get("Metadata", {})),
            ))
        return results

    def hybrid_search(
        self,
        collection: str,
        text: str,
        vector: list[float],
        top_k: int = 100,
        alpha: float = 0.7,
    ) -> list[SearchResult]:
        # v2 hybrid search with graph weight
        payload = {
            "collection": collection,
            "queries": {"embedding": vector},
            "query_text": text,
            "top_k": top_k,
            "graph_weight": alpha,
        }
        resp = self._http.post("/v2/search", json=payload)
        resp.raise_for_status()
        data = resp.json()

        results = []
        documents = data.get("documents", [])
        scores = data.get("scores", [])
        for j, doc in enumerate(documents):
            internal_id = doc.get("id", doc.get("ID", 0))
            chunk_id = self._id_map.get(internal_id, str(internal_id))
            score = scores[j] if j < len(scores) else 0.0
            results.append(SearchResult(id=chunk_id, score=score))
        return results

    def supports_hybrid(self) -> bool:
        return True

    def supports_graph(self) -> bool:
        return True

    def get_memory_usage(self) -> int | None:
        try:
            resp = self._http.get("/health")
            data = resp.json()
            return data.get("index_bytes")
        except Exception:
            return None

    def delete_collection(self, name: str) -> None:
        try:
            self._http.delete(f"/v2/collections/{name}")
        except Exception:
            pass

    def teardown(self) -> None:
        self._http.close()
