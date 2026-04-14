"""DeepData client for VectorDBBench.

Implements the VectorDB interface to integrate DeepData with the
industry-standard VectorDBBench benchmarking framework.
"""

import logging
import time
from contextlib import contextmanager

import httpx

from vectordb_bench.backend.clients.api import VectorDB
from vectordb_bench.backend.filter import Filter, FilterOp

from .config import DeepDataIndexConfig

log = logging.getLogger(__name__)

BATCH_SIZE = 500


class DeepData(VectorDB):
    """DeepData VectorDBBench client.

    Uses DeepData's v2 HTTP API for collection management,
    batch insert, and vector search operations.
    """

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
    ]
    name: str = "DeepData"

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DeepDataIndexConfig,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name

        base_url = db_config.get("url", "http://localhost:8080")
        self._base_url = base_url

        client = httpx.Client(base_url=base_url, timeout=60.0)
        try:
            resp = client.get("/health")
            resp.raise_for_status()
            log.info(f"DeepData health check OK: {resp.json()}")
        except Exception as e:
            log.warning(f"DeepData health check failed: {e}")

        if drop_old:
            try:
                client.delete(f"/v2/collections/{collection_name}")
                log.info(f"Dropped collection: {collection_name}")
            except Exception:
                pass
            self._create_collection(dim, client)

        client.close()
        self._client = None
        self._query_filter = None

    def _create_collection(self, dim: int, client: httpx.Client) -> None:
        """Create a new collection with HNSW index."""
        index_params = self.case_config.index_param() if self.case_config else {}

        schema = {
            "Name": self.collection_name,
            "Fields": [
                {
                    "Name": "embedding",
                    "Type": 0,  # Dense
                    "Dim": dim,
                    "Index": {
                        "Type": 0,  # HNSW
                        "Params": {
                            "m": index_params.get("m", 16),
                            "ef_construction": index_params.get("ef_construction", 200),
                        },
                    },
                }
            ],
        }
        resp = client.post("/v2/collections", json=schema)
        resp.raise_for_status()
        log.info(f"Created collection: {self.collection_name} (dim={dim})")

    @contextmanager
    def init(self):
        self._client = httpx.Client(base_url=self._base_url, timeout=120.0)
        try:
            yield
        finally:
            self._client.close()
            self._client = None

    def optimize(self, data_size: int | None = None):
        """DeepData indexes on insert, wait briefly for background flush."""
        assert self._client is not None, "Call init() first"
        # Poll health until index is stable
        for _ in range(30):
            try:
                resp = self._client.get(
                    f"/v2/collections/{self.collection_name}/stats"
                )
                if resp.status_code == 200:
                    stats = resp.json()
                    log.info(f"Collection stats: {stats}")
                    break
            except Exception:
                pass
            time.sleep(1)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings using v2 batch insert API."""
        assert self._client is not None, "Call init() first"

        total_inserted = 0
        try:
            for offset in range(0, len(embeddings), BATCH_SIZE):
                batch_embeddings = embeddings[offset : offset + BATCH_SIZE]
                batch_metadata = metadata[offset : offset + BATCH_SIZE]

                docs = []
                for i, (emb, mid) in enumerate(
                    zip(batch_embeddings, batch_metadata)
                ):
                    doc = {
                        "doc": "",
                        "vectors": {"embedding": emb},
                        "metadata": {"id": mid},
                    }
                    if labels_data:
                        idx = offset + i
                        if idx < len(labels_data):
                            doc["metadata"]["labels"] = labels_data[idx]
                    docs.append(doc)

                payload = {
                    "collection": self.collection_name,
                    "documents": docs,
                }

                # Retry on 429
                for attempt in range(5):
                    resp = self._client.post(
                        "/v2/insert/batch", json=payload, timeout=120.0
                    )
                    if resp.status_code == 429:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    resp.raise_for_status()
                    break
                else:
                    return total_inserted, RuntimeError("Insert failed: 429 rate limit")

                total_inserted += len(batch_embeddings)

        except Exception as e:
            log.warning(f"Insert failed at offset {total_inserted}: {e}")
            return total_inserted, e

        return total_inserted, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
    ) -> list[int]:
        """Search for k nearest neighbors."""
        assert self._client is not None, "Call init() first"

        search_params = self.case_config.search_param() if self.case_config else {}

        payload = {
            "collection": self.collection_name,
            "queries": {"embedding": query},
            "top_k": k,
        }

        if self._query_filter is not None:
            payload["filters"] = self._query_filter

        resp = self._client.post("/v2/search", json=payload)
        resp.raise_for_status()
        data = resp.json()

        results = []
        documents = data.get("documents", [])
        for doc in documents:
            meta = doc.get("Metadata", {})
            doc_id = meta.get("id", doc.get("ID", 0))
            if isinstance(doc_id, str):
                try:
                    doc_id = int(doc_id)
                except ValueError:
                    doc_id = 0
            results.append(doc_id)

        return results

    def prepare_filter(self, filters: Filter):
        """Prepare filter for search queries."""
        if filters.type == FilterOp.NonFilter:
            self._query_filter = None
        elif filters.type == FilterOp.NumGE:
            self._query_filter = {
                "id": {"$gte": filters.int_value}
            }

    def need_normalize_cosine(self) -> bool:
        return False
