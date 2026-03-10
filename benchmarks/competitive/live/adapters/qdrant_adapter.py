"""Qdrant benchmark adapter using qdrant-client."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from .base import SearchResult, VDBAdapter


class QdrantAdapter(VDBAdapter):
    name = "qdrant"

    def __init__(self, url: str = "http://localhost:6333"):
        self._client = QdrantClient(url=url, timeout=60)
        self._next_id = 0

    def create_collection(
        self, name: str, dim: int, hnsw_m: int = 16, ef_construction: int = 200
    ) -> None:
        # Delete if exists
        try:
            self._client.delete_collection(name)
        except Exception:
            pass

        self._next_id = 0
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=hnsw_m,
                ef_construct=ef_construction,
            ),
        )

    def insert_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        points = []
        for i in range(len(ids)):
            payload = {
                "text": texts[i],
                "doc_id": ids[i],
                **metadata[i],
            }
            points.append(PointStruct(
                id=self._next_id,
                vector=vectors[i],
                payload=payload,
            ))
            self._next_id += 1

        # Upsert in sub-batches of 100
        for start in range(0, len(points), 100):
            batch = points[start:start + 100]
            self._client.upsert(collection_name=collection, points=batch)

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 100,
        ef_search: int = 128,
        meta_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        query_filter = None
        if meta_filter:
            conditions = []
            for key, value in meta_filter.items():
                conditions.append(FieldCondition(
                    key=key, match=MatchValue(value=value)
                ))
            query_filter = Filter(must=conditions)

        hits = self._client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=ef_search),
            query_filter=query_filter,
            with_payload=["doc_id"],
        )

        return [
            SearchResult(
                id=hit.payload.get("doc_id", str(hit.id)),
                score=hit.score,
            )
            for hit in hits
        ]

    def supports_hybrid(self) -> bool:
        return False

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:
            pass

    def teardown(self) -> None:
        self._client.close()
