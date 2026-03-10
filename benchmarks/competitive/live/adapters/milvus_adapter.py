"""Milvus benchmark adapter using pymilvus."""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from .base import SearchResult, VDBAdapter


class MilvusAdapter(VDBAdapter):
    name = "milvus"

    def __init__(self, host: str = "localhost", port: int = 19530):
        connections.connect("default", host=host, port=port)

    def create_collection(
        self, name: str, dim: int, hnsw_m: int = 16, ef_construction: int = 200
    ) -> None:
        if utility.has_collection(name):
            utility.drop_collection(name)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="package", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chunk_index", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="has_test", dtype=DataType.VARCHAR, max_length=8),
            FieldSchema(name="loc", dtype=DataType.VARCHAR, max_length=16),
        ]

        schema = CollectionSchema(fields=fields)
        coll = Collection(name=name, schema=schema)

        # Create HNSW index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": hnsw_m, "efConstruction": ef_construction},
        }
        coll.create_index("embedding", index_params)

    def insert_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        coll = Collection(name=collection)
        data = [
            ids,
            vectors,
            texts,
            [m.get("file_path", "") for m in metadata],
            [m.get("package", "") for m in metadata],
            [m.get("chunk_index", "") for m in metadata],
            [m.get("language", "") for m in metadata],
            [m.get("has_test", "") for m in metadata],
            [m.get("loc", "") for m in metadata],
        ]
        coll.insert(data)
        coll.flush()

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 100,
        ef_search: int = 128,
        meta_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        coll = Collection(name=collection)
        coll.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": ef_search}}

        expr = None
        if meta_filter:
            parts = []
            for key, value in meta_filter.items():
                parts.append(f'{key} == "{value}"')
            expr = " and ".join(parts)

        results = coll.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id"],
        )

        search_results = []
        if results:
            for hit in results[0]:
                search_results.append(SearchResult(
                    id=hit.id,
                    score=hit.score,
                ))
        return search_results

    def supports_hybrid(self) -> bool:
        return True

    def hybrid_search(
        self,
        collection: str,
        text: str,
        vector: list[float],
        top_k: int = 100,
        alpha: float = 0.7,
    ) -> list[SearchResult]:
        # Milvus hybrid requires sparse vectors which need separate setup.
        # For this benchmark, fall back to dense-only search.
        # A full implementation would use BM25 sparse + RRFRanker.
        return self.search(collection, vector, top_k)

    def delete_collection(self, name: str) -> None:
        try:
            if utility.has_collection(name):
                utility.drop_collection(name)
        except Exception:
            pass

    def teardown(self) -> None:
        connections.disconnect("default")
