"""Weaviate benchmark adapter using weaviate-client v4."""

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Filter

from .base import SearchResult, VDBAdapter


class WeaviateAdapter(VDBAdapter):
    name = "weaviate"

    def __init__(self, url: str = "http://localhost:8081"):
        self._client = weaviate.connect_to_custom(
            http_host=url.replace("http://", "").split(":")[0],
            http_port=int(url.split(":")[-1]),
            http_secure=False,
            grpc_host=url.replace("http://", "").split(":")[0],
            grpc_port=50051,
            grpc_secure=False,
        )

    def _class_name(self, collection: str) -> str:
        """Weaviate class names must be PascalCase."""
        return collection.replace("-", "_").replace(" ", "_").title().replace("_", "")

    def create_collection(
        self, name: str, dim: int, hnsw_m: int = 16, ef_construction: int = 200
    ) -> None:
        class_name = self._class_name(name)
        # Delete if exists
        try:
            self._client.collections.delete(class_name)
        except Exception:
            pass

        self._client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=ef_construction,
                max_connections=hnsw_m,
                ef=128,
            ),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="file_path", data_type=DataType.TEXT),
                Property(name="doc_package", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.TEXT),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="has_test", data_type=DataType.TEXT),
                Property(name="loc", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
            ],
        )

    def insert_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        class_name = self._class_name(collection)
        coll = self._client.collections.get(class_name)

        with coll.batch.dynamic() as batch:
            for i in range(len(ids)):
                props = {
                    "text": texts[i],
                    "doc_id": ids[i],
                    "file_path": metadata[i].get("file_path", ""),
                    "doc_package": metadata[i].get("package", ""),
                    "chunk_index": metadata[i].get("chunk_index", ""),
                    "language": metadata[i].get("language", ""),
                    "has_test": metadata[i].get("has_test", ""),
                    "loc": metadata[i].get("loc", ""),
                }
                batch.add_object(properties=props, vector=vectors[i])

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 100,
        ef_search: int = 128,
        meta_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        class_name = self._class_name(collection)
        coll = self._client.collections.get(class_name)

        filters = None
        if meta_filter:
            filter_parts = []
            for key, value in meta_filter.items():
                prop_name = "doc_package" if key == "package" else key
                filter_parts.append(Filter.by_property(prop_name).equal(value))
            if len(filter_parts) == 1:
                filters = filter_parts[0]
            else:
                filters = Filter.all_of(filter_parts)

        resp = coll.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["doc_id"],
            filters=filters,
        )

        results = []
        for obj in resp.objects:
            # Weaviate returns distance, convert to similarity
            distance = obj.metadata.distance or 0.0
            score = 1.0 - distance  # cosine distance → similarity
            results.append(SearchResult(
                id=obj.properties.get("doc_id", ""),
                score=score,
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
        class_name = self._class_name(collection)
        coll = self._client.collections.get(class_name)

        resp = coll.query.hybrid(
            query=text,
            vector=vector,
            alpha=alpha,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
            return_properties=["doc_id"],
        )

        results = []
        for obj in resp.objects:
            score = obj.metadata.score or 0.0
            results.append(SearchResult(
                id=obj.properties.get("doc_id", ""),
                score=score,
            ))
        return results

    def supports_hybrid(self) -> bool:
        return True

    def delete_collection(self, name: str) -> None:
        try:
            self._client.collections.delete(self._class_name(name))
        except Exception:
            pass

    def teardown(self) -> None:
        self._client.close()
