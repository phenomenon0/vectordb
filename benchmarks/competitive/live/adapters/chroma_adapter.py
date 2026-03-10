"""ChromaDB benchmark adapter using chromadb HTTP client."""

import chromadb

from .base import SearchResult, VDBAdapter


class ChromaAdapter(VDBAdapter):
    name = "chromadb"

    def __init__(self, host: str = "localhost", port: int = 8010):
        self._client = chromadb.HttpClient(host=host, port=port)

    def create_collection(
        self, name: str, dim: int, hnsw_m: int = 16, ef_construction: int = 200
    ) -> None:
        # Delete if exists
        try:
            self._client.delete_collection(name)
        except Exception:
            pass

        self._client.create_collection(
            name=name,
            metadata={
                "hnsw:M": hnsw_m,
                "hnsw:construction_ef": ef_construction,
                "hnsw:search_ef": 128,
                "hnsw:space": "cosine",
            },
        )

    def insert_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        coll = self._client.get_collection(collection)
        # Ensure all metadata values are primitives (str, int, float, bool)
        clean_meta = []
        for m in metadata:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_meta.append(clean)

        # ChromaDB has a batch limit, insert in sub-batches
        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            coll.add(
                ids=ids[start:end],
                embeddings=vectors[start:end],
                documents=texts[start:end],
                metadatas=clean_meta[start:end],
            )

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 100,
        ef_search: int = 128,
        meta_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        coll = self._client.get_collection(collection)

        where = None
        if meta_filter:
            if len(meta_filter) == 1:
                key, value = next(iter(meta_filter.items()))
                where = {key: {"$eq": value}}
            else:
                where = {
                    "$and": [
                        {k: {"$eq": v}} for k, v in meta_filter.items()
                    ]
                }

        resp = coll.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["distances"],
        )

        results = []
        if resp["ids"] and resp["ids"][0]:
            for doc_id, distance in zip(resp["ids"][0], resp["distances"][0]):
                # ChromaDB returns cosine distance; convert to similarity
                score = 1.0 - distance
                results.append(SearchResult(id=doc_id, score=score))
        return results

    def supports_hybrid(self) -> bool:
        return False

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:
            pass

    def teardown(self) -> None:
        pass
