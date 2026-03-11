"""DeepData benchmark adapter using gRPC for low-latency binary transport."""

import grpc

from .base import SearchResult, VDBAdapter
from .deepdata.v1 import deepdata_pb2, deepdata_pb2_grpc


class DeepDataGRPCAdapter(VDBAdapter):
    name = "deepdata-grpc"

    def __init__(self, url: str = "localhost:50051"):
        self._channel = grpc.insecure_channel(url, options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ])
        self._stub = deepdata_pb2_grpc.DeepDataStub(self._channel)
        self._id_map: dict[int, str] = {}

    def create_collection(
        self, name: str, dim: int, hnsw_m: int = 16, ef_construction: int = 200
    ) -> None:
        # Delete first (ignore errors)
        try:
            self._stub.DeleteCollection(
                deepdata_pb2.DeleteCollectionRequest(name=name)
            )
        except Exception:
            pass

        field = deepdata_pb2.VectorFieldConfig(
            name="embedding",
            type=0,  # dense
            dim=dim,
            index_type="hnsw",
            index_params={"m": float(hnsw_m), "ef_construction": float(ef_construction)},
        )
        self._stub.CreateCollection(
            deepdata_pb2.CreateCollectionRequest(name=name, fields=[field])
        )
        self._id_map.clear()

    def insert_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        docs = []
        for i in range(len(ids)):
            meta = {**{k: str(v) for k, v in metadata[i].items()}, "chunk_id": ids[i]}
            vec = deepdata_pb2.VectorData(
                dense=deepdata_pb2.DenseVector(values=vectors[i])
            )
            docs.append(deepdata_pb2.BatchDoc(
                vectors={"embedding": vec},
                metadata=meta,
                text=texts[i],
            ))

        resp = self._stub.BatchInsert(
            deepdata_pb2.BatchInsertRequest(collection=collection, docs=docs)
        )

        # Map internal IDs to chunk IDs
        for j, internal_id in enumerate(resp.ids):
            if j < len(ids):
                self._id_map[internal_id] = ids[j]

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 100,
        ef_search: int = 128,
        meta_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        query_vec = deepdata_pb2.VectorData(
            dense=deepdata_pb2.DenseVector(values=vector)
        )
        req = deepdata_pb2.SearchRequest(
            collection=collection,
            queries={"embedding": query_vec},
            top_k=top_k,
            ef_search=ef_search,
        )
        if meta_filter:
            for k, v in meta_filter.items():
                req.metadata_filter[k] = str(v)

        resp = self._stub.Search(req)

        results = []
        for hit in resp.results:
            chunk_id = self._id_map.get(hit.id, str(hit.id))
            results.append(SearchResult(
                id=chunk_id,
                score=hit.score,
                metadata=dict(hit.metadata),
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
        # gRPC adapter uses same search endpoint (no separate hybrid RPC)
        return self.search(collection, vector, top_k=top_k)

    def supports_hybrid(self) -> bool:
        return True

    def supports_graph(self) -> bool:
        return False

    def get_memory_usage(self) -> int | None:
        return None

    def delete_collection(self, name: str) -> None:
        try:
            self._stub.DeleteCollection(
                deepdata_pb2.DeleteCollectionRequest(name=name)
            )
        except Exception:
            pass

    def teardown(self) -> None:
        self._channel.close()
