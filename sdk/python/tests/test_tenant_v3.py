"""Contract tests for the canonical tenant-aware v3 SDK surface."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import ValidationError as PydanticValidationError

from deepdata import (
    AsyncDeepDataClient,
    DeepDataClient,
    RetryConfig,
    TenantBatchInsertResponse,
    TenantCollectionListResponse,
    TenantCollectionMutationResponse,
    TenantCollectionSchema,
    TenantDeleteDocumentResponse,
    TenantDocumentInput,
    TenantGetCollectionResponse,
    TenantInfoResponse,
    TenantInsertResponse,
    TenantSearchRequest,
    TenantSearchResponse,
    TenantVectorField,
)
from deepdata.errors import (
    ConnectionError as DeepDataConnectionError,
    PermissionError as DeepDataPermissionError,
)


BASE = "http://testserver:8080"
TENANT = "tenant-123"
ROOT = f"{BASE}/v3/tenants/{TENANT}"
FIELD = {
    "name": "embedding",
    "type": "dense",
    "dim": 2,
    "index": {"type": "flat"},
}
COLLECTION_INFO = {
    "Name": "papers",
    "Fields": [FIELD],
    "Description": "Research papers",
    "Metadata": {"owner": "sdk"},
    "DocCount": 2,
}


def _body(route: respx.Route, call: int = 0) -> object:
    return json.loads(route.calls[call].request.content)


@pytest.mark.parametrize(
    ("vector_type", "index_type"),
    [("dense", "hnsw"), ("dense", "flat"), ("sparse", "inverted")],
)
def test_supported_canonical_index_matrix(
    vector_type: str, index_type: str
) -> None:
    field = TenantVectorField.model_validate(
        {
            "name": "vector",
            "type": vector_type,
            "dim": 2,
            "index": {"type": index_type},
        }
    )
    assert field.type == vector_type
    assert field.index.type == index_type


@pytest.mark.parametrize(
    ("index_type", "params"),
    [
        ("flat", {"quantization": {"type": "pq"}}),
        ("hnsw", {"quantization": {"type": "float16"}}),
        ("flat", {"gpu": True}),
        ("flat", {"metric": "dot"}),
        ("hnsw", {"m": 1}),
        ("inverted", {"b": 2.0}),
    ],
)
def test_rejects_out_of_scope_or_invalid_index_params(
    index_type: str, params: dict[str, object]
) -> None:
    vector_type = "sparse" if index_type == "inverted" else "dense"
    with pytest.raises(PydanticValidationError):
        TenantVectorField.model_validate(
            {
                "name": "vector",
                "type": vector_type,
                "dim": 2,
                "index": {"type": index_type, "params": params},
            }
        )


@pytest.mark.parametrize(
    ("index_type", "params"),
    [
        ("flat", {"metric": "euclidean"}),
        ("hnsw", {"m": 16, "ef_construction": 200, "prenormalize": True}),
        ("inverted", {"k1": 1.2, "b": 0.75}),
    ],
)
def test_accepts_bounded_canonical_index_params(
    index_type: str, params: dict[str, object]
) -> None:
    vector_type = "sparse" if index_type == "inverted" else "dense"
    field = TenantVectorField.model_validate(
        {
            "name": "vector",
            "type": vector_type,
            "dim": 2,
            "index": {"type": index_type, "params": params},
        }
    )
    assert field.index.params == params


def test_schema_field_and_dimension_limits_match_server_admission() -> None:
    field = {
        "name": "vector",
        "type": "dense",
        "dim": 65_536,
        "index": {"type": "flat"},
    }
    TenantVectorField.model_validate(field)

    with pytest.raises(PydanticValidationError):
        TenantVectorField.model_validate({**field, "dim": 65_537})

    fields = [{**field, "name": f"vector_{index}"} for index in range(8)]
    TenantCollectionSchema(name="bounded", fields=fields)
    with pytest.raises(PydanticValidationError):
        TenantCollectionSchema(
            name="too_many_fields",
            fields=[*fields, {**field, "name": "vector_8"}],
        )


def test_rejects_search_field_count_outside_canonical_contract() -> None:
    with pytest.raises(PydanticValidationError):
        TenantSearchRequest.model_validate(
            {
                "queries": {
                    "dense_a": [0.1, 0.2],
                    "dense_b": [0.2, 0.3],
                    "sparse": {"indices": [1], "values": [1.0], "dim": 2},
                }
            }
        )


def test_multi_field_search_requires_valid_hybrid_params() -> None:
    with pytest.raises(PydanticValidationError, match="require hybrid_params"):
        TenantSearchRequest.model_validate(
            {"queries": {"dense": [0.1], "sparse": {"1": 1.0}}}
        )

    with pytest.raises(PydanticValidationError, match="unknown query fields"):
        TenantSearchRequest.model_validate(
            {
                "queries": {"dense": [0.1], "sparse": {"1": 1.0}},
                "hybrid_params": {
                    "strategy": "weighted",
                    "weights": {"dense": 1.0, "missing": 1.0},
                },
            }
        )

    for invalid_weights in (
        {"dense": -1.0},
        {"dense": float("nan")},
        {"dense": float("inf")},
        {"dense": 0.0, "sparse": 0.0},
    ):
        with pytest.raises(PydanticValidationError):
            TenantSearchRequest.model_validate(
                {
                    "queries": {"dense": [0.1], "sparse": {"1": 1.0}},
                    "hybrid_params": {
                        "strategy": "weighted",
                        "weights": invalid_weights,
                    },
                }
            )

    request = TenantSearchRequest.model_validate(
        {
            "queries": {"dense": [0.1], "sparse": {"1": 1.0}},
            "hybrid_params": {
                "strategy": "weighted",
                "weights": {"dense": 0.8, "sparse": 0.2},
            },
        }
    )
    assert request.hybrid_params is not None


class TestTenantV3Sync:
    def test_collection_lifecycle_exact_contract(self) -> None:
        with respx.mock:
            create = respx.post(f"{ROOT}/collections").mock(
                return_value=httpx.Response(
                    201,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "message": "collection created",
                    },
                )
            )
            listing = respx.get(f"{ROOT}/collections").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "count": 1,
                        "collections": [COLLECTION_INFO],
                    },
                )
            )
            get = respx.get(f"{ROOT}/collections/papers").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "collection": COLLECTION_INFO,
                    },
                )
            )
            delete = respx.delete(f"{ROOT}/collections/papers").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "message": "collection deleted",
                    },
                )
            )

            with DeepDataClient(BASE, retry=None) as client:
                tenant = client.tenant(TENANT)
                created = tenant.create_collection(
                    "papers",
                    [FIELD],
                    metadata={"owner": "sdk"},
                    description="Research papers",
                )
                collections = tenant.list_collections()
                collection = tenant.get_collection("papers")
                deleted = tenant.delete_collection("papers")

            assert isinstance(created, TenantCollectionMutationResponse)
            assert isinstance(collections, TenantCollectionListResponse)
            assert isinstance(collection, TenantGetCollectionResponse)
            assert isinstance(deleted, TenantCollectionMutationResponse)
            assert collection.collection.doc_count == 2
            assert _body(create) == {
                "name": "papers",
                "fields": [FIELD],
                "metadata": {"owner": "sdk"},
                "description": "Research papers",
            }
            assert listing.call_count == get.call_count == delete.call_count == 1

    def test_document_mutations_exact_contract(self) -> None:
        with respx.mock:
            insert = respx.post(f"{ROOT}/collections/papers/docs").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "id": 41,
                        "message": "document added",
                    },
                )
            )
            batch = respx.post(f"{ROOT}/collections/papers/docs/batch").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "ids": [42, 43],
                        "inserted": 2,
                    },
                )
            )
            delete = respx.delete(f"{ROOT}/collections/papers/docs").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "message": "document 41 deleted",
                    },
                )
            )

            with DeepDataClient(BASE, retry=None) as client:
                tenant = client.tenant(TENANT)
                inserted = tenant.insert(
                    "papers",
                    id=41,
                    vectors={"embedding": [0.1, 0.2]},
                    metadata={"topic": "ml"},
                )
                batched = tenant.batch_insert(
                    "papers",
                    [
                        {
                            "id": 42,
                            "vectors": {"embedding": [0.2, 0.3]},
                            "metadata": {"topic": "db"},
                        },
                        TenantDocumentInput(
                            id=43, vectors={"embedding": [0.3, 0.4]}
                        ),
                    ],
                )
                deleted = tenant.delete_document("papers", 41)

            assert isinstance(inserted, TenantInsertResponse)
            assert isinstance(batched, TenantBatchInsertResponse)
            assert isinstance(deleted, TenantDeleteDocumentResponse)
            assert batched.ids == [42, 43]
            assert _body(insert) == {
                "id": 41,
                "vectors": {"embedding": [0.1, 0.2]},
                "metadata": {"topic": "ml"},
            }
            assert _body(batch) == {
                "documents": [
                    {
                        "id": 42,
                        "vectors": {"embedding": [0.2, 0.3]},
                        "metadata": {"topic": "db"},
                    },
                    {"id": 43, "vectors": {"embedding": [0.3, 0.4]}},
                ]
            }
            assert _body(delete) == {"doc_id": 41}

    def test_search_and_info_exact_contract(self) -> None:
        with respx.mock:
            search = respx.post(f"{ROOT}/collections/papers/search").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "documents": [{"id": 41, "metadata": {"topic": "ml"}}],
                        "scores": [0.98],
                        "candidates_examined": 7,
                    },
                )
            )
            info = respx.get(ROOT).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "collection_count": 1,
                        "total_documents": 2,
                        "collections": {
                            "papers": {
                                "Name": "papers",
                                "DocCount": 2,
                                "FieldCount": 1,
                            }
                        },
                    },
                )
            )

            with DeepDataClient(BASE, retry=None) as client:
                tenant = client.tenant(TENANT)
                result = tenant.search(
                    "papers",
                    queries={"embedding": [0.1, 0.2]},
                    top_k=3,
                    ef_search=32,
                    filters={"topic": {"$eq": "ml"}},
                    hybrid_params={
                        "strategy": "weighted",
                        "weights": {"embedding": 1.0},
                    },
                    include_vectors=False,
                )
                tenant_info = tenant.info()

            assert isinstance(result, TenantSearchResponse)
            assert isinstance(tenant_info, TenantInfoResponse)
            assert result.documents[0].vectors is None
            assert tenant_info.collections is not None
            assert tenant_info.collections["papers"].doc_count == 2
            assert _body(search) == {
                "queries": {"embedding": [0.1, 0.2]},
                "top_k": 3,
                "ef_search": 32,
                "include_vectors": False,
                "filters": {"topic": {"$eq": "ml"}},
                "hybrid_params": {
                    "strategy": "weighted",
                    "weights": {"embedding": 1.0},
                },
            }
            assert info.call_count == 1

    @pytest.mark.parametrize(
        "tenant_id", ["", "a" * 65, "../escape", "space here", "ténant"]
    )
    def test_rejects_unsafe_tenant_ids(self, tenant_id: str) -> None:
        with DeepDataClient(BASE, retry=None) as client:
            with pytest.raises(ValueError, match="tenant_id must be"):
                client.tenant(tenant_id)

    def test_rejects_unsafe_collection_and_invalid_batch_before_io(self) -> None:
        with respx.mock:
            batch = respx.post(f"{ROOT}/collections/papers/docs/batch").mock(
                return_value=httpx.Response(500)
            )
            with DeepDataClient(BASE, retry=None) as client:
                tenant = client.tenant(TENANT)
                with pytest.raises(ValueError, match="collection must be"):
                    tenant.insert("../papers", vectors={"embedding": [0.1, 0.2]})
                with pytest.raises(PydanticValidationError):
                    tenant.batch_insert(
                        "papers",
                        [
                            {"vectors": {"embedding": [0.1, 0.2]}},
                            {"id": "not-an-int", "vectors": {"embedding": [0.2, 0.3]}},
                        ],
                    )
            assert not batch.called

    @pytest.mark.parametrize(
        "field",
        [
            {
                "name": "vector",
                "type": "dense",
                "dim": 2,
                "index": {"type": "inverted"},
            },
            {
                "name": "vector",
                "type": "sparse",
                "dim": 2,
                "index": {"type": "hnsw"},
            },
            {
                "name": "vector",
                "type": "dense",
                "dim": 2,
                "index": {"type": "ivf"},
            },
            {
                "name": "vector",
                "type": "dense",
                "dim": 2,
                "index": {"type": "diskann"},
            },
        ],
    )
    def test_rejects_indexes_outside_frozen_contract_before_io(
        self, field: dict[str, object]
    ) -> None:
        with respx.mock:
            create = respx.post(f"{ROOT}/collections").mock(
                return_value=httpx.Response(500)
            )
            with DeepDataClient(BASE, retry=None) as client:
                with pytest.raises(PydanticValidationError):
                    client.tenant(TENANT).create_collection("papers", [field])
            assert not create.called

    def test_preserves_typed_http_errors(self) -> None:
        with respx.mock:
            respx.get(f"{ROOT}/collections").mock(
                return_value=httpx.Response(403, text="forbidden")
            )
            with DeepDataClient(BASE, retry=None) as client:
                with pytest.raises(DeepDataPermissionError):
                    client.tenant(TENANT).list_collections()

    def test_does_not_retry_mutation_after_ambiguous_transport_failure(self) -> None:
        with respx.mock:
            route = respx.post(f"{ROOT}/collections/papers/docs").mock(
                side_effect=[
                    httpx.ReadError("response lost after request write"),
                    httpx.Response(
                        200,
                        json={
                            "status": "success",
                            "tenant_id": TENANT,
                            "id": 42,
                            "message": "document added",
                        },
                    ),
                ]
            )
            with DeepDataClient(
                BASE,
                retry=RetryConfig(max_retries=2, initial_delay=0, max_delay=0),
            ) as client:
                with pytest.raises(DeepDataConnectionError):
                    client.tenant(TENANT).insert(
                        "papers", vectors={"embedding": [0.1, 0.2]}
                    )
            assert route.call_count == 1


@pytest.mark.asyncio
class TestTenantV3Async:
    async def test_does_not_retry_mutation_after_ambiguous_transport_failure(
        self,
    ) -> None:
        with respx.mock:
            route = respx.post(f"{ROOT}/collections/papers/docs").mock(
                side_effect=[
                    httpx.ReadError("response lost after request write"),
                    httpx.Response(
                        200,
                        json={
                            "status": "success",
                            "tenant_id": TENANT,
                            "id": 42,
                            "message": "document added",
                        },
                    ),
                ]
            )
            async with AsyncDeepDataClient(
                BASE,
                retry=RetryConfig(max_retries=2, initial_delay=0, max_delay=0),
            ) as client:
                with pytest.raises(DeepDataConnectionError):
                    await client.tenant(TENANT).insert(
                        "papers", vectors={"embedding": [0.1, 0.2]}
                    )
            assert route.call_count == 1

    async def test_collection_lifecycle_exact_contract(self) -> None:
        with respx.mock:
            create = respx.post(f"{ROOT}/collections").mock(
                return_value=httpx.Response(
                    201,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "message": "collection created",
                    },
                )
            )
            listing = respx.get(f"{ROOT}/collections").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "count": 1,
                        "collections": [COLLECTION_INFO],
                    },
                )
            )
            get = respx.get(f"{ROOT}/collections/papers").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "collection": COLLECTION_INFO,
                    },
                )
            )
            delete = respx.delete(f"{ROOT}/collections/papers").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "message": "collection deleted",
                    },
                )
            )

            async with AsyncDeepDataClient(BASE, retry=None) as client:
                tenant = client.tenant(TENANT)
                created = await tenant.create_collection(
                    "papers", [FIELD], description="Research papers"
                )
                collections = await tenant.list_collections()
                collection = await tenant.get_collection("papers")
                deleted = await tenant.delete_collection("papers")

            assert isinstance(created, TenantCollectionMutationResponse)
            assert isinstance(collections, TenantCollectionListResponse)
            assert isinstance(collection, TenantGetCollectionResponse)
            assert isinstance(deleted, TenantCollectionMutationResponse)
            assert _body(create) == {
                "name": "papers",
                "fields": [FIELD],
                "description": "Research papers",
            }
            assert listing.call_count == get.call_count == delete.call_count == 1

    async def test_document_mutations_exact_contract(self) -> None:
        with respx.mock:
            insert = respx.post(f"{ROOT}/collections/papers/docs").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "id": 51,
                        "message": "document added",
                    },
                )
            )
            batch = respx.post(f"{ROOT}/collections/papers/docs/batch").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "ids": [52],
                        "inserted": 1,
                    },
                )
            )
            delete = respx.delete(f"{ROOT}/collections/papers/docs").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "message": "document 51 deleted",
                    },
                )
            )

            async with AsyncDeepDataClient(BASE, retry=None) as client:
                tenant = client.tenant(TENANT)
                inserted = await tenant.insert(
                    "papers", id=51, vectors={"embedding": [0.1, 0.2]}
                )
                batched = await tenant.batch_insert(
                    "papers",
                    [{"id": 52, "vectors": {"embedding": [0.2, 0.3]}}],
                )
                deleted = await tenant.delete_document("papers", 51)

            assert isinstance(inserted, TenantInsertResponse)
            assert isinstance(batched, TenantBatchInsertResponse)
            assert isinstance(deleted, TenantDeleteDocumentResponse)
            assert _body(insert) == {
                "id": 51,
                "vectors": {"embedding": [0.1, 0.2]},
            }
            assert _body(batch) == {
                "documents": [
                    {"id": 52, "vectors": {"embedding": [0.2, 0.3]}}
                ]
            }
            assert _body(delete) == {"doc_id": 51}

    async def test_search_info_and_errors_exact_contract(self) -> None:
        with respx.mock:
            search = respx.post(f"{ROOT}/collections/papers/search").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "documents": [
                            {
                                "id": 51,
                                "vectors": {"embedding": [0.1, 0.2]},
                            }
                        ],
                        "scores": [0.99],
                        "candidates_examined": 3,
                    },
                )
            )
            info = respx.get(ROOT).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "collection_count": 0,
                        "total_documents": 0,
                    },
                )
            )
            forbidden = respx.get(f"{ROOT}/collections").mock(
                return_value=httpx.Response(403, text="forbidden")
            )

            async with AsyncDeepDataClient(BASE, retry=None) as client:
                tenant = client.tenant(TENANT)
                result = await tenant.search(
                    "papers",
                    queries={"embedding": [0.1, 0.2]},
                    top_k=1,
                    filters={"topic": "ml"},
                    hybrid_params={"strategy": "rrf", "rrf_constant": 60.0},
                    include_vectors=True,
                )
                tenant_info = await tenant.info()
                with pytest.raises(DeepDataPermissionError):
                    await tenant.list_collections()

            assert isinstance(result, TenantSearchResponse)
            assert isinstance(tenant_info, TenantInfoResponse)
            assert result.documents[0].vectors == {"embedding": [0.1, 0.2]}
            assert _body(search) == {
                "queries": {"embedding": [0.1, 0.2]},
                "top_k": 1,
                "include_vectors": True,
                "filters": {"topic": "ml"},
                "hybrid_params": {"strategy": "rrf", "rrf_constant": 60.0},
            }
            assert info.call_count == forbidden.call_count == 1

    async def test_validation_happens_before_io(self) -> None:
        with respx.mock:
            batch = respx.post(f"{ROOT}/collections/papers/docs/batch").mock(
                return_value=httpx.Response(500)
            )
            create = respx.post(f"{ROOT}/collections").mock(
                return_value=httpx.Response(500)
            )
            async with AsyncDeepDataClient(BASE, retry=None) as client:
                with pytest.raises(ValueError, match="tenant_id must be"):
                    client.tenant("../escape")
                tenant = client.tenant(TENANT)
                with pytest.raises(ValueError, match="collection must be"):
                    await tenant.get_collection("papers/escape")
                with pytest.raises(PydanticValidationError):
                    await tenant.batch_insert(
                        "papers",
                        [{"id": True, "vectors": {"embedding": [0.1, 0.2]}}],
                    )
                with pytest.raises(PydanticValidationError):
                    await tenant.create_collection(
                        "papers",
                        [
                            {
                                "name": "vector",
                                "type": "sparse",
                                "dim": 2,
                                "index": {"type": "flat"},
                            }
                        ],
                    )
            assert not batch.called
            assert not create.called
