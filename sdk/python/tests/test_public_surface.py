"""Public-surface and lifecycle tests for the canonical-only SDK."""

from __future__ import annotations

import inspect

import httpx
import pytest
import respx

import deepdata
import deepdata.models as models
from deepdata import AsyncDeepDataClient, DeepDataClient, RetryConfig


BASE = "http://testserver:8080"
TENANT = "tenant-123"
ROOT = f"{BASE}/v3/tenants/{TENANT}"

LEGACY_METHODS = {
    "insert",
    "batch_insert",
    "search",
    "delete",
    "health",
    "scroll",
    "list_collections",
    "create_collection",
    "get_collection",
    "delete_collection",
    "collection_stats",
    "compact",
    "insert_sparse",
}

LEGACY_MODELS = {
    "InsertRequest",
    "InsertResponse",
    "BatchDoc",
    "BatchInsertRequest",
    "BatchInsertResponse",
    "RangeFilter",
    "SearchRequest",
    "SearchResult",
    "DeleteRequest",
    "DeleteResponse",
    "ScrollRequest",
    "ScrollResponse",
    "HealthResponse",
    "CollectionListResponse",
    "FieldSchema",
    "CollectionSchema",
    "CollectionInfo",
    "CollectionStatsResponse",
    "CompactResponse",
    "SparseInsertRequest",
}


@pytest.mark.parametrize("client_type", [DeepDataClient, AsyncDeepDataClient])
def test_root_client_has_no_unscoped_database_methods(client_type: type[object]) -> None:
    for method in LEGACY_METHODS:
        assert not hasattr(client_type, method), method

    signature = inspect.signature(client_type)
    assert "tenant_id" not in signature.parameters
    assert hasattr(client_type, "tenant")
    assert hasattr(client_type, "close")


def test_only_canonical_models_are_exported() -> None:
    for name in LEGACY_MODELS:
        assert name not in deepdata.__all__
        assert not hasattr(deepdata, name)
        assert not hasattr(models, name)

    model_exports = {
        name for name in deepdata.__all__ if name.startswith("Tenant")
    }
    assert model_exports
    assert all(hasattr(models, name) or name.endswith("Client") for name in model_exports)


def test_sync_lifecycle_and_canonical_auth_headers() -> None:
    with respx.mock:
        route = respx.get(ROOT).mock(
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
        client = DeepDataClient(
            BASE,
            api_token="strong-test-token",
            headers={"X-Trace-ID": "trace-1"},
            retry=None,
        )
        with client as active:
            assert active is client
            assert active.tenant(TENANT).info().tenant_id == TENANT

        assert client._http.is_closed
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer strong-test-token"
        assert request.headers["X-Trace-ID"] == "trace-1"
        assert "X-Tenant-ID" not in request.headers


@pytest.mark.asyncio
async def test_async_lifecycle_and_canonical_auth_headers() -> None:
    with respx.mock:
        route = respx.get(ROOT).mock(
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
        client = AsyncDeepDataClient(
            BASE,
            api_token="strong-test-token",
            headers={"X-Trace-ID": "trace-2"},
            retry=None,
        )
        async with client as active:
            assert active is client
            assert (await active.tenant(TENANT).info()).tenant_id == TENANT

        assert client._http.is_closed
        request = route.calls[0].request
        assert request.headers["Authorization"] == "Bearer strong-test-token"
        assert request.headers["X-Trace-ID"] == "trace-2"
        assert "X-Tenant-ID" not in request.headers


def test_sync_retries_canonical_get_and_read_only_search() -> None:
    retry = RetryConfig(
        max_retries=1,
        initial_delay=0,
        max_delay=0,
        jitter_percent=0,
    )
    with respx.mock:
        listing = respx.get(f"{ROOT}/collections").mock(
            side_effect=[
                httpx.Response(503, text="temporarily unavailable"),
                httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "count": 0,
                        "collections": [],
                    },
                ),
            ]
        )
        search = respx.post(f"{ROOT}/collections/docs/search").mock(
            side_effect=[
                httpx.Response(503, text="temporarily unavailable"),
                httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "documents": [],
                        "scores": [],
                        "candidates_examined": 0,
                    },
                ),
            ]
        )
        with DeepDataClient(BASE, retry=retry) as client:
            tenant = client.tenant(TENANT)
            assert tenant.list_collections().count == 0
            assert tenant.search("docs", queries={"embedding": [0.1]}).documents == []

        assert listing.call_count == 2
        assert search.call_count == 2


@pytest.mark.asyncio
async def test_async_retries_canonical_get_and_read_only_search() -> None:
    retry = RetryConfig(
        max_retries=1,
        initial_delay=0,
        max_delay=0,
        jitter_percent=0,
    )
    with respx.mock:
        listing = respx.get(f"{ROOT}/collections").mock(
            side_effect=[
                httpx.Response(503, text="temporarily unavailable"),
                httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "count": 0,
                        "collections": [],
                    },
                ),
            ]
        )
        search = respx.post(f"{ROOT}/collections/docs/search").mock(
            side_effect=[
                httpx.Response(503, text="temporarily unavailable"),
                httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "tenant_id": TENANT,
                        "documents": [],
                        "scores": [],
                        "candidates_examined": 0,
                    },
                ),
            ]
        )
        async with AsyncDeepDataClient(BASE, retry=retry) as client:
            tenant = client.tenant(TENANT)
            assert (await tenant.list_collections()).count == 0
            result = await tenant.search("docs", queries={"embedding": [0.1]})
            assert result.documents == []

        assert listing.call_count == 2
        assert search.call_count == 2
