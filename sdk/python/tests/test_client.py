"""Unit tests for the synchronous DeepData client."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from deepdata import (
    DeepDataClient,
    InsertResponse,
    BatchInsertResponse,
    SearchResult,
    DeleteResponse,
    HealthResponse,
    CollectionListResponse,
    CompactResponse,
    ScrollResponse,
    CollectionStatsResponse,
)
from deepdata.errors import (
    APIError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from deepdata._utils import RetryConfig

BASE = "http://testserver:8080"


def _client(**kwargs) -> DeepDataClient:
    kwargs.setdefault("retry", None)
    return DeepDataClient(BASE, **kwargs)


class TestInsert:
    def test_insert_basic(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/insert").mock(
                return_value=httpx.Response(200, json={"id": "abc-123"})
            )
            client = _client()
            result = client.insert("Hello world")
            assert isinstance(result, InsertResponse)
            assert result.id == "abc-123"

    def test_insert_with_options(self) -> None:
        with respx.mock:
            route = respx.post(f"{BASE}/insert").mock(
                return_value=httpx.Response(200, json={"id": "doc-1"})
            )
            client = _client()
            client.insert(
                "Test doc",
                id="doc-1",
                meta={"author": "Alice"},
                upsert=True,
                collection="papers",
            )
            data = json.loads(route.calls[0].request.content)
            assert data["doc"] == "Test doc"
            assert data["id"] == "doc-1"
            assert data["meta"] == {"author": "Alice"}
            assert data["upsert"] is True
            assert data["collection"] == "papers"

    def test_insert_omits_none_fields(self) -> None:
        with respx.mock:
            route = respx.post(f"{BASE}/insert").mock(
                return_value=httpx.Response(200, json={"id": "x"})
            )
            client = _client()
            client.insert("Simple doc")
            data = json.loads(route.calls[0].request.content)
            assert "id" not in data
            assert "meta" not in data
            assert "upsert" not in data
            assert "collection" not in data


class TestBatchInsert:
    def test_batch_insert(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/batch_insert").mock(
                return_value=httpx.Response(200, json={"ids": ["a", "b"]})
            )
            client = _client()
            result = client.batch_insert([
                {"doc": "Doc 1", "meta": {"year": "2026"}},
                {"doc": "Doc 2"},
            ])
            assert isinstance(result, BatchInsertResponse)
            assert result.ids == ["a", "b"]

    def test_batch_insert_with_upsert(self) -> None:
        with respx.mock:
            route = respx.post(f"{BASE}/batch_insert").mock(
                return_value=httpx.Response(200, json={"ids": ["a"]})
            )
            client = _client()
            client.batch_insert([{"doc": "Doc 1"}], upsert=True)
            data = json.loads(route.calls[0].request.content)
            assert data["upsert"] is True

    def test_batch_insert_with_errors(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/batch_insert").mock(
                return_value=httpx.Response(200, json={
                    "ids": ["a"],
                    "errors": ["doc 1: empty document"],
                })
            )
            client = _client()
            result = client.batch_insert([{"doc": "OK"}, {"doc": ""}])
            assert result.ids == ["a"]
            assert result.errors == ["doc 1: empty document"]


class TestSearch:
    def test_search_basic(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/query").mock(
                return_value=httpx.Response(200, json={
                    "ids": ["a", "b"],
                    "docs": ["Doc A", "Doc B"],
                    "scores": [0.95, 0.87],
                })
            )
            client = _client()
            result = client.search("test query")
            assert isinstance(result, SearchResult)
            assert result.ids == ["a", "b"]
            assert result.docs == ["Doc A", "Doc B"]
            assert result.scores == [0.95, 0.87]

    def test_search_with_all_options(self) -> None:
        with respx.mock:
            route = respx.post(f"{BASE}/query").mock(
                return_value=httpx.Response(200, json={"ids": [], "docs": [], "scores": []})
            )
            client = _client()
            client.search(
                "query",
                top_k=20,
                mode="ann",
                collection="papers",
                meta={"author": "Alice"},
                meta_not={"status": "draft"},
                include_meta=True,
                hybrid_alpha=0.7,
                ef_search=128,
                offset=10,
                limit=5,
            )
            data = json.loads(route.calls[0].request.content)
            assert data["query"] == "query"
            assert data["top_k"] == 20
            assert data["mode"] == "ann"
            assert data["collection"] == "papers"
            assert data["meta"] == {"author": "Alice"}
            assert data["meta_not"] == {"status": "draft"}
            assert data["include_meta"] is True
            assert data["hybrid_alpha"] == 0.7
            assert data["ef_search"] == 128
            assert data["offset"] == 10
            assert data["limit"] == 5

    def test_search_with_meta(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/query").mock(
                return_value=httpx.Response(200, json={
                    "ids": ["a"],
                    "docs": ["Doc A"],
                    "scores": [0.9],
                    "meta": [{"author": "Alice"}],
                })
            )
            client = _client()
            result = client.search("test", include_meta=True)
            assert result.meta == [{"author": "Alice"}]

    def test_search_with_pagination(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/query").mock(
                return_value=httpx.Response(200, json={
                    "ids": ["a"],
                    "docs": ["Doc A"],
                    "scores": [0.9],
                    "next": "token-abc",
                })
            )
            client = _client()
            result = client.search("test", page_size=1)
            assert result.next == "token-abc"


class TestDelete:
    def test_delete(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/delete").mock(
                return_value=httpx.Response(200, json={"deleted": "abc-123"})
            )
            client = _client()
            result = client.delete("abc-123")
            assert isinstance(result, DeleteResponse)
            assert result.deleted == "abc-123"


class TestHealth:
    def test_health(self) -> None:
        with respx.mock:
            respx.get(f"{BASE}/health").mock(
                return_value=httpx.Response(200, json={
                    "ok": True,
                    "total": 1000,
                    "active": 950,
                    "deleted": 50,
                    "hnsw_ids": 950,
                    "checksum": "abc123",
                    "wal_bytes": 1024,
                })
            )
            client = _client()
            result = client.health()
            assert isinstance(result, HealthResponse)
            assert result.ok is True
            assert result.total == 1000
            assert result.active == 950
            assert result.deleted == 50


class TestScroll:
    def test_scroll(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/scroll").mock(
                return_value=httpx.Response(200, json={
                    "ids": ["a", "b"],
                    "docs": ["Doc A", "Doc B"],
                    "total": 100,
                    "next_offset": 2,
                })
            )
            client = _client()
            result = client.scroll(limit=2)
            assert isinstance(result, ScrollResponse)
            assert result.ids == ["a", "b"]
            assert result.total == 100
            assert result.next_offset == 2


class TestCollections:
    def test_list_collections(self) -> None:
        with respx.mock:
            respx.get(f"{BASE}/admin/collection/list").mock(
                return_value=httpx.Response(200, json={
                    "status": "ok",
                    "count": 2,
                    "collections": [
                        {"name": "default"},
                        {"name": "papers"},
                    ],
                })
            )
            client = _client()
            result = client.list_collections()
            assert isinstance(result, CollectionListResponse)
            assert result.count == 2
            assert len(result.collections) == 2

    def test_create_collection(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/v2/collections").mock(
                return_value=httpx.Response(201, json={
                    "status": "success",
                    "message": 'collection "papers" created',
                })
            )
            client = _client()
            result = client.create_collection("papers", fields=[
                {"name": "embedding", "type": "dense", "dim": 384, "index_type": "hnsw"},
            ])
            assert result["status"] == "success"

    def test_delete_collection(self) -> None:
        with respx.mock:
            respx.delete(f"{BASE}/v2/collections/papers").mock(
                return_value=httpx.Response(200, json={
                    "status": "success",
                    "message": 'collection "papers" deleted',
                })
            )
            client = _client()
            result = client.delete_collection("papers")
            assert result["status"] == "success"

    def test_collection_stats(self) -> None:
        with respx.mock:
            respx.get(f"{BASE}/v2/collections/papers/stats").mock(
                return_value=httpx.Response(200, json={
                    "status": "success",
                    "name": "papers",
                    "doc_count": 42,
                })
            )
            client = _client()
            result = client.collection_stats("papers")
            assert isinstance(result, CollectionStatsResponse)
            assert result.name == "papers"
            assert result.doc_count == 42


class TestCompact:
    def test_compact(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/compact").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            client = _client()
            result = client.compact()
            assert isinstance(result, CompactResponse)
            assert result.ok is True


class TestSparseInsert:
    def test_insert_sparse(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/insert/sparse").mock(
                return_value=httpx.Response(200, json={"id": "sparse-1"})
            )
            client = _client()
            result = client.insert_sparse(
                "Sparse doc",
                indices=[1, 5, 10],
                values=[1.0, 2.0, 0.5],
                dimension=100,
            )
            assert result.id == "sparse-1"


class TestErrorHandling:
    def test_401_raises_authentication_error(self) -> None:
        with respx.mock:
            respx.get(f"{BASE}/health").mock(
                return_value=httpx.Response(401, text="unauthorized")
            )
            with pytest.raises(AuthenticationError):
                _client().health()

    def test_404_raises_not_found(self) -> None:
        with respx.mock:
            respx.get(f"{BASE}/v2/collections/missing").mock(
                return_value=httpx.Response(404, text="collection not found")
            )
            with pytest.raises(NotFoundError):
                _client().get_collection("missing")

    def test_422_raises_validation_error(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/insert").mock(
                return_value=httpx.Response(422, text="doc required")
            )
            with pytest.raises(ValidationError):
                _client().insert("")

    def test_429_raises_rate_limit_error(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/query").mock(
                return_value=httpx.Response(429, text="rate limited")
            )
            with pytest.raises(RateLimitError):
                _client().search("test")

    def test_500_raises_server_error(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/insert").mock(
                return_value=httpx.Response(500, text="internal error")
            )
            with pytest.raises(ServerError):
                _client().insert("test")


class TestAuth:
    def test_auth_headers_sent(self) -> None:
        with respx.mock:
            route = respx.get(f"{BASE}/health").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            client = DeepDataClient(
                BASE,
                api_token="sk-test-token",
                tenant_id="tenant-123",
                retry=None,
            )
            client.health()
            req = route.calls[0].request
            assert req.headers["authorization"] == "Bearer sk-test-token"
            assert req.headers["x-tenant-id"] == "tenant-123"


class TestRetry:
    def test_retry_on_500(self) -> None:
        with respx.mock:
            route = respx.get(f"{BASE}/health").mock(
                side_effect=[
                    httpx.Response(500, text="error"),
                    httpx.Response(200, json={"ok": True}),
                ]
            )
            client = DeepDataClient(
                BASE,
                retry=RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.05),
            )
            result = client.health()
            assert result.ok is True
            assert route.call_count == 2

    def test_no_retry_on_400(self) -> None:
        with respx.mock:
            route = respx.post(f"{BASE}/insert").mock(
                return_value=httpx.Response(400, text="bad request")
            )
            client = DeepDataClient(
                BASE,
                retry=RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.05),
            )
            with pytest.raises(APIError):
                client.insert("test")
            assert route.call_count == 1

    def test_retry_exhausted(self) -> None:
        with respx.mock:
            route = respx.get(f"{BASE}/health").mock(
                return_value=httpx.Response(500, text="error")
            )
            client = DeepDataClient(
                BASE,
                retry=RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.05),
            )
            with pytest.raises(ServerError):
                client.health()
            assert route.call_count == 3  # 1 initial + 2 retries


class TestContextManager:
    def test_context_manager(self) -> None:
        with DeepDataClient(BASE) as client:
            assert client._http is not None


class TestClientDefaults:
    def test_default_url(self) -> None:
        client = DeepDataClient()
        assert client._base_url == "http://localhost:8080"

    def test_custom_url(self) -> None:
        client = DeepDataClient("http://custom:9090/")
        assert client._base_url == "http://custom:9090"
