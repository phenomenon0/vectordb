"""Tests for Pydantic model serialization round-trips."""

from __future__ import annotations

from deepdata.models import (
    BatchDoc,
    BatchInsertRequest,
    DeleteRequest,
    FieldSchema,
    HealthResponse,
    InsertRequest,
    InsertResponse,
    RangeFilter,
    ScrollRequest,
    SearchRequest,
    SearchResult,
    SparseInsertRequest,
)


class TestInsertRequest:
    def test_minimal(self) -> None:
        req = InsertRequest(doc="Hello")
        data = req.model_dump(exclude_none=True)
        assert data == {"doc": "Hello", "upsert": False}

    def test_full(self) -> None:
        req = InsertRequest(
            doc="Hello",
            id="doc-1",
            meta={"k": "v"},
            upsert=True,
            collection="test",
        )
        data = req.model_dump(exclude_none=True)
        assert data["id"] == "doc-1"
        assert data["meta"] == {"k": "v"}
        assert data["upsert"] is True
        assert data["collection"] == "test"

    def test_roundtrip(self) -> None:
        req = InsertRequest(doc="Test", id="1", meta={"a": "b"})
        data = req.model_dump()
        restored = InsertRequest.model_validate(data)
        assert restored == req


class TestSearchRequest:
    def test_minimal(self) -> None:
        req = SearchRequest(query="test")
        data = req.model_dump(exclude_none=True)
        assert data["query"] == "test"
        assert data["top_k"] == 10
        assert data["include_meta"] is False

    def test_with_range_filters(self) -> None:
        req = SearchRequest(
            query="test",
            meta_ranges=[RangeFilter(key="year", min=2020, max=2026)],
        )
        data = req.model_dump(exclude_none=True)
        assert len(data["meta_ranges"]) == 1
        assert data["meta_ranges"][0]["key"] == "year"


class TestSearchResult:
    def test_empty(self) -> None:
        result = SearchResult()
        assert result.ids == []
        assert result.docs == []
        assert result.scores == []
        assert result.meta is None

    def test_with_data(self) -> None:
        result = SearchResult.model_validate({
            "ids": ["a", "b"],
            "docs": ["Doc A", "Doc B"],
            "scores": [0.9, 0.8],
            "meta": [{"k": "v1"}, {"k": "v2"}],
            "next": "token-123",
        })
        assert len(result.ids) == 2
        assert result.next == "token-123"

    def test_roundtrip(self) -> None:
        original = SearchResult(
            ids=["a"], docs=["Doc A"], scores=[0.95],
            meta=[{"author": "Alice"}],
        )
        data = original.model_dump()
        restored = SearchResult.model_validate(data)
        assert restored == original


class TestBatchInsertRequest:
    def test_serialization(self) -> None:
        req = BatchInsertRequest(
            docs=[
                BatchDoc(doc="Doc 1", meta={"year": "2026"}),
                BatchDoc(doc="Doc 2", id="custom-id"),
            ],
            upsert=True,
        )
        data = req.model_dump(exclude_none=True)
        assert len(data["docs"]) == 2
        assert data["upsert"] is True


class TestHealthResponse:
    def test_from_server(self) -> None:
        resp = HealthResponse.model_validate({
            "ok": True,
            "total": 1000,
            "active": 950,
            "deleted": 50,
            "hnsw_ids": 950,
            "checksum": "abc",
            "wal_bytes": 2048,
        })
        assert resp.ok is True
        assert resp.total == 1000

    def test_defaults(self) -> None:
        resp = HealthResponse()
        assert resp.ok is True
        assert resp.total == 0


class TestFieldSchema:
    def test_dense_field(self) -> None:
        f = FieldSchema(name="embedding", type="dense", dim=384, index_type="hnsw")
        data = f.model_dump(exclude_none=True)
        assert data == {"name": "embedding", "type": "dense", "dim": 384, "index_type": "hnsw"}

    def test_sparse_field(self) -> None:
        f = FieldSchema(name="sparse", type="sparse")
        data = f.model_dump(exclude_none=True)
        assert data == {"name": "sparse", "type": "sparse"}


class TestSparseInsertRequest:
    def test_serialization(self) -> None:
        req = SparseInsertRequest(
            doc="Sparse doc",
            indices=[1, 5, 10],
            values=[1.0, 2.0, 0.5],
            dimension=100,
        )
        data = req.model_dump(exclude_none=True)
        assert data["indices"] == [1, 5, 10]
        assert data["values"] == [1.0, 2.0, 0.5]
        assert data["dimension"] == 100


class TestDeleteRequest:
    def test_serialization(self) -> None:
        req = DeleteRequest(id="doc-1")
        data = req.model_dump()
        assert data["id"] == "doc-1"


class TestScrollRequest:
    def test_minimal(self) -> None:
        req = ScrollRequest()
        data = req.model_dump(exclude_none=True)
        assert data == {}

    def test_full(self) -> None:
        req = ScrollRequest(collection="papers", limit=10, offset=20)
        data = req.model_dump(exclude_none=True)
        assert data == {"collection": "papers", "limit": 10, "offset": 20}
