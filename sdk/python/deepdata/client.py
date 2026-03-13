"""Synchronous DeepData client using httpx.

Mirrors the Go client API (client/client.go) with Pythonic conventions.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .errors import DeepDataError
from .models import (
    BatchDoc,
    BatchInsertResponse,
    CollectionInfo,
    CollectionListResponse,
    CollectionSchema,
    CollectionStatsResponse,
    CompactResponse,
    DeleteResponse,
    FieldSchema,
    HealthResponse,
    InsertResponse,
    ScrollResponse,
    SearchResult,
    SparseInsertRequest,
)
from ._utils import (
    DEFAULT_RETRY,
    RetryConfig,
    build_headers,
    handle_request_error,
    handle_response,
    retry_delay,
    should_retry,
)


class DeepDataClient:
    """Synchronous HTTP client for DeepData vector database.

    Usage::

        client = DeepDataClient("http://localhost:8080")
        result = client.insert("Hello world", meta={"source": "test"})
        results = client.search("Hello", top_k=5)
        client.close()

    Or as a context manager::

        with DeepDataClient("http://localhost:8080") as client:
            results = client.search("Hello", top_k=5)
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        *,
        api_token: str | None = None,
        tenant_id: str | None = None,
        timeout: float = 15.0,
        retry: RetryConfig | None = DEFAULT_RETRY,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._retry = retry
        self._headers = build_headers(
            token=api_token,
            tenant_id=tenant_id,
            extra=headers,
        )
        self._http = httpx.Client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> DeepDataClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Internal request helper ─────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
    ) -> Any:
        """Execute an HTTP request with retry logic.

        Mirrors doJSON in the Go client.
        """
        last_exc: Exception | None = None
        max_attempts = 1 + (self._retry.max_retries if self._retry else 0)

        for attempt in range(max_attempts):
            if attempt > 0 and self._retry:
                delay = retry_delay(attempt - 1, self._retry)
                time.sleep(delay)

            try:
                response = self._http.request(method, path, json=json)
                data = handle_response(response)

                # Retry on retryable API errors
                from .errors import APIError
                return data

            except DeepDataError as exc:
                last_exc = exc
                from .errors import APIError
                if isinstance(exc, APIError) and should_retry(exc.status_code, attempt, self._retry):
                    continue
                raise

            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < max_attempts - 1 and self._retry:
                    continue
                handle_request_error(exc)

        if last_exc is not None:
            if isinstance(last_exc, DeepDataError):
                raise last_exc
            handle_request_error(last_exc)

    # ── Core Operations (Tier 1) ────────────────────────────────────────

    def insert(
        self,
        doc: str,
        *,
        id: str | None = None,
        meta: dict[str, str] | None = None,
        upsert: bool = False,
        collection: str | None = None,
    ) -> InsertResponse:
        """Insert a document. Returns the assigned ID.

        Mirrors Go Client.Insert().
        """
        payload: dict[str, Any] = {"doc": doc}
        if id is not None:
            payload["id"] = id
        if meta is not None:
            payload["meta"] = meta
        if upsert:
            payload["upsert"] = True
        if collection is not None:
            payload["collection"] = collection

        data = self._request("POST", "/insert", json=payload)
        return InsertResponse.model_validate(data)

    def batch_insert(
        self,
        docs: list[dict[str, Any] | BatchDoc],
        *,
        upsert: bool = False,
    ) -> BatchInsertResponse:
        """Insert multiple documents in one request.

        Mirrors Go Client.BatchInsert().

        Each doc can be a dict with keys: doc, id, meta, collection
        or a BatchDoc instance.
        """
        normalized: list[dict[str, Any]] = []
        for d in docs:
            if isinstance(d, BatchDoc):
                normalized.append(d.model_dump(exclude_none=True))
            else:
                normalized.append(d)

        payload: dict[str, Any] = {"docs": normalized}
        if upsert:
            payload["upsert"] = True

        data = self._request("POST", "/batch_insert", json=payload)
        return BatchInsertResponse.model_validate(data)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: str | None = None,
        collection: str | None = None,
        meta: dict[str, str] | None = None,
        meta_any: list[dict[str, str]] | None = None,
        meta_not: dict[str, str] | None = None,
        meta_ranges: list[dict[str, Any]] | None = None,
        include_meta: bool = False,
        hybrid_alpha: float | None = None,
        score_mode: str | None = None,
        ef_search: int | None = None,
        offset: int | None = None,
        limit: int | None = None,
        page_token: str | None = None,
        page_size: int | None = None,
    ) -> SearchResult:
        """Search for similar documents.

        Mirrors Go Client.Query().
        """
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if mode is not None:
            payload["mode"] = mode
        if collection is not None:
            payload["collection"] = collection
        if meta is not None:
            payload["meta"] = meta
        if meta_any is not None:
            payload["meta_any"] = meta_any
        if meta_not is not None:
            payload["meta_not"] = meta_not
        if meta_ranges is not None:
            payload["meta_ranges"] = meta_ranges
        if include_meta:
            payload["include_meta"] = True
        if hybrid_alpha is not None:
            payload["hybrid_alpha"] = hybrid_alpha
        if score_mode is not None:
            payload["score_mode"] = score_mode
        if ef_search is not None:
            payload["ef_search"] = ef_search
        if offset is not None:
            payload["offset"] = offset
        if limit is not None:
            payload["limit"] = limit
        if page_token is not None:
            payload["page_token"] = page_token
        if page_size is not None:
            payload["page_size"] = page_size

        data = self._request("POST", "/query", json=payload)
        return SearchResult.model_validate(data)

    def delete(self, id: str) -> DeleteResponse:
        """Delete a document by ID.

        Mirrors Go Client.Delete().
        """
        data = self._request("POST", "/delete", json={"id": id})
        return DeleteResponse.model_validate(data)

    def health(self) -> HealthResponse:
        """Check server health.

        Mirrors Go Client.Health().
        """
        data = self._request("GET", "/health")
        return HealthResponse.model_validate(data)

    def scroll(
        self,
        *,
        collection: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> ScrollResponse:
        """Paginated iteration over all documents.

        Mirrors Go Client.Scroll().
        """
        payload: dict[str, Any] = {}
        if collection is not None:
            payload["collection"] = collection
        if limit is not None:
            payload["limit"] = limit
        if offset is not None:
            payload["offset"] = offset

        data = self._request("POST", "/scroll", json=payload)
        return ScrollResponse.model_validate(data)

    # ── Collection Management ───────────────────────────────────────────

    def list_collections(self) -> CollectionListResponse:
        """List all collections (v1 admin endpoint).

        Mirrors Go Client.ListCollections().
        """
        data = self._request("GET", "/admin/collection/list")
        return CollectionListResponse.model_validate(data)

    def create_collection(
        self,
        name: str,
        fields: list[dict[str, Any] | FieldSchema] | None = None,
    ) -> dict[str, Any]:
        """Create a new collection (v2 endpoint).

        Args:
            name: Collection name (1-64 alphanumeric, underscores, hyphens).
            fields: Optional list of field schemas.

        Returns:
            Server response dict with status and message.
        """
        payload: dict[str, Any] = {"name": name}
        if fields is not None:
            normalized: list[dict[str, Any]] = []
            for f in fields:
                if isinstance(f, FieldSchema):
                    normalized.append(f.model_dump(exclude_none=True))
                else:
                    normalized.append(f)
            payload["fields"] = normalized

        return self._request("POST", "/v2/collections", json=payload)

    def get_collection(self, name: str) -> dict[str, Any]:
        """Get collection info (v2 endpoint)."""
        return self._request("GET", f"/v2/collections/{name}")

    def delete_collection(self, name: str) -> dict[str, Any]:
        """Delete a collection (v2 endpoint)."""
        return self._request("DELETE", f"/v2/collections/{name}")

    def collection_stats(self, name: str) -> CollectionStatsResponse:
        """Get collection statistics (v2 endpoint)."""
        data = self._request("GET", f"/v2/collections/{name}/stats")
        return CollectionStatsResponse.model_validate(data)

    # ── Admin ───────────────────────────────────────────────────────────

    def compact(self) -> CompactResponse:
        """Trigger index compaction.

        Mirrors Go Client.Compact().
        """
        data = self._request("POST", "/compact")
        return CompactResponse.model_validate(data)

    # ── Sparse Vectors ──────────────────────────────────────────────────

    def insert_sparse(
        self,
        doc: str,
        *,
        indices: list[int],
        values: list[float],
        id: str | None = None,
        dimension: int | None = None,
        meta: dict[str, str] | None = None,
        upsert: bool = False,
        collection: str | None = None,
    ) -> InsertResponse:
        """Insert a document with a sparse vector.

        Uses the /insert/sparse endpoint.
        """
        payload: dict[str, Any] = {
            "doc": doc,
            "indices": indices,
            "values": values,
        }
        if id is not None:
            payload["id"] = id
        if dimension is not None:
            payload["dimension"] = dimension
        if meta is not None:
            payload["meta"] = meta
        if upsert:
            payload["upsert"] = True
        if collection is not None:
            payload["collection"] = collection

        data = self._request("POST", "/insert/sparse", json=payload)
        return InsertResponse.model_validate(data)

    # ── Tenant Operations (v3) ──────────────────────────────────────────

    def tenant(self, tenant_id: str) -> TenantClient:
        """Get a tenant-scoped client for multi-tenant operations (v3 API)."""
        return TenantClient(self, tenant_id)


class TenantClient:
    """Tenant-scoped operations using the v3 API.

    Usage::

        tenant = client.tenant("org-123")
        tenant.create_collection("docs", fields=[...])
        tenant.insert("docs", vectors={"embedding": [0.1, 0.2, 0.3]})
        results = tenant.search("docs", queries={"embedding": [0.1, 0.2, 0.3]}, top_k=5)
    """

    def __init__(self, client: DeepDataClient, tenant_id: str) -> None:
        self._client = client
        self._tenant_id = tenant_id

    def _request(self, method: str, path: str, *, json: Any = None) -> Any:
        return self._client._request(method, f"/v3/tenants/{self._tenant_id}{path}", json=json)

    def create_collection(
        self,
        name: str,
        fields: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a collection within this tenant."""
        payload: dict[str, Any] = {"name": name}
        if fields is not None:
            payload["fields"] = fields
        return self._request("POST", "/collections", json=payload)

    def delete_collection(self, name: str) -> dict[str, Any]:
        """Delete a collection within this tenant."""
        return self._request("DELETE", f"/collections/{name}")

    def insert(
        self,
        collection: str,
        *,
        vectors: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert a document into a tenant collection."""
        payload: dict[str, Any] = {"vectors": vectors}
        if metadata is not None:
            payload["metadata"] = metadata
        return self._request("POST", f"/collections/{collection}/docs", json=payload)

    def search(
        self,
        collection: str,
        *,
        queries: dict[str, Any],
        top_k: int = 10,
        ef_search: int | None = None,
    ) -> dict[str, Any]:
        """Search within a tenant collection."""
        payload: dict[str, Any] = {"queries": queries, "top_k": top_k}
        if ef_search is not None:
            payload["ef_search"] = ef_search
        return self._request("POST", f"/collections/{collection}/search", json=payload)

    def info(self) -> dict[str, Any]:
        """Get tenant info."""
        return self._request("GET", "")
