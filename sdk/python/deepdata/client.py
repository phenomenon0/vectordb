"""Synchronous client for DeepData's canonical tenant-aware V3 API."""

from __future__ import annotations

import time
from typing import Any

import httpx

from .errors import APIError, DeepDataError
from .models import (
    TenantBatchInsertRequest,
    TenantBatchInsertResponse,
    TenantCollectionListResponse,
    TenantCollectionMutationResponse,
    TenantCollectionSchema,
    TenantDeleteDocumentRequest,
    TenantDeleteDocumentResponse,
    TenantDocumentInput,
    TenantGetCollectionResponse,
    TenantHybridParams,
    TenantInfoResponse,
    TenantInsertResponse,
    TenantSearchRequest,
    TenantSearchResponse,
    TenantVectorField,
)
from ._tenant import (
    collection_segment,
    request_payload,
    response_model,
    tenant_base_path,
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


def _require_response_object(data: object) -> dict[str, Any]:
    """Validate that an untyped JSON response is an object with string keys."""
    if not isinstance(data, dict):
        raise DeepDataError("expected a JSON object response")

    result: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise DeepDataError("expected JSON object response keys to be strings")
        result[key] = value
    return result


class DeepDataClient:
    """Synchronous HTTP client for DeepData vector database.

    Usage::

        client = DeepDataClient("http://localhost:8080", api_token="sk-...")
        tenant = client.tenant("org-123")
        result = tenant.insert("docs", vectors={"embedding": [0.1, 0.2]})
        client.close()

    Or as a context manager::

        with DeepDataClient(
            "http://localhost:8080", api_token="sk-..."
        ) as client:
            results = client.tenant("org-123").search(
                "docs", queries={"embedding": [0.1, 0.2]}, top_k=5
            )

    All database operations are tenant scoped. Use :meth:`tenant` to obtain a
    typed V3 client for a validated tenant identifier.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        *,
        api_token: str | None = None,
        timeout: float = 15.0,
        retry: RetryConfig | None = DEFAULT_RETRY,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._retry = retry
        self._headers = build_headers(
            token=api_token,
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
        retryable: bool | None = None,
    ) -> Any:
        """Execute an HTTP request with retry logic for safe operations.

        GET/HEAD/OPTIONS are retryable by default. Callers may explicitly mark a
        read-only POST (canonical search) retryable, but mutations remain
        single-attempt until the protocol provides idempotency keys.
        """
        last_exc: Exception | None = None
        if retryable is None:
            retryable = method.upper() in {"GET", "HEAD", "OPTIONS"}
        retry_config = self._retry if retryable else None
        max_attempts = 1 + (retry_config.max_retries if retry_config else 0)

        for attempt in range(max_attempts):
            if attempt > 0 and retry_config:
                delay = retry_delay(attempt - 1, retry_config)
                time.sleep(delay)

            try:
                response = self._http.request(method, path, json=json)
                return handle_response(response)

            except DeepDataError as exc:
                last_exc = exc
                if isinstance(exc, APIError) and should_retry(
                    exc.status_code, attempt, retry_config
                ):
                    continue
                raise

            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < max_attempts - 1 and retry_config:
                    continue
                handle_request_error(exc)

        if last_exc is not None:
            if isinstance(last_exc, DeepDataError):
                raise last_exc
            handle_request_error(last_exc)

    def tenant(self, tenant_id: str) -> TenantClient:
        """Get a typed client for one tenant on the canonical V3 API."""
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
        self._base_path = tenant_base_path(tenant_id)

    def _request(self, method: str, path: str, *, json: Any = None) -> dict[str, Any]:
        data = self._client._request(
            method,
            f"{self._base_path}{path}",
            json=json,
            retryable=method.upper() == "GET",
        )
        return _require_response_object(data)

    def create_collection(
        self,
        name: str,
        fields: list[dict[str, Any] | TenantVectorField],
        *,
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> TenantCollectionMutationResponse:
        """Create a canonical collection within this tenant."""
        # Validate path usability at creation time as this name will become a
        # URL segment for every subsequent collection operation.
        collection_segment(name)
        normalized_fields = [
            field
            if isinstance(field, TenantVectorField)
            else TenantVectorField.model_validate(field)
            for field in fields
        ]
        schema = TenantCollectionSchema(
            name=name,
            fields=normalized_fields,
            metadata=metadata,
            description=description,
        )
        data = self._request("POST", "/collections", json=request_payload(schema))
        return response_model(TenantCollectionMutationResponse, data)

    def list_collections(self) -> TenantCollectionListResponse:
        """List canonical collections belonging to this tenant."""
        data = self._request("GET", "/collections")
        return response_model(TenantCollectionListResponse, data)

    def get_collection(self, name: str) -> TenantGetCollectionResponse:
        """Get one canonical tenant collection."""
        segment = collection_segment(name)
        data = self._request("GET", f"/collections/{segment}")
        return response_model(TenantGetCollectionResponse, data)

    def delete_collection(self, name: str) -> TenantCollectionMutationResponse:
        """Delete a collection within this tenant."""
        segment = collection_segment(name)
        data = self._request("DELETE", f"/collections/{segment}")
        return response_model(TenantCollectionMutationResponse, data)

    def insert(
        self,
        collection: str,
        *,
        id: int | None = None,
        vectors: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> TenantInsertResponse:
        """Insert a document into a tenant collection."""
        segment = collection_segment(collection)
        document = TenantDocumentInput(id=id, vectors=vectors, metadata=metadata)
        data = self._request(
            "POST",
            f"/collections/{segment}/docs",
            json=request_payload(document),
        )
        return response_model(TenantInsertResponse, data)

    def batch_insert(
        self,
        collection: str,
        documents: list[dict[str, Any] | TenantDocumentInput],
    ) -> TenantBatchInsertResponse:
        """Insert a batch atomically; the server never reports partial success."""
        segment = collection_segment(collection)
        normalized = [
            document
            if isinstance(document, TenantDocumentInput)
            else TenantDocumentInput.model_validate(document)
            for document in documents
        ]
        request = TenantBatchInsertRequest(documents=normalized)
        data = self._request(
            "POST",
            f"/collections/{segment}/docs/batch",
            json=request_payload(request),
        )
        return response_model(TenantBatchInsertResponse, data)

    def delete_document(
        self,
        collection: str,
        doc_id: int,
    ) -> TenantDeleteDocumentResponse:
        """Delete one document from a canonical tenant collection."""
        segment = collection_segment(collection)
        request = TenantDeleteDocumentRequest(doc_id=doc_id)
        data = self._request(
            "DELETE",
            f"/collections/{segment}/docs",
            json=request_payload(request),
        )
        return response_model(TenantDeleteDocumentResponse, data)

    def search(
        self,
        collection: str,
        *,
        queries: dict[str, Any],
        top_k: int = 10,
        ef_search: int | None = None,
        filters: dict[str, Any] | None = None,
        hybrid_params: dict[str, Any] | TenantHybridParams | None = None,
        include_vectors: bool | None = None,
    ) -> TenantSearchResponse:
        """Search within a tenant collection."""
        segment = collection_segment(collection)
        normalized_hybrid = (
            hybrid_params
            if isinstance(hybrid_params, TenantHybridParams) or hybrid_params is None
            else TenantHybridParams.model_validate(hybrid_params)
        )
        request = TenantSearchRequest(
            queries=queries,
            top_k=top_k,
            ef_search=ef_search,
            filters=filters,
            hybrid_params=normalized_hybrid,
            include_vectors=include_vectors,
        )
        data = self._client._request(
            "POST",
            f"{self._base_path}/collections/{segment}/search",
            json=request_payload(request),
            retryable=True,
        )
        data = _require_response_object(data)
        return response_model(TenantSearchResponse, data)

    def info(self) -> TenantInfoResponse:
        """Get tenant info."""
        data = self._request("GET", "")
        return response_model(TenantInfoResponse, data)
