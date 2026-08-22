"""Typed client for DeepData's canonical tenant-aware V3 API.

Usage::

    from deepdata import DeepDataClient

    with DeepDataClient(
        "http://localhost:8080", api_token="sk-..."
    ) as client:
        tenant = client.tenant("org-123")
        result = tenant.insert(
            "docs", vectors={"embedding": [0.1, 0.2]}
        )
        results = tenant.search(
            "docs", queries={"embedding": [0.1, 0.2]}, top_k=5
        )
"""

from .async_client import AsyncDeepDataClient, AsyncTenantClient
from .client import DeepDataClient, TenantClient
from .errors import (
    APIError,
    AuthenticationError,
    ConnectionError,
    DeepDataError,
    NotFoundError,
    PermissionError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from .models import (
    TenantBatchInsertRequest,
    TenantBatchInsertResponse,
    TenantCollectionInfo,
    TenantCollectionListResponse,
    TenantCollectionMutationResponse,
    TenantCollectionSchema,
    TenantCollectionStats,
    TenantDeleteDocumentRequest,
    TenantDeleteDocumentResponse,
    TenantDocument,
    TenantDocumentInput,
    TenantFallbackParams,
    TenantGetCollectionResponse,
    TenantHybridParams,
    TenantIndexConfig,
    TenantInfoResponse,
    TenantInsertResponse,
    TenantSearchRequest,
    TenantSearchResponse,
    TenantUpsertDocumentRequest,
    TenantUpsertResponse,
    TenantVectorField,
)
from ._utils import RetryConfig

__version__ = "0.2.0rc1"

__all__ = [
    # Clients and configuration
    "DeepDataClient",
    "AsyncDeepDataClient",
    "TenantClient",
    "AsyncTenantClient",
    "RetryConfig",
    # Canonical V3 models
    "TenantBatchInsertRequest",
    "TenantBatchInsertResponse",
    "TenantCollectionInfo",
    "TenantCollectionListResponse",
    "TenantCollectionMutationResponse",
    "TenantCollectionSchema",
    "TenantCollectionStats",
    "TenantDeleteDocumentRequest",
    "TenantDeleteDocumentResponse",
    "TenantDocument",
    "TenantDocumentInput",
    "TenantFallbackParams",
    "TenantGetCollectionResponse",
    "TenantHybridParams",
    "TenantIndexConfig",
    "TenantInfoResponse",
    "TenantInsertResponse",
    "TenantSearchRequest",
    "TenantSearchResponse",
    "TenantUpsertDocumentRequest",
    "TenantUpsertResponse",
    "TenantVectorField",
    # Errors
    "APIError",
    "AuthenticationError",
    "ConnectionError",
    "DeepDataError",
    "NotFoundError",
    "PermissionError",
    "RateLimitError",
    "ServerError",
    "TimeoutError",
    "ValidationError",
]
