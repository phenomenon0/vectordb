"""DeepData Python SDK — client for the DeepData vector database.

Usage::

    from deepdata import DeepDataClient

    client = DeepDataClient("http://localhost:8080")
    result = client.insert("Hello world", meta={"source": "test"})
    results = client.search("Hello", top_k=5)

Async::

    from deepdata import AsyncDeepDataClient

    async with AsyncDeepDataClient("http://localhost:8080") as client:
        results = await client.search("Hello", top_k=5)
"""

from .client import DeepDataClient, TenantClient
from .async_client import AsyncDeepDataClient, AsyncTenantClient
from .models import (
    BatchDoc,
    BatchInsertRequest,
    BatchInsertResponse,
    CollectionInfo,
    CollectionListResponse,
    CollectionSchema,
    CollectionStatsResponse,
    CompactResponse,
    DeleteRequest,
    DeleteResponse,
    FieldSchema,
    HealthResponse,
    InsertRequest,
    InsertResponse,
    RangeFilter,
    ScrollRequest,
    ScrollResponse,
    SearchRequest,
    SearchResult,
    SparseInsertRequest,
)
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
from ._utils import RetryConfig

__version__ = "0.1.0"

__all__ = [
    # Clients
    "DeepDataClient",
    "AsyncDeepDataClient",
    "TenantClient",
    "AsyncTenantClient",
    # Config
    "RetryConfig",
    # Models
    "BatchDoc",
    "BatchInsertRequest",
    "BatchInsertResponse",
    "CollectionInfo",
    "CollectionListResponse",
    "CollectionSchema",
    "CollectionStatsResponse",
    "CompactResponse",
    "DeleteRequest",
    "DeleteResponse",
    "FieldSchema",
    "HealthResponse",
    "InsertRequest",
    "InsertResponse",
    "RangeFilter",
    "ScrollRequest",
    "ScrollResponse",
    "SearchRequest",
    "SearchResult",
    "SparseInsertRequest",
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
