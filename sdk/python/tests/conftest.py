"""Test fixtures for DeepData SDK tests."""

from __future__ import annotations

import pytest
import httpx
import respx

from deepdata import DeepDataClient, AsyncDeepDataClient
from deepdata._utils import RetryConfig

BASE_URL = "http://testserver:8080"


@pytest.fixture
def base_url() -> str:
    return BASE_URL


@pytest.fixture
def client(base_url: str) -> DeepDataClient:
    """Sync client with retries disabled for predictable test behavior."""
    return DeepDataClient(base_url, retry=None)


@pytest.fixture
def client_with_retry(base_url: str) -> DeepDataClient:
    """Sync client with fast retries for retry tests."""
    return DeepDataClient(
        base_url,
        retry=RetryConfig(max_retries=2, initial_delay=0.01, max_delay=0.05),
    )


@pytest.fixture
def async_client(base_url: str) -> AsyncDeepDataClient:
    """Async client with retries disabled."""
    return AsyncDeepDataClient(base_url, retry=None)


@pytest.fixture
def client_with_auth(base_url: str) -> DeepDataClient:
    """Sync client with auth token and tenant."""
    return DeepDataClient(
        base_url,
        api_token="sk-test-token",
        tenant_id="tenant-123",
        retry=None,
    )
