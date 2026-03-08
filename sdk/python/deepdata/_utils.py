"""Internal helpers: retry logic, auth headers, response parsing.

Mirrors retry.go and the doJSON flow from the Go client.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .errors import (
    ConnectionError,
    TimeoutError,
    classify_error,
)


@dataclass
class RetryConfig:
    """Retry configuration matching the Go client's RetryConfig."""

    max_retries: int = 3
    initial_delay: float = 0.5  # seconds
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter_percent: float = 0.1


DEFAULT_RETRY = RetryConfig()

# Status codes that trigger retry — matches Go client
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def retry_delay(attempt: int, cfg: RetryConfig) -> float:
    """Compute backoff delay for the given attempt (0-indexed)."""
    delay = cfg.initial_delay * (cfg.multiplier ** attempt)
    delay = min(delay, cfg.max_delay)
    jitter = delay * cfg.jitter_percent * (2 * random.random() - 1)
    return max(0.0, delay + jitter)


def should_retry(status_code: int, attempt: int, cfg: RetryConfig | None) -> bool:
    """Check if a request should be retried."""
    if cfg is None or attempt >= cfg.max_retries:
        return False
    return status_code in _RETRYABLE_STATUS_CODES


def build_headers(
    token: str | None = None,
    tenant_id: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build default request headers."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    if extra:
        headers.update(extra)
    return headers


def handle_response(response: httpx.Response) -> Any:
    """Parse response, raising typed errors for non-2xx."""
    if 200 <= response.status_code < 300:
        if not response.content:
            return None
        return response.json()

    raise classify_error(response.status_code, response.text)


def handle_request_error(exc: Exception) -> None:
    """Convert httpx transport errors into SDK errors."""
    if isinstance(exc, httpx.TimeoutException):
        raise TimeoutError(str(exc)) from exc
    if isinstance(exc, httpx.ConnectError):
        raise ConnectionError(str(exc)) from exc
    if isinstance(exc, httpx.HTTPError):
        raise ConnectionError(str(exc)) from exc
    raise ConnectionError(str(exc)) from exc
