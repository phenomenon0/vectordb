"""Exception hierarchy for the DeepData Python SDK.

Mirrors the Go client's error classification (client/errors.go).
"""

from __future__ import annotations


class DeepDataError(Exception):
    """Base exception for all DeepData SDK errors."""

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(message)


class ConnectionError(DeepDataError):
    """Cannot reach the DeepData server."""


class TimeoutError(DeepDataError):
    """Request timed out."""


class APIError(DeepDataError):
    """Non-2xx HTTP response from the server."""

    def __init__(
        self,
        status_code: int,
        message: str = "",
        *,
        retryable: bool = False,
    ) -> None:
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(message or f"HTTP {status_code}")

    def __str__(self) -> str:
        return f"deepdata api {self.status_code}: {self.message}"


class AuthenticationError(APIError):
    """401 Unauthorized."""

    def __init__(self, message: str = "unauthorized") -> None:
        super().__init__(401, message, retryable=False)


class PermissionError(APIError):
    """403 Forbidden."""

    def __init__(self, message: str = "forbidden") -> None:
        super().__init__(403, message, retryable=False)


class NotFoundError(APIError):
    """404 Not Found (collection or document missing)."""

    def __init__(self, message: str = "not found") -> None:
        super().__init__(404, message, retryable=False)


class ValidationError(APIError):
    """422 Unprocessable Entity."""

    def __init__(self, message: str = "validation error") -> None:
        super().__init__(422, message, retryable=False)


class RateLimitError(APIError):
    """429 Too Many Requests."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None) -> None:
        super().__init__(429, message, retryable=True)
        self.retry_after = retry_after


class ServerError(APIError):
    """5xx server-side error."""

    def __init__(self, status_code: int = 500, message: str = "server error") -> None:
        super().__init__(status_code, message, retryable=True)


# Retryable status codes — matches Go client/errors.go
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def classify_error(status_code: int, body: str) -> APIError:
    """Convert an HTTP status code + body into the appropriate exception.

    Mirrors classifyHTTPError in the Go client.
    """
    retryable = status_code in _RETRYABLE_STATUS_CODES

    if status_code == 401:
        return AuthenticationError(body)
    if status_code == 403:
        return PermissionError(body)
    if status_code == 404:
        return NotFoundError(body)
    if status_code == 422:
        return ValidationError(body)
    if status_code == 429:
        return RateLimitError(body)
    if status_code >= 500:
        return ServerError(status_code, body)

    return APIError(status_code, body, retryable=retryable)
