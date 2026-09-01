"""Exception hierarchy for the DeepData Python SDK.

Servers that speak the structured error envelope
(internal/collection/API.md, section Errors) populate ``code``, ``hint``,
``field``, ``request_id``, ``docs`` and ``retryable`` on every
:class:`APIError`; older plain-text servers leave the envelope fields empty.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


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
        code: str = "",
        hint: str = "",
        field: str = "",
        request_id: str = "",
        retry_after: float | None = None,
        docs: str = "",
    ) -> None:
        self.status_code = status_code
        self.retryable = retryable
        self.code = code
        self.hint = hint
        self.field = field
        self.request_id = request_id
        self.retry_after = retry_after
        self.docs = docs
        super().__init__(message or f"HTTP {status_code}")

    def __str__(self) -> str:
        head = f"deepdata api {self.status_code}"
        if self.code:
            head += f" {self.code}"
        text = f"{head}: {self.message}"
        if self.hint:
            text += f" Hint: {self.hint}"
        return text


class AuthenticationError(APIError):
    """401 Unauthorized."""

    def __init__(self, message: str = "unauthorized", **envelope: Any) -> None:
        super().__init__(401, message, retryable=False, **envelope)


class PermissionError(APIError):
    """403 Forbidden."""

    def __init__(self, message: str = "forbidden", **envelope: Any) -> None:
        super().__init__(403, message, retryable=False, **envelope)


class NotFoundError(APIError):
    """404 Not Found (collection or document missing)."""

    def __init__(self, message: str = "not found", **envelope: Any) -> None:
        super().__init__(404, message, retryable=False, **envelope)


class ValidationError(APIError):
    """400/422 validation failure."""

    def __init__(
        self, message: str = "validation error", *, status_code: int = 422, **envelope: Any
    ) -> None:
        super().__init__(status_code, message, retryable=False, **envelope)


class RateLimitError(APIError):
    """429 Too Many Requests."""

    def __init__(
        self, message: str = "rate limited", retry_after: float | None = None, **envelope: Any
    ) -> None:
        super().__init__(429, message, retryable=True, retry_after=retry_after, **envelope)


class ServerError(APIError):
    """5xx server-side error."""

    def __init__(self, status_code: int = 500, message: str = "server error", **envelope: Any) -> None:
        super().__init__(status_code, message, retryable=True, **envelope)


# Retryable status codes — the transport-level fallback when the server sends
# no structured envelope.
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _parse_envelope(body: str) -> dict[str, Any]:
    """Return the structured error envelope, or {} for a plain-text body."""
    try:
        data = json.loads(body)
    except ValueError:
        return {}
    if isinstance(data, dict) and isinstance(data.get("code"), str):
        return data
    return {}


def classify_error(
    status_code: int, body: str, headers: Mapping[str, str] | None = None
) -> APIError:
    """Convert an HTTP status code + body into the appropriate exception.

    Mirrors classifyHTTPError in the Go client. A structured envelope supplies
    the message, the envelope fields and the server's own retryable verdict;
    a plain-text body is the message and the status code decides retryability.
    """
    envelope = _parse_envelope(body)
    message = str(envelope.get("message", "")) or body
    retry_after: float | None = None
    if envelope.get("retry_after_ms"):
        retry_after = float(envelope["retry_after_ms"]) / 1000
    elif headers is not None and headers.get("Retry-After"):
        try:
            retry_after = float(headers["Retry-After"])
        except ValueError:
            retry_after = None
    fields: dict[str, Any] = {
        "code": str(envelope.get("code", "")),
        "hint": str(envelope.get("hint", "")),
        "field": str(envelope.get("field", "")),
        "request_id": str(envelope.get("request_id", "")),
        "docs": str(envelope.get("docs", "")),
        "retry_after": retry_after,
    }

    err: APIError
    if status_code == 401:
        err = AuthenticationError(message, **fields)
    elif status_code == 403:
        err = PermissionError(message, **fields)
    elif status_code == 404:
        err = NotFoundError(message, **fields)
    elif status_code in (400, 422):
        err = ValidationError(message, status_code=status_code, **fields)
    elif status_code == 429:
        err = RateLimitError(message, **fields)
    elif status_code >= 500:
        err = ServerError(status_code, message, **fields)
    else:
        err = APIError(
            status_code, message, retryable=status_code in _RETRYABLE_STATUS_CODES, **fields
        )
    if isinstance(envelope.get("retryable"), bool):
        err.retryable = envelope["retryable"]
    return err
