"""Tests for the error hierarchy and classification."""

from __future__ import annotations

from deepdata.errors import (
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
    classify_error,
)


class TestErrorHierarchy:
    def test_all_inherit_from_deepdata_error(self) -> None:
        assert issubclass(ConnectionError, DeepDataError)
        assert issubclass(TimeoutError, DeepDataError)
        assert issubclass(APIError, DeepDataError)
        assert issubclass(AuthenticationError, DeepDataError)
        assert issubclass(PermissionError, DeepDataError)
        assert issubclass(NotFoundError, DeepDataError)
        assert issubclass(ValidationError, DeepDataError)
        assert issubclass(RateLimitError, DeepDataError)
        assert issubclass(ServerError, DeepDataError)

    def test_http_errors_inherit_from_api_error(self) -> None:
        assert issubclass(AuthenticationError, APIError)
        assert issubclass(PermissionError, APIError)
        assert issubclass(NotFoundError, APIError)
        assert issubclass(ValidationError, APIError)
        assert issubclass(RateLimitError, APIError)
        assert issubclass(ServerError, APIError)


class TestClassifyError:
    def test_401(self) -> None:
        err = classify_error(401, "unauthorized")
        assert isinstance(err, AuthenticationError)
        assert err.status_code == 401
        assert not err.retryable

    def test_403(self) -> None:
        err = classify_error(403, "forbidden")
        assert isinstance(err, PermissionError)
        assert err.status_code == 403

    def test_404(self) -> None:
        err = classify_error(404, "not found")
        assert isinstance(err, NotFoundError)
        assert err.status_code == 404

    def test_422(self) -> None:
        err = classify_error(422, "validation")
        assert isinstance(err, ValidationError)
        assert err.status_code == 422

    def test_400(self) -> None:
        err = classify_error(400, "validation")
        assert isinstance(err, ValidationError)
        assert err.status_code == 400

    def test_429(self) -> None:
        err = classify_error(429, "rate limited")
        assert isinstance(err, RateLimitError)
        assert err.retryable

    def test_500(self) -> None:
        err = classify_error(500, "internal error")
        assert isinstance(err, ServerError)
        assert err.retryable

    def test_502(self) -> None:
        err = classify_error(502, "bad gateway")
        assert isinstance(err, ServerError)
        assert err.retryable

    def test_unknown_4xx(self) -> None:
        err = classify_error(418, "teapot")
        assert isinstance(err, APIError)
        assert not isinstance(err, ServerError)
        assert not err.retryable

    def test_error_message(self) -> None:
        err = classify_error(500, "something broke")
        assert "500" in str(err)
        assert "something broke" in str(err)


class TestRateLimitError:
    def test_retry_after(self) -> None:
        err = RateLimitError("rate limited", retry_after=5.0)
        assert err.retry_after == 5.0
        assert err.retryable

    def test_no_retry_after(self) -> None:
        err = RateLimitError()
        assert err.retry_after is None
