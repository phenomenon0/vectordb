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


class TestEnvelope:
    """The structured envelope (API.md, section Errors) drives classification."""

    ENVELOPE = (
        '{"code":"not_found","message":"collection not found: docs","hint":"list the collections",'
        '"request_id":"req-1","retryable":false,"docs":"internal/collection/API.md#errors"}'
    )

    def test_envelope_fields_are_exposed(self) -> None:
        err = classify_error(404, self.ENVELOPE)
        assert isinstance(err, NotFoundError)
        assert err.code == "not_found"
        assert err.message == "collection not found: docs"
        assert err.hint == "list the collections"
        assert err.request_id == "req-1"
        assert err.docs == "internal/collection/API.md#errors"
        assert "not_found" in str(err) and "Hint: list the collections" in str(err)

    def test_quota_409_is_not_retryable(self) -> None:
        # A tenant or collection limit is permanent for the process lifetime;
        # retrying it would loop forever against a wall.
        err = classify_error(409, '{"code":"quota_exceeded","message":"tenant limit","retryable":false}')
        assert type(err) is APIError
        assert err.code == "quota_exceeded"
        assert not err.retryable

    def test_server_retryable_verdict_overrides_status_default(self) -> None:
        err = classify_error(503, '{"code":"internal","message":"do not retry","retryable":false}')
        assert isinstance(err, ServerError)
        assert not err.retryable

    def test_retry_after_from_envelope(self) -> None:
        err = classify_error(
            429, '{"code":"rate_limited","message":"slow down","retryable":true,"retry_after_ms":1000}'
        )
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 1.0
        assert err.retryable

    def test_retry_after_from_header_for_plain_text(self) -> None:
        err = classify_error(429, "rate limited", {"Retry-After": "2"})
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 2.0

    def test_plain_text_body_has_empty_envelope(self) -> None:
        err = classify_error(404, "collection not found")
        assert err.message == "collection not found"
        assert err.code == "" and err.hint == "" and err.retry_after is None


class TestShouldRetry:
    def test_honours_server_verdict(self) -> None:
        from deepdata._utils import DEFAULT_RETRY, should_retry

        retryable = classify_error(429, '{"code":"rate_limited","message":"x","retryable":true}')
        permanent = classify_error(409, '{"code":"quota_exceeded","message":"x","retryable":false}')
        assert should_retry(retryable, 0, DEFAULT_RETRY)
        assert not should_retry(permanent, 0, DEFAULT_RETRY)
        assert not should_retry(retryable, DEFAULT_RETRY.max_retries, DEFAULT_RETRY)
        assert not should_retry(retryable, 0, None)
