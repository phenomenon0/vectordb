"""Shared canonical tenant API validation and serialization helpers."""

from __future__ import annotations

import re
from typing import TypeVar
from urllib.parse import quote

from pydantic import BaseModel, ValidationError as PydanticValidationError

from .errors import DeepDataError


_CANONICAL_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}")
_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


def _validated_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _CANONICAL_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be 1-64 ASCII alphanumeric, hyphen, or underscore characters"
        )
    return value


def tenant_base_path(tenant_id: str) -> str:
    """Return a validated, encoded tenant path prefix."""

    validated = _validated_identifier(tenant_id, label="tenant_id")
    return f"/v3/tenants/{quote(validated, safe='')}"


def collection_segment(collection: str) -> str:
    """Return a validated, encoded collection path segment.

    Canonical identifiers intentionally exclude path separators, dot segments,
    percent escapes, and control characters. Quoting remains explicit so a
    future identifier expansion cannot silently introduce an unsafe URL.
    """

    validated = _validated_identifier(collection, label="collection")
    return quote(validated, safe="")


def request_payload(model: BaseModel) -> dict[str, object]:
    """Serialize a validated canonical request without absent optional fields."""

    return model.model_dump(mode="json", exclude_none=True)


def response_model(model_type: type[_ResponseT], data: object) -> _ResponseT:
    """Validate a canonical response and expose malformed payloads as SDK errors."""

    try:
        return model_type.model_validate(data)
    except PydanticValidationError as exc:
        raise DeepDataError(
            f"invalid canonical API response for {model_type.__name__}: {exc}"
        ) from exc
