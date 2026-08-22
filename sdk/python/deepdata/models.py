"""Pydantic models for DeepData's canonical tenant-aware V3 API."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


# ── Canonical multi-tenant API (v3) ────────────────────────────────────────


class _TenantRequestModel(BaseModel):
    """Strict request model for the canonical tenant API."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _TenantResponseModel(BaseModel):
    """Typed response model which remains tolerant of additive server fields."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TenantIndexConfig(_TenantRequestModel):
    """Index configuration for a canonical vector field."""

    type: Literal["hnsw", "flat", "inverted"]
    params: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_canonical_params(self) -> TenantIndexConfig:
        """Exclude advanced/ignored index knobs from the supported SDK."""

        params = self.params or {}
        allowed = {
            "hnsw": {"m", "ml", "ef_search", "ef_construction", "prenormalize"},
            "flat": {"metric"},
            "inverted": {"k1", "b"},
        }[self.type]
        unknown = params.keys() - allowed
        if unknown:
            raise ValueError(
                "index parameters are outside the canonical release contract: "
                + ", ".join(sorted(unknown))
            )

        def finite_number(key: str) -> float | None:
            if key not in params:
                return None
            value = params[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"index parameter {key} must be a finite number")
            return float(value)

        if self.type == "hnsw":
            for key, minimum, maximum in (
                ("m", 2, 100),
                ("ef_search", 1, 1_000_000),
                ("ef_construction", 1, 1_000_000),
            ):
                value = finite_number(key)
                if value is not None and (
                    not value.is_integer() or value < minimum or value > maximum
                ):
                    raise ValueError(
                        f"index parameter {key} must be an integer in "
                        f"[{minimum}, {maximum}]"
                    )
            ml = finite_number("ml")
            if ml is not None and not 0 < ml <= 10:
                raise ValueError("index parameter ml must be in (0, 10]")
            if "prenormalize" in params and not isinstance(
                params["prenormalize"], bool
            ):
                raise ValueError("index parameter prenormalize must be a boolean")
        elif self.type == "flat" and "metric" in params:
            if params["metric"] not in {"cosine", "euclidean"}:
                raise ValueError("index parameter metric must be cosine or euclidean")
        elif self.type == "inverted":
            k1 = finite_number("k1")
            if k1 is not None and not 0 < k1 <= 100:
                raise ValueError("index parameter k1 must be in (0, 100]")
            b = finite_number("b")
            if b is not None and not 0 <= b <= 1:
                raise ValueError("index parameter b must be in [0, 1]")
        return self


class TenantVectorField(_TenantRequestModel):
    """Vector field accepted by ``POST /v3/.../collections``."""

    name: str
    type: Literal["dense", "sparse"]
    dim: int = Field(gt=0, le=65_536)
    index: TenantIndexConfig

    @model_validator(mode="after")
    def validate_canonical_index(self) -> TenantVectorField:
        """Enforce the frozen canonical release's vector/index matrix."""

        if self.type == "dense" and self.index.type not in {"hnsw", "flat"}:
            raise ValueError("dense fields require an hnsw or flat index")
        if self.type == "sparse" and self.index.type != "inverted":
            raise ValueError("sparse fields require an inverted index")
        return self


class TenantCollectionSchema(_TenantRequestModel):
    """Canonical collection creation payload."""

    name: str
    fields: list[TenantVectorField] = Field(min_length=1, max_length=8)
    metadata: dict[str, Any] | None = None
    description: str | None = None


class TenantCollectionInfo(_TenantResponseModel):
    """Information returned for a tenant collection.

    Current servers serialize the Go ``CollectionInfo`` fields with title-case
    names. The aliases also accept the intended snake-case wire spelling so the
    SDK remains compatible when the server normalizes those keys.
    """

    name: str = Field(validation_alias=AliasChoices("name", "Name"))
    fields: list[TenantVectorField] = Field(
        validation_alias=AliasChoices("fields", "Fields")
    )
    description: str = Field(
        default="", validation_alias=AliasChoices("description", "Description")
    )
    metadata: dict[str, Any] | None = Field(
        default=None, validation_alias=AliasChoices("metadata", "Metadata")
    )
    doc_count: int = Field(
        default=0, validation_alias=AliasChoices("doc_count", "DocCount"), ge=0
    )


class TenantCollectionMutationResponse(_TenantResponseModel):
    """Response from creating or deleting a tenant collection."""

    status: Literal["success"]
    tenant_id: str
    message: str


class TenantCollectionListResponse(_TenantResponseModel):
    """Response from listing a tenant's collections."""

    status: Literal["success"]
    tenant_id: str
    count: int = Field(ge=0)
    collections: list[TenantCollectionInfo]


class TenantGetCollectionResponse(_TenantResponseModel):
    """Response from fetching one tenant collection."""

    status: Literal["success"]
    tenant_id: str
    collection: TenantCollectionInfo


class TenantDocumentInput(_TenantRequestModel):
    """One document accepted by canonical insert endpoints."""

    id: int | None = Field(default=None, gt=0)
    vectors: dict[str, Any] = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class TenantBatchInsertRequest(_TenantRequestModel):
    """All-or-nothing canonical batch insert payload."""

    documents: list[TenantDocumentInput] = Field(min_length=1, max_length=10_000)


class TenantDeleteDocumentRequest(_TenantRequestModel):
    """Canonical document deletion payload."""

    doc_id: int = Field(gt=0)


class TenantDocument(_TenantResponseModel):
    """A canonical document returned by tenant search."""

    id: int = Field(gt=0)
    vectors: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class TenantGetDocumentRequest(_TenantRequestModel):
    """Canonical document read-by-ID path payload (empty today, reserved)."""

    pass


class TenantUpsertDocumentRequest(_TenantRequestModel):
    """One document upserted under a caller-supplied ID. Unlike insert, the ID
    is required and never auto-assigned by the server."""

    vectors: dict[str, Any] = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class TenantUpsertResponse(_TenantResponseModel):
    """Response from upserting one canonical document."""

    status: Literal["success"]
    tenant_id: str
    id: int = Field(gt=0)
    message: str


class TenantInsertResponse(_TenantResponseModel):
    """Response from inserting one canonical document."""

    status: Literal["success"]
    tenant_id: str
    id: int = Field(gt=0)
    message: str


class TenantBatchInsertResponse(_TenantResponseModel):
    """Response from an all-or-nothing canonical batch insert."""

    status: Literal["success"]
    tenant_id: str
    ids: list[int]
    inserted: int = Field(ge=0)


class TenantDeleteDocumentResponse(_TenantResponseModel):
    """Response from deleting one canonical document."""

    status: Literal["success"]
    tenant_id: str
    message: str


class TenantHybridParams(_TenantRequestModel):
    """Fusion settings for a multi-field tenant search."""

    strategy: Literal["rrf", "weighted", "linear"]
    weights: dict[str, float] | None = None
    rrf_constant: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_numeric_params(self) -> TenantHybridParams:
        """Reject values that the canonical server cannot fuse safely."""

        if self.rrf_constant is not None and not math.isfinite(self.rrf_constant):
            raise ValueError("rrf_constant must be finite")
        if self.weights is not None:
            if any(
                not math.isfinite(weight) or weight < 0
                for weight in self.weights.values()
            ):
                raise ValueError("hybrid weights must be finite and non-negative")
            if not any(weight > 0 for weight in self.weights.values()):
                raise ValueError("hybrid weights must contain a positive value")
        return self


class TenantFallbackParams(_TenantRequestModel):
    """Auto-fallback ladder: primary field first, secondary if it is weak.

    With ``threshold=None`` the ladder only falls back on zero hits; with a
    threshold it also falls back when the primary's best score is worse than
    the threshold in the field's score direction (best distance > threshold
    on dense fields, best score < threshold on sparse fields).
    """

    primary: str = Field(min_length=1)
    secondary: str = Field(min_length=1)
    threshold: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_fallback_contract(self) -> TenantFallbackParams:
        """Reject ladders the server cannot run."""

        if self.primary == self.secondary:
            raise ValueError("fallback primary and secondary must differ")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("fallback threshold must be finite")
        return self


class TenantSearchRequest(_TenantRequestModel):
    """Canonical tenant search payload.

    ``score_floor`` is a confidence filter on the returned raw scores: on
    dense (distance) fields it is a maximum acceptable distance (keep
    ``score <= score_floor``), on sparse (BM25) and hybrid scores a minimum
    acceptable score (keep ``score >= score_floor``). ``usage_boost`` blends
    non-durable per-tenant usage (frecency) into ranking without altering
    the reported raw scores.
    """

    queries: dict[str, Any] = Field(min_length=1, max_length=2)
    top_k: int = Field(default=10, gt=0, le=1000)
    ef_search: int | None = Field(default=None, ge=0)
    include_vectors: bool | None = None
    filters: dict[str, Any] | None = None
    hybrid_params: TenantHybridParams | None = None
    score_floor: float | None = Field(default=None, ge=0)
    fallback: TenantFallbackParams | None = None
    usage_boost: float | None = Field(default=None, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_hybrid_contract(self) -> TenantSearchRequest:
        """Keep SDK admission aligned with the server's two-field contract."""

        if (
            len(self.queries) > 1
            and self.hybrid_params is None
            and self.fallback is None
        ):
            raise ValueError(
                "multiple query fields require hybrid_params or fallback"
            )
        if self.hybrid_params is not None and self.hybrid_params.weights is not None:
            unknown = self.hybrid_params.weights.keys() - self.queries.keys()
            if unknown:
                raise ValueError(
                    "hybrid weights reference unknown query fields: "
                    + ", ".join(sorted(unknown))
                )
        if self.fallback is not None:
            if self.hybrid_params is not None:
                raise ValueError(
                    "fallback and hybrid_params are mutually exclusive"
                )
            if len(self.queries) != 2:
                raise ValueError("fallback requires exactly two query fields")
            missing = {self.fallback.primary, self.fallback.secondary} - self.queries.keys()
            if missing:
                raise ValueError(
                    "fallback fields must be query fields: "
                    + ", ".join(sorted(missing))
                )
        for name, value in (("score_floor", self.score_floor), ("usage_boost", self.usage_boost)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class TenantSearchResponse(_TenantResponseModel):
    """Typed canonical tenant search result.

    ``best_score`` is the best raw score among ``documents`` (0 when empty)
    and calibrates ``score_floor``. ``weak_match`` is true when
    ``score_floor`` is set and no document survived it — treat it as "no
    confident answer" rather than consuming the (empty) results.
    ``fell_back_to`` names the secondary field when the fallback ladder
    fired.
    """

    status: Literal["success"]
    tenant_id: str
    documents: list[TenantDocument]
    scores: list[float]
    candidates_examined: int = Field(ge=0)
    best_score: float = 0.0
    weak_match: bool = False
    fell_back_to: str = ""


class TenantCollectionStats(_TenantResponseModel):
    """Per-collection counters embedded in tenant info."""

    name: str = Field(validation_alias=AliasChoices("name", "Name"))
    doc_count: int = Field(
        validation_alias=AliasChoices("doc_count", "DocCount"), ge=0
    )
    field_count: int = Field(
        validation_alias=AliasChoices("field_count", "FieldCount"), ge=0
    )


class TenantInfoResponse(_TenantResponseModel):
    """Tenant-level collection and document counters."""

    status: Literal["success"]
    tenant_id: str
    collection_count: int = Field(ge=0)
    total_documents: int = Field(ge=0)
    collections: dict[str, TenantCollectionStats] | None = None
