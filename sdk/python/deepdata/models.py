"""Pydantic models for DeepData request/response types.

Mirrors the Go client structs (client/client.go) and server request schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Insert ──────────────────────────────────────────────────────────────────


class InsertRequest(BaseModel):
    doc: str
    id: str | None = None
    meta: dict[str, str] | None = None
    upsert: bool = False
    collection: str | None = None


class InsertResponse(BaseModel):
    id: str


# ── Batch Insert ────────────────────────────────────────────────────────────


class BatchDoc(BaseModel):
    doc: str
    id: str | None = None
    meta: dict[str, str] | None = None
    collection: str | None = None


class BatchInsertRequest(BaseModel):
    docs: list[BatchDoc]
    upsert: bool = False


class BatchInsertResponse(BaseModel):
    ids: list[str]
    errors: list[str] | None = None


# ── Search / Query ──────────────────────────────────────────────────────────


class RangeFilter(BaseModel):
    key: str
    min: float | None = None
    max: float | None = None
    time_min: str | None = None
    time_max: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: Literal["ann", "scan", "lex"] | None = None
    collection: str | None = None
    meta: dict[str, str] | None = None
    meta_any: list[dict[str, str]] | None = None
    meta_not: dict[str, str] | None = None
    meta_ranges: list[RangeFilter] | None = None
    include_meta: bool = False
    hybrid_alpha: float | None = None
    score_mode: str | None = None
    ef_search: int | None = None
    offset: int | None = None
    limit: int | None = None
    page_token: str | None = None
    page_size: int | None = None


class SearchResult(BaseModel):
    ids: list[str] = Field(default_factory=list)
    docs: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    stats: str | None = None
    meta: list[dict[str, str]] | None = None
    next: str | None = None


# ── Delete ──────────────────────────────────────────────────────────────────


class DeleteRequest(BaseModel):
    id: str


class DeleteResponse(BaseModel):
    deleted: str


# ── Scroll ──────────────────────────────────────────────────────────────────


class ScrollRequest(BaseModel):
    collection: str | None = None
    limit: int | None = None
    offset: int | None = None


class ScrollResponse(BaseModel):
    ids: list[str] = Field(default_factory=list)
    docs: list[str] = Field(default_factory=list)
    meta: list[dict[str, str]] | None = None
    total: int = 0
    next_offset: int = 0


# ── Health ──────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    ok: bool = True
    total: int = 0
    active: int = 0
    deleted: int = 0
    hnsw_ids: int = 0
    checksum: str = ""
    wal_bytes: int = 0
    index_bytes: int | None = None
    snapshot_age_ms: int | None = None
    wal_age_ms: int | None = None


# ── Collections (v1 admin) ──────────────────────────────────────────────────


class CollectionListResponse(BaseModel):
    status: str = ""
    count: int = 0
    collections: list[dict[str, Any]] = Field(default_factory=list)


# ── Collections (v2) ────────────────────────────────────────────────────────


class FieldSchema(BaseModel):
    name: str
    type: str  # "dense", "sparse", etc.
    dim: int | None = None
    index_type: str | None = None


class CollectionSchema(BaseModel):
    name: str
    fields: list[FieldSchema] | None = None


class CollectionInfo(BaseModel):
    name: str = ""
    fields: list[dict[str, Any]] | None = None
    doc_count: int | None = None


class CollectionStatsResponse(BaseModel):
    status: str = ""
    name: str = ""
    doc_count: int | None = None
    manager_stats: dict[str, Any] | None = None


# ── Compact ─────────────────────────────────────────────────────────────────


class CompactResponse(BaseModel):
    ok: bool = False


# ── Sparse Insert ───────────────────────────────────────────────────────────


class SparseInsertRequest(BaseModel):
    doc: str
    id: str | None = None
    indices: list[int]
    values: list[float]
    dimension: int | None = None
    meta: dict[str, str] | None = None
    upsert: bool = False
    collection: str | None = None
