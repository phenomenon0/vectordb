from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(slots=True)
class SearchResult:
    ids: list[int]
    scores: list[float] | None = None
    raw: Any = None

class VDBClient(Protocol):
    name: str

    def create_collection(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        metadata_schema: dict | None = None,
    ) -> None:
        ...

    def delete_collection(self, name: str) -> None:
        ...

    def collection_exists(self, name: str) -> bool:
        ...

    def count(self, name: str) -> int:
        ...

    def insert(self, name: str, ids, vectors, payloads=None) -> None:
        ...

    def upsert(self, name: str, ids, vectors, payloads=None) -> None:
        ...

    def delete_ids(self, name: str, ids) -> None:
        ...

    def get_by_ids(self, name: str, ids) -> list[dict]:
        ...

    def search(
        self,
        name: str,
        vector,
        top_k: int,
        filters: dict | None = None,
        ef_search: int | None = None,
    ) -> SearchResult:
        ...

    def flush(self, name: str | None = None) -> None:
        ...

    def close(self) -> None:
        ...
