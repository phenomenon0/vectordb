"""Abstract base class for VDB benchmark adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VDBAdapter(ABC):
    """Abstract adapter for vector database benchmarking.

    Each adapter wraps a single VDB's Python client and exposes a
    uniform interface for the benchmark runner.
    """

    name: str = "base"

    @abstractmethod
    def create_collection(
        self,
        name: str,
        dim: int,
        hnsw_m: int = 16,
        ef_construction: int = 200,
    ) -> None:
        """Create a collection/index with HNSW parameters."""

    @abstractmethod
    def insert_batch(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        """Insert a batch of documents with pre-computed vectors."""

    @abstractmethod
    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 100,
        ef_search: int = 128,
        meta_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Dense vector search, optionally with metadata filter."""

    def hybrid_search(
        self,
        collection: str,
        text: str,
        vector: list[float],
        top_k: int = 100,
        alpha: float = 0.7,
    ) -> list[SearchResult]:
        """Hybrid (dense + sparse/keyword) search. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support hybrid search")

    def get_memory_usage(self) -> int | None:
        """Return memory usage in bytes, or None if unavailable."""
        return None

    def supports_hybrid(self) -> bool:
        """Whether this adapter supports hybrid search."""
        return False

    def supports_graph(self) -> bool:
        """Whether this adapter supports graph-boosted search."""
        return False

    def delete_collection(self, name: str) -> None:
        """Delete a collection. Override if cleanup needed."""
        pass

    def teardown(self) -> None:
        """Clean up resources. Override if needed."""
        pass
