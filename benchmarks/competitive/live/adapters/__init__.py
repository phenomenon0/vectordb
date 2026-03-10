"""VDB benchmark adapters."""

from .base import VDBAdapter
from .deepdata_adapter import DeepDataAdapter
from .weaviate_adapter import WeaviateAdapter
from .milvus_adapter import MilvusAdapter
from .qdrant_adapter import QdrantAdapter
from .chroma_adapter import ChromaAdapter

ALL_ADAPTERS = {
    "deepdata": DeepDataAdapter,
    "weaviate": WeaviateAdapter,
    "milvus": MilvusAdapter,
    "qdrant": QdrantAdapter,
    "chromadb": ChromaAdapter,
}

__all__ = [
    "VDBAdapter",
    "DeepDataAdapter",
    "WeaviateAdapter",
    "MilvusAdapter",
    "QdrantAdapter",
    "ChromaAdapter",
    "ALL_ADAPTERS",
]
