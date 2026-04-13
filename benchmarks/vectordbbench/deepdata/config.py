"""DeepData configuration for VectorDBBench."""

from pydantic import SecretStr

from vectordb_bench.backend.clients.api import DBCaseConfig, DBConfig, MetricType


class DeepDataConfig(DBConfig):
    """DeepData connection configuration."""

    url: str = "http://localhost:8080"
    api_key: SecretStr | None = None

    def to_dict(self) -> dict:
        config = {"url": self.url}
        if self.api_key:
            config["api_key"] = self.api_key.get_secret_value()
        return config


class DeepDataIndexConfig(DBCaseConfig):
    """DeepData HNSW index and search configuration."""

    metric_type: MetricType = MetricType.COSINE
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 128

    def index_param(self) -> dict:
        return {
            "m": self.m,
            "ef_construction": self.ef_construction,
            "metric_type": self.metric_type,
        }

    def search_param(self) -> dict:
        return {
            "ef_search": self.ef_search,
        }
