"""Integration tests against a live DeepData server.

Skipped unless DEEPDATA_URL is set in the environment.

    DEEPDATA_URL=http://localhost:8080 pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

from deepdata import DeepDataClient, AsyncDeepDataClient


DEEPDATA_URL = os.environ.get("DEEPDATA_URL")
pytestmark = pytest.mark.skipif(
    DEEPDATA_URL is None,
    reason="DEEPDATA_URL not set — skipping integration tests",
)


@pytest.fixture
def live_client() -> DeepDataClient:
    assert DEEPDATA_URL is not None
    return DeepDataClient(DEEPDATA_URL, timeout=30.0)


@pytest.fixture
def async_live_client() -> AsyncDeepDataClient:
    assert DEEPDATA_URL is not None
    return AsyncDeepDataClient(DEEPDATA_URL, timeout=30.0)


class TestLiveHealth:
    def test_health(self, live_client: DeepDataClient) -> None:
        health = live_client.health()
        assert health.ok is True
        assert health.total >= 0


class TestLiveInsertSearchDelete:
    def test_roundtrip(self, live_client: DeepDataClient) -> None:
        doc_id = f"sdk-test-{uuid.uuid4().hex[:8]}"

        # Insert
        result = live_client.insert(
            "The Python SDK integration test document",
            id=doc_id,
            meta={"source": "sdk-test"},
        )
        assert result.id == doc_id

        # Search
        results = live_client.search(
            "Python SDK test",
            top_k=5,
            include_meta=True,
        )
        assert len(results.ids) > 0

        # Delete
        delete_result = live_client.delete(doc_id)
        assert delete_result.deleted == doc_id


class TestLiveBatchInsert:
    def test_batch(self, live_client: DeepDataClient) -> None:
        ids = [f"sdk-batch-{uuid.uuid4().hex[:8]}" for _ in range(3)]
        docs = [
            {"doc": f"Batch doc {i}", "id": ids[i], "meta": {"batch": "true"}}
            for i in range(3)
        ]
        result = live_client.batch_insert(docs)
        assert len(result.ids) == 3

        # Cleanup
        for doc_id in ids:
            live_client.delete(doc_id)


class TestLiveCollections:
    def test_list(self, live_client: DeepDataClient) -> None:
        collections = live_client.list_collections()
        assert collections.count >= 0


@pytest.mark.asyncio
class TestAsyncLive:
    async def test_health(self, async_live_client: AsyncDeepDataClient) -> None:
        health = await async_live_client.health()
        assert health.ok is True
