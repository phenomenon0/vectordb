"""Live contract tests for the authenticated canonical V3 client.

The seed phase leaves one collection on disk. CI restarts the server against
the same data directory before running the verify phase, proving that the
Python surface observes the durable canonical state rather than a mock.
"""

from __future__ import annotations

import os

import pytest

from deepdata import AsyncDeepDataClient, DeepDataClient


DEEPDATA_URL = os.environ.get("DEEPDATA_URL")
DEEPDATA_API_TOKEN = os.environ.get("DEEPDATA_API_TOKEN")
INTEGRATION_REQUIRED = os.environ.get("DEEPDATA_INTEGRATION_REQUIRED") == "1"
RUN_ID = os.environ.get("DEEPDATA_INTEGRATION_ID", "local")
TENANT = f"sdk-{RUN_ID}"
RESTART_COLLECTION = f"restart-{RUN_ID}"
ASYNC_COLLECTION = f"async-{RUN_ID}"
FIELD = {
    "name": "embedding",
    "type": "dense",
    "dim": 3,
    "index": {"type": "flat"},
}

if INTEGRATION_REQUIRED and (not DEEPDATA_URL or not DEEPDATA_API_TOKEN):
    raise RuntimeError(
        "required live SDK contract needs DEEPDATA_URL and DEEPDATA_API_TOKEN"
    )

pytestmark = pytest.mark.skipif(
    not DEEPDATA_URL or not DEEPDATA_API_TOKEN,
    reason="live canonical server credentials are not configured",
)


def _sync_client() -> DeepDataClient:
    assert DEEPDATA_URL is not None
    assert DEEPDATA_API_TOKEN is not None
    return DeepDataClient(
        DEEPDATA_URL,
        api_token=DEEPDATA_API_TOKEN,
        timeout=30.0,
        retry=None,
    )


def _async_client() -> AsyncDeepDataClient:
    assert DEEPDATA_URL is not None
    assert DEEPDATA_API_TOKEN is not None
    return AsyncDeepDataClient(
        DEEPDATA_URL,
        api_token=DEEPDATA_API_TOKEN,
        timeout=30.0,
        retry=None,
    )


@pytest.mark.integration_seed
def test_sync_canonical_lifecycle_seeds_restart_fixture() -> None:
    with _sync_client() as client:
        tenant = client.tenant(TENANT)
        created = tenant.create_collection(
            RESTART_COLLECTION,
            [FIELD],
            metadata={"contract": "python-live"},
            description="restart persistence fixture",
        )
        assert created.tenant_id == TENANT

        inserted = tenant.insert(
            RESTART_COLLECTION,
            id=101,
            vectors={"embedding": [1.0, 0.0, 0.0]},
            metadata={"source": "sync"},
        )
        batch = tenant.batch_insert(
            RESTART_COLLECTION,
            [
                {
                    "id": 102,
                    "vectors": {"embedding": [0.0, 1.0, 0.0]},
                    "metadata": {"source": "batch"},
                },
                {
                    "id": 103,
                    "vectors": {"embedding": [0.0, 0.0, 1.0]},
                },
            ],
        )
        assert inserted.id == 101
        assert batch.ids == [102, 103]
        assert batch.inserted == 2

        deleted = tenant.delete_document(RESTART_COLLECTION, 103)
        assert deleted.tenant_id == TENANT
        result = tenant.search(
            RESTART_COLLECTION,
            queries={"embedding": [1.0, 0.0, 0.0]},
            top_k=10,
            include_vectors=True,
        )
        assert {document.id for document in result.documents} == {101, 102}
        assert all(document.vectors is not None for document in result.documents)

        collection = tenant.get_collection(RESTART_COLLECTION)
        listing = tenant.list_collections()
        info = tenant.info()
        assert collection.collection.name == RESTART_COLLECTION
        assert [item.name for item in listing.collections] == [RESTART_COLLECTION]
        assert info.collection_count == 1
        assert info.total_documents == 2


@pytest.mark.integration_seed
@pytest.mark.asyncio
async def test_async_canonical_lifecycle_and_cleanup() -> None:
    async with _async_client() as client:
        tenant = client.tenant(TENANT)
        await tenant.create_collection(ASYNC_COLLECTION, [FIELD])
        inserted = await tenant.insert(
            ASYNC_COLLECTION,
            id=201,
            vectors={"embedding": [0.0, 1.0, 0.0]},
            metadata={"source": "async"},
        )
        result = await tenant.search(
            ASYNC_COLLECTION,
            queries={"embedding": [0.0, 1.0, 0.0]},
            top_k=1,
            include_vectors=False,
        )
        assert inserted.id == 201
        assert [document.id for document in result.documents] == [201]
        assert result.documents[0].vectors is None
        await tenant.delete_document(ASYNC_COLLECTION, 201)
        await tenant.delete_collection(ASYNC_COLLECTION)


@pytest.mark.integration_verify
def test_restart_persistence_and_cleanup() -> None:
    with _sync_client() as client:
        tenant = client.tenant(TENANT)
        collection = tenant.get_collection(RESTART_COLLECTION)
        assert collection.collection.doc_count == 2

        result = tenant.search(
            RESTART_COLLECTION,
            queries={"embedding": [0.0, 1.0, 0.0]},
            top_k=10,
            include_vectors=False,
        )
        assert {document.id for document in result.documents} == {101, 102}
        assert all(document.vectors is None for document in result.documents)

        tenant.delete_document(RESTART_COLLECTION, 101)
        tenant.delete_document(RESTART_COLLECTION, 102)
        deleted = tenant.delete_collection(RESTART_COLLECTION)
        assert deleted.tenant_id == TENANT
        assert tenant.list_collections().count == 0
