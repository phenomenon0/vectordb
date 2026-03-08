"""Unit tests for the async DeepData client."""

from __future__ import annotations

import httpx
import pytest
import respx

from deepdata import (
    AsyncDeepDataClient,
    InsertResponse,
    SearchResult,
    HealthResponse,
)
from deepdata.errors import AuthenticationError

BASE = "http://testserver:8080"

pytestmark = pytest.mark.asyncio


class TestAsyncInsert:
    async def test_insert(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/insert").mock(
                return_value=httpx.Response(200, json={"id": "abc-123"})
            )
            async with AsyncDeepDataClient(BASE, retry=None) as client:
                result = await client.insert("Hello world")
                assert isinstance(result, InsertResponse)
                assert result.id == "abc-123"


class TestAsyncSearch:
    async def test_search(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/query").mock(
                return_value=httpx.Response(200, json={
                    "ids": ["a", "b"],
                    "docs": ["Doc A", "Doc B"],
                    "scores": [0.95, 0.87],
                })
            )
            async with AsyncDeepDataClient(BASE, retry=None) as client:
                result = await client.search("test query", top_k=10)
                assert isinstance(result, SearchResult)
                assert len(result.ids) == 2


class TestAsyncHealth:
    async def test_health(self) -> None:
        with respx.mock:
            respx.get(f"{BASE}/health").mock(
                return_value=httpx.Response(200, json={
                    "ok": True,
                    "total": 100,
                    "active": 90,
                    "deleted": 10,
                })
            )
            async with AsyncDeepDataClient(BASE, retry=None) as client:
                result = await client.health()
                assert isinstance(result, HealthResponse)
                assert result.ok is True


class TestAsyncErrors:
    async def test_401(self) -> None:
        with respx.mock:
            respx.get(f"{BASE}/health").mock(
                return_value=httpx.Response(401, text="unauthorized")
            )
            async with AsyncDeepDataClient(BASE, retry=None) as client:
                with pytest.raises(AuthenticationError):
                    await client.health()


class TestAsyncContextManager:
    async def test_context_manager(self) -> None:
        async with AsyncDeepDataClient(BASE) as client:
            assert client._http is not None


class TestAsyncCollections:
    async def test_create_collection(self) -> None:
        with respx.mock:
            respx.post(f"{BASE}/v2/collections").mock(
                return_value=httpx.Response(201, json={
                    "status": "success",
                    "message": 'collection "papers" created',
                })
            )
            async with AsyncDeepDataClient(BASE, retry=None) as client:
                result = await client.create_collection("papers")
                assert result["status"] == "success"

    async def test_delete_collection(self) -> None:
        with respx.mock:
            respx.delete(f"{BASE}/v2/collections/papers").mock(
                return_value=httpx.Response(200, json={"status": "success"})
            )
            async with AsyncDeepDataClient(BASE, retry=None) as client:
                result = await client.delete_collection("papers")
                assert result["status"] == "success"
