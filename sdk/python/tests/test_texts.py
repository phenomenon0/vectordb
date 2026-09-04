"""Text in, text out (CTL-02): the SDK sends ``texts`` beside or instead of
vectors and reads ``embedded_by`` back. Wire bodies are asserted exactly so a
field the SDK invents or drops is caught here, not by a server 400."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import ValidationError as PydanticValidationError

from deepdata import (
    AsyncDeepDataClient,
    DeepDataClient,
    TenantCollectionInfo,
    TenantDocumentInput,
    TenantSearchRequest,
    TenantUpsertDocumentRequest,
    TenantVectorField,
)

BASE = "http://deepdata.test"
TENANT = "acme"
ROOT = f"{BASE}/v3/tenants/{TENANT}"
_TEXT_FIELD = {
    "name": "text",
    "type": "dense",
    "index": {"type": "hnsw"},
    "embedding": {"provider": "hash"},
}


def _body(route: respx.Route, call: int = 0) -> object:
    return json.loads(route.calls[call].request.content)


def test_embedding_binding_makes_dim_optional() -> None:
    dense = {"name": "text", "type": "dense", "index": {"type": "hnsw"}}
    # Without a binding the server has nothing to size the field from.
    with pytest.raises(PydanticValidationError):
        TenantVectorField.model_validate(dense)
    field = TenantVectorField.model_validate({**dense, "embedding": {"provider": "ollama"}})
    assert field.dim == 0 and field.embedding is not None

    # The provider matrix mirrors the server: bm25 is sparse-only, and the only sparse option.
    with pytest.raises(PydanticValidationError):
        TenantVectorField.model_validate({**dense, "embedding": {"provider": "bm25"}})
    sparse = {"name": "terms", "type": "sparse", "dim": 4096, "index": {"type": "inverted"}}
    TenantVectorField.model_validate({**sparse, "embedding": {"provider": "bm25"}})
    with pytest.raises(PydanticValidationError):
        TenantVectorField.model_validate({**sparse, "embedding": {"provider": "ollama"}})

    # Collection info from a server that filled dim/model parses into the same model.
    info = TenantCollectionInfo.model_validate(
        {
            "name": "notes",
            "fields": [
                {
                    **dense,
                    "dim": 768,
                    "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
                }
            ],
        }
    )
    assert info.fields[0].embedding is not None
    assert info.fields[0].embedding.model == "nomic-embed-text"


def test_documents_and_searches_need_texts_or_vectors_not_both() -> None:
    for model in (TenantDocumentInput, TenantUpsertDocumentRequest):
        with pytest.raises(PydanticValidationError):
            model.model_validate({})
        with pytest.raises(PydanticValidationError):
            model.model_validate({"vectors": {"text": [0.1]}, "texts": {"text": "dup"}})
        assert model.model_validate({"texts": {"text": "hello"}}).vectors is None

    with pytest.raises(PydanticValidationError):
        TenantSearchRequest.model_validate({})
    with pytest.raises(PydanticValidationError):
        TenantSearchRequest.model_validate({"queries": {"a": [0.1]}, "texts": {"a": "x"}})
    # texts and queries share the two-field ceiling and the hybrid rule.
    with pytest.raises(PydanticValidationError):
        TenantSearchRequest.model_validate({"queries": {"a": [0.1]}, "texts": {"b": "x"}})
    with pytest.raises(PydanticValidationError):
        TenantSearchRequest.model_validate(
            {"queries": {"a": [0.1], "b": [0.2]}, "texts": {"c": "x"}, "hybrid_params": {"strategy": "rrf"}}
        )
    mixed = TenantSearchRequest.model_validate(
        {
            "queries": {"terms": {"indices": [1], "values": [1.0], "dim": 8}},
            "texts": {"text": "x"},
            "fallback": {"primary": "text", "secondary": "terms", "threshold": 0.5},
        }
    )
    assert mixed.texts == {"text": "x"}


def _text_routes() -> tuple[respx.Route, respx.Route, respx.Route, respx.Route]:
    create = respx.post(f"{ROOT}/collections").mock(
        return_value=httpx.Response(
            201, json={"status": "success", "tenant_id": TENANT, "message": "collection created"}
        )
    )
    insert = respx.post(f"{ROOT}/collections/notes/docs").mock(
        return_value=httpx.Response(201, json={"status": "success", "tenant_id": TENANT, "id": 7, "message": "ok"})
    )
    upsert = respx.put(f"{ROOT}/collections/notes/docs/7").mock(
        return_value=httpx.Response(200, json={"status": "success", "tenant_id": TENANT, "id": 7, "message": "ok"})
    )
    search = respx.post(f"{ROOT}/collections/notes/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "tenant_id": TENANT,
                "documents": [{"id": 7, "metadata": {"text": "hello"}}],
                "scores": [0.12],
                "candidates_examined": 1,
                "embedded_by": {"text": "hash:4"},
            },
        )
    )
    return create, insert, upsert, search


def _assert_text_bodies(
    create: respx.Route, insert: respx.Route, upsert: respx.Route, search: respx.Route
) -> None:
    assert _body(create) == {
        "name": "notes",
        "fields": [
            {
                "name": "text",
                "type": "dense",
                "dim": 0,
                "index": {"type": "hnsw"},
                "embedding": {"provider": "hash"},
            }
        ],
    }
    assert _body(insert) == {"id": 7, "texts": {"text": "hello"}, "metadata": {"text": "hello"}}
    assert _body(upsert) == {"texts": {"text": "hello again"}}
    assert _body(search) == {"texts": {"text": "hello"}, "top_k": 3}


def test_sync_texts_round_trip_exact_contract() -> None:
    with respx.mock:
        create, insert, upsert, search = _text_routes()
        with DeepDataClient(BASE, retry=None) as client:
            tenant = client.tenant(TENANT)
            tenant.create_collection("notes", [_TEXT_FIELD])
            tenant.insert("notes", id=7, texts={"text": "hello"}, metadata={"text": "hello"})
            tenant.upsert("notes", id=7, texts={"text": "hello again"})
            result = tenant.search("notes", texts={"text": "hello"}, top_k=3)
        _assert_text_bodies(create, insert, upsert, search)
        assert result.embedded_by == {"text": "hash:4"}
        assert result.documents[0].id == 7


@pytest.mark.asyncio
async def test_async_texts_round_trip_exact_contract() -> None:
    with respx.mock:
        create, insert, upsert, search = _text_routes()
        async with AsyncDeepDataClient(BASE, retry=None) as client:
            tenant = client.tenant(TENANT)
            await tenant.create_collection("notes", [_TEXT_FIELD])
            await tenant.insert("notes", id=7, texts={"text": "hello"}, metadata={"text": "hello"})
            await tenant.upsert("notes", id=7, texts={"text": "hello again"})
            result = await tenant.search("notes", texts={"text": "hello"}, top_k=3)
        _assert_text_bodies(create, insert, upsert, search)
        assert result.embedded_by == {"text": "hash:4"}
