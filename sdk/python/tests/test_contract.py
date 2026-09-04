"""The SDK models against the contract they project.

The Go engine, the JSON schemas under ``api/contract/v3`` and these Pydantic
models are three renderings of one API. Nothing in Python imports the Go
side, so without these tests a field added on the server reaches Python only
when somebody notices — and a field removed reaches it as a runtime
``ValidationError``. Each test names the exceptions explicitly, so a new
divergence has to be argued for in the diff rather than appearing silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest

from deepdata.models import (
    TenantCollectionInfo,
    TenantFieldInfo,
    TenantIndexConfig,
    TenantSearchRequest,
    TenantSearchResponse,
)


CONTRACT = Path(__file__).resolve().parents[3] / "api" / "contract" / "v3"


def _schema(tool: str, section: str) -> dict:
    return json.loads((CONTRACT / "schemas" / f"{tool}.json").read_text())[section]


def _properties(node: dict) -> dict:
    return node["properties"]


def test_operations_list_parses_with_unique_names() -> None:
    """The operation list is what ``routes`` and ``/v3/status`` render."""

    operations = json.loads((CONTRACT / "operations.json").read_text())
    assert operations, "operations.json lists no operations"
    names = [op["name"] for op in operations]
    assert len(names) == len(set(names)), f"duplicate operation names in {names}"
    for op in operations:
        assert op["path"].startswith("/v3/"), op
        assert op["permission"] in {"read", "write", "admin"}, op
        assert op["method"] in {"GET", "POST", "PUT", "DELETE"}, op


def test_index_type_literal_matches_the_create_collection_enum() -> None:
    """The SDK's index vocabulary is the contract's, not a copy that drifted.

    The Go engine derives its list from ``collection.IndexTypes``; this Literal
    is the third rendering of it. Narrower than the enum, the SDK refuses a
    collection the server would create; wider, it sends a type the server 400s
    on. Either way the caller only finds out at runtime.
    """

    fields = _properties(_schema("deepdata_create_collection", "input"))["fields"]
    index = _properties(_properties(fields["items"])["index"])
    enum = index["type"]["enum"]
    assert list(get_args(TenantIndexConfig.model_fields["type"].annotation)) == enum
    # Every name in the enum must also survive the SDK's per-type parameter
    # rules, which key off the same vocabulary.
    for name in enum:
        assert TenantIndexConfig(type=name).type == name


def test_search_request_fields_are_in_the_recall_input_schema() -> None:
    """Every knob the SDK offers is one the agent contract documents."""

    schema = _properties(_schema("deepdata_recall", "input"))
    # The MCP tool is a narrowed projection of the HTTP call: it takes a
    # collection name instead of a client-held one, and hides the index and
    # payload knobs that only a vector-level caller sets.
    http_only = {
        "texts": "MCP sends query/queries; texts is the HTTP spelling",
        "ef_search": "index tuning, deliberately not exposed to agents",
        "include_vectors": "MCP never returns vectors",
        "hybrid_params": "MCP fuses implicitly when two fields are bound",
    }
    for name in TenantSearchRequest.model_fields:
        assert name in schema or name in http_only, (
            f"TenantSearchRequest.{name} is neither in the recall input schema "
            "nor a documented HTTP-only knob"
        )

    # The SDK must not accept more than the server will take.
    top_k = TenantSearchRequest.model_fields["top_k"]
    assert any(getattr(m, "le", None) == 1000 for m in top_k.metadata), (
        "top_k must stay capped at the engine's MaxSearchTopK"
    )
    assert schema["top_k"]["maximum"] <= 1000


def test_search_response_fields_track_the_recall_output_schema() -> None:
    """A confidence field the server sends must reach the caller typed."""

    schema = _properties(_schema("deepdata_recall", "output"))
    # The HTTP answer is richer than the MCP projection: it names the tenant,
    # returns whole documents rather than elided hits, and carries the
    # operational fields MCP does not forward.
    transport_only = {
        "status": "HTTP envelope",
        "tenant_id": "HTTP envelope",
        "documents": "projected onto hits[] over MCP",
        "scores": "projected onto hits[].score over MCP",
        "candidates_examined": "operational, not forwarded over MCP",
        "query_time_ms": "operational, not forwarded over MCP",
        "request_id": "operational, not forwarded over MCP",
    }
    for name in TenantSearchResponse.model_fields:
        assert name in schema or name in transport_only, (
            f"TenantSearchResponse.{name} is neither in the recall output "
            "schema nor a documented transport concern"
        )
    for name in (
        "best_score",
        "weak_match",
        "fell_back_to",
        "embedded_by",
        "score_direction",
    ):
        assert name in TenantSearchResponse.model_fields, (
            f"the contract publishes {name}; the SDK must decode it"
        )


def test_collection_info_fields_track_the_collections_output_schema() -> None:
    schema = _properties(_schema("deepdata_collections", "output"))
    item = _properties(schema["collections"]["items"])
    # metadata is free-form and left to the schema's additionalProperties.
    for name in TenantCollectionInfo.model_fields:
        assert name in item or name == "metadata", (
            f"TenantCollectionInfo.{name} is not a property of the "
            "collections output schema"
        )
    field_schema = _properties(item["fields"]["items"])
    for name in TenantFieldInfo.model_fields:
        assert name in field_schema, (
            f"TenantFieldInfo.{name} is not a property of the collections output schema"
        )
    assert "score_direction" in field_schema


@pytest.mark.parametrize("direction", ["lower_is_better", "higher_is_better"])
def test_field_info_accepts_score_direction_and_ignores_unknown_keys(
    direction: str,
) -> None:
    """Read models must survive a server that grew a field.

    ``TenantVectorField`` forbids extras so a typo in a create request fails
    loudly; the read shape cannot inherit that, or every additive server
    change would break existing clients.
    """

    info = TenantFieldInfo.model_validate(
        {
            "name": "dense",
            "type": "dense",
            "dim": 4,
            "index": {"type": "flat"},
            "score_direction": direction,
            "invented_later": True,
        }
    )
    assert info.score_direction == direction
