# DeepData Python SDK

Typed sync and async clients for the DeepData vector database.

Candidate distribution: `deepdata-client==0.2.0rc1`; the Python import remains
`deepdata`.

The tenant-aware V3 API is the package's only database surface. Start every
operation with `client.tenant(...)`; root and V1/V2 helpers are not included.

## Installation

```bash
pip install deepdata-client
```

Or install from source:

```bash
cd sdk/python
pip install -e ".[dev]"
```

## Canonical V3 quick start

```python
from deepdata import DeepDataClient

with DeepDataClient(
    "http://localhost:8080",
    api_token="sk-...",
) as client:
    tenant = client.tenant("org-123")

    tenant.create_collection(
        "papers",
        fields=[
            {
                "name": "embedding",
                "type": "dense",
                "dim": 3,
                "index": {"type": "hnsw"},
            }
        ],
        description="Research papers",
    )

    inserted = tenant.insert(
        "papers",
        vectors={"embedding": [0.1, 0.2, 0.3]},
        metadata={"topic": "ml"},
    )

    results = tenant.search(
        "papers",
        queries={"embedding": [0.1, 0.2, 0.3]},
        top_k=5,
    )

    tenant.delete_document("papers", inserted.id)
```

Tenant and collection identifiers must contain 1–64 ASCII letters, digits,
hyphens, or underscores. The SDK validates them before issuing a request so a
dynamic identifier cannot alter the URL path.

Canonical V3 accepts caller-supplied vectors. The SDK does not invoke an
embedding model or external embedding provider.

## Collections

Collection methods return Pydantic response models rather than untyped
dictionaries.

```python
collections = tenant.list_collections()
for collection in collections.collections:
    print(collection.name, collection.doc_count)

details = tenant.get_collection("papers")
print(details.collection.fields)

tenant.delete_collection("papers")
```

Canonical field definitions use an `index` object. Dense fields support only
`hnsw` and `flat`; sparse fields support only `inverted`. The SDK rejects other
index names and invalid type/index combinations before making an HTTP request.

## Document inserts

Single inserts may request a positive integer ID. If `id` is omitted, the
server assigns one.

```python
result = tenant.insert(
    "papers",
    id=1001,
    vectors={"embedding": [0.1, 0.2, 0.3]},
    metadata={"source": "archive"},
)
print(result.id)
```

V3 batch insert is all-or-nothing. The SDK validates each document's typed
shape before sending the request; the server validates vector contents and
either acknowledges the whole batch or returns an error. There is no
partial-success mode.

```python
batch = tenant.batch_insert(
    "papers",
    [
        {"vectors": {"embedding": [0.1, 0.2, 0.3]}},
        {
            "id": 1002,
            "vectors": {"embedding": [0.3, 0.2, 0.1]},
            "metadata": {"topic": "databases"},
        },
    ],
)
print(batch.ids, batch.inserted)
```

## Search filters, hybrid fusion, and vectors

```python
results = tenant.search(
    "papers",
    queries={
        "embedding": [0.1, 0.2, 0.3],
        "keywords": {"indices": [4, 9], "values": [0.8, 0.4], "dim": 10000},
    },
    top_k=20,
    ef_search=128,
    filters={"topic": {"$eq": "ml"}},
    hybrid_params={
        "strategy": "weighted",
        "weights": {"embedding": 0.8, "keywords": 0.2},
    },
    include_vectors=True,
)

for document, score in zip(results.documents, results.scores):
    print(document.id, score, document.metadata)
```

Search accepts one vector field, or exactly two fields for hybrid fusion.

`include_vectors` defaults to the server default (currently omitted vectors).
Pass `True` only when the returned vectors are needed.

## Async client

The async tenant surface has the same paths, payloads, validation, and response
models as the synchronous client.

```python
import asyncio
from deepdata import AsyncDeepDataClient

async def main() -> None:
    async with AsyncDeepDataClient(
        "http://localhost:8080",
        api_token="sk-...",
    ) as client:
        tenant = client.tenant("org-123")
        result = await tenant.search(
            "papers",
            queries={"embedding": [0.1, 0.2, 0.3]},
            top_k=5,
        )
        print(result.documents)

asyncio.run(main())
```

## Errors and retries

```python
from deepdata.errors import NotFoundError, PermissionError, ServerError

try:
    tenant.get_collection("missing")
except NotFoundError:
    print("Collection does not exist")
except PermissionError:
    print("Token cannot access this tenant or collection")
except ServerError:
    print("Server error after configured retries")
```

Every `APIError` carries the server's envelope: `code`, `message`, `hint`,
`field`, `request_id`, `retryable`, `retry_after` (seconds, from
`retry_after_ms` or the `Retry-After` header) and `docs`
(`deepdata/errors.py:32-55`). `str(err)` reads
`deepdata api 404 not_found: collection not found: missing for tenant acme Hint: list the tenant's collections ...`.
The subclass follows the status code (`classify_error`, `deepdata/errors.py:129`);
a `quota_exceeded` 409 is a plain `APIError` with `retryable=False`.

Retries are enabled by default and follow the server's `retryable` verdict
(`deepdata/_utils.py:45`): a 429 `rate_limited` is retried, a 409
`quota_exceeded` is not. They can be configured or disabled:

```python
from deepdata import DeepDataClient, RetryConfig

client = DeepDataClient(
    "http://localhost:8080",
    api_token="sk-...",
    retry=RetryConfig(max_retries=5, initial_delay=1.0),
)

client_without_retries = DeepDataClient(
    "http://localhost:8080",
    api_token="sk-...",
    retry=None,
)
```

## Development

```bash
cd sdk/python
pip install -e ".[dev]"
pytest -q
mypy deepdata
python -m build
```

Integration tests require a running server:

```bash
DEEPDATA_URL=http://localhost:8080 pytest tests/test_integration.py -v
```
