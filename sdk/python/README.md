# DeepData Python SDK

Python client for the [DeepData](https://github.com/Neumenon/deepdata) vector database.

## Installation

```bash
pip install deepdata
```

Or from source:

```bash
cd sdk/python
pip install -e ".[dev]"
```

## Quick Start

```python
from deepdata import DeepDataClient

client = DeepDataClient("http://localhost:8080")

# Insert a document
result = client.insert("Machine learning is transforming AI", meta={"topic": "ml"})
print(f"Inserted: {result.id}")

# Search
results = client.search("artificial intelligence", top_k=5)
for doc_id, doc, score in zip(results.ids, results.docs, results.scores):
    print(f"  {score:.3f}  {doc_id}  {doc[:60]}")

# Delete
client.delete(result.id)
```

## Authentication

```python
client = DeepDataClient(
    "http://localhost:8080",
    api_token="sk-...",
    tenant_id="org-123",
)
```

## Async

```python
import asyncio
from deepdata import AsyncDeepDataClient

async def main():
    async with AsyncDeepDataClient("http://localhost:8080") as client:
        results = await client.search("hello world", top_k=5)
        print(results.ids)

asyncio.run(main())
```

## Batch Insert

```python
results = client.batch_insert([
    {"doc": "First document", "meta": {"year": "2026"}},
    {"doc": "Second document", "meta": {"year": "2025"}},
])
print(f"Inserted {len(results.ids)} documents")
```

## Collections

```python
# Create
client.create_collection("papers", fields=[
    {"name": "embedding", "type": "dense", "dim": 384, "index_type": "hnsw"},
])

# List
collections = client.list_collections()

# Stats
stats = client.collection_stats("papers")

# Delete
client.delete_collection("papers")
```

## Multi-Tenant (v3 API)

```python
tenant = client.tenant("org-123")
tenant.create_collection("docs", fields=[...])
tenant.insert("docs", doc="Hello world")
results = tenant.search("docs", query="Hello", top_k=5)
```

## Search Options

```python
results = client.search(
    "query text",
    top_k=20,
    mode="ann",                    # "ann", "scan", or "lex"
    collection="papers",
    meta={"author": "Alice"},      # exact match filter
    meta_not={"status": "draft"},  # exclusion filter
    include_meta=True,
    hybrid_alpha=0.7,
    ef_search=128,
)
```

## Error Handling

```python
from deepdata.errors import NotFoundError, RateLimitError, ServerError

try:
    client.get_collection("missing")
except NotFoundError:
    print("Collection doesn't exist")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except ServerError:
    print("Server error (will auto-retry by default)")
```

## Retry Configuration

```python
from deepdata import DeepDataClient, RetryConfig

# Custom retry settings
client = DeepDataClient(
    "http://localhost:8080",
    retry=RetryConfig(max_retries=5, initial_delay=1.0),
)

# Disable retries
client = DeepDataClient("http://localhost:8080", retry=None)
```

## Development

```bash
cd sdk/python
pip install -e ".[dev]"
pytest tests/ -v
mypy deepdata/
```

Integration tests (requires a running server):

```bash
DEEPDATA_URL=http://localhost:8080 pytest tests/test_integration.py -v
```
