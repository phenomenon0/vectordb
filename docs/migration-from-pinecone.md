# Migrating from Pinecone to VectorDB

## Overview

Pinecone is a managed vector database. VectorDB is self-hosted, giving you full control over data, costs, and infrastructure. This guide covers migrating from Pinecone's serverless or pod-based indexes.

## Concept Mapping

| Pinecone | VectorDB | Notes |
|----------|----------|-------|
| Index | Collection | One index → one collection |
| Namespace | Collection name or metadata | Use separate collections or `meta` tags |
| Vector | Document/Vector | VectorDB auto-embeds text |
| Metadata | `meta` (string map) | Similar key-value metadata |
| API key | JWT token | `VECTORDB_TOKEN` env var |
| Serverless | Self-hosted | Run your own server |
| `upsert()` | `POST /insert` with `upsert: true` | Same semantics |
| `query()` | `POST /query` | Semantic search |
| `delete()` | `POST /delete` | By ID |
| `fetch()` | `POST /query` with scan mode | No direct fetch-by-ID |
| `list()` | `POST /query` with scan mode | Paginated scan |

## Step-by-Step Migration

### 1. Export from Pinecone

```python
from pinecone import Pinecone
import json

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-index")

# List all namespaces
stats = index.describe_index_stats()
namespaces = list(stats.namespaces.keys()) or [""]

for ns in namespaces:
    # Paginate through all vectors
    all_ids = []
    for ids_batch in index.list(namespace=ns):
        all_ids.extend(ids_batch)

    # Fetch in batches of 100
    records = []
    for i in range(0, len(all_ids), 100):
        batch_ids = all_ids[i:i+100]
        result = index.fetch(ids=batch_ids, namespace=ns)

        for vid, vec in result.vectors.items():
            record = {
                "id": vid,
                "doc": vec.metadata.get("text", vec.metadata.get("content", "")),
                "meta": {k: str(v) for k, v in (vec.metadata or {}).items()
                         if k not in ("text", "content")},
                "collection": ns if ns else "default",
            }
            if record["doc"]:
                records.append(record)

    filename = f"pinecone-{ns or 'default'}.jsonl"
    with open(filename, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Exported {len(records)} vectors from namespace '{ns or 'default'}'")
```

### 2. Import into VectorDB

```bash
./vectordb-server &

for file in pinecone-*.jsonl; do
    vectordb-cli import --file "$file" --batch-size 500
done
```

### 3. Update Application Code

**Pinecone (before):**
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("my-index")

# Upsert
index.upsert(vectors=[{
    "id": "doc1",
    "values": [0.1, 0.2, ...],  # pre-computed embedding
    "metadata": {"source": "web", "category": "tech"}
}])

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"source": {"$eq": "web"}},
    include_metadata=True
)
```

**VectorDB (after):**
```python
from vectordb import VectorDBClient

client = VectorDBClient("http://localhost:8080")

# Insert (auto-embeds text — no need for external embedding)
client.insert(
    doc="Document text here",
    id="doc1",
    meta={"source": "web", "category": "tech"},
    collection="default"
)

# Search
results = client.search(
    query="search term",
    top_k=5,
    meta={"source": "web"},
    include_meta=True
)
```

### 4. Filter Translation

| Pinecone Filter | VectorDB equivalent |
|-----------------|---------------------|
| `{"field": {"$eq": "val"}}` | `"meta": {"field": "val"}` |
| `{"$and": [...]}` | `"meta": {"k1": "v1", "k2": "v2"}` |
| `{"$or": [...]}` | `"meta_any": [{"k": "v1"}, {"k": "v2"}]` |
| `{"field": {"$ne": "x"}}` | `"meta_not": {"field": "x"}` |
| `{"field": {"$gt": 5}}` | `"meta_ranges": [{"key": "field", "min": 5}]` |
| `{"field": {"$in": [...]}}` | `"meta_any": [{"field": "a"}, {"field": "b"}]` |

## Key Differences

1. **Self-hosted**: No API costs. You control your data and infrastructure.
2. **No embedding pipeline needed**: VectorDB embeds text server-side. No need for OpenAI/Cohere embedding API calls.
3. **Namespaces → Collections**: Map Pinecone namespaces to VectorDB collections.
4. **Hybrid search**: VectorDB supports dense + sparse (BM25) hybrid search natively — Pinecone requires separate sparse-dense indexes.
5. **No vendor lock-in**: Standard JSONL import/export. Data is always yours.
6. **Cost**: Self-hosted on a $20/mo Hetzner box can handle what costs $70+/mo on Pinecone.
