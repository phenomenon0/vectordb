# Migrating from Qdrant to VectorDB

## Overview

Qdrant is a Rust-based vector database with gRPC and REST APIs. VectorDB offers similar features with a Go-native stack, Cowrie codec integration, and built-in agent tooling.

## Concept Mapping

| Qdrant | VectorDB | Notes |
|--------|----------|-------|
| Collection | Collection | Same concept |
| Point | Document/Vector | VectorDB uses string IDs |
| Named vectors | v2 multi-vector fields | Via `/v2/collections` API |
| Payload | Metadata (`meta`) | String key-value pairs |
| Payload index | Metadata bitmap index | Auto-indexed |
| Scroll API | `mode: "scan"` with pagination | Use `page_token` |
| Snapshot | `/export` + `/import` | Binary snapshot |
| API key | JWT token | Multi-tenant by default |
| gRPC API | HTTP REST | No gRPC yet |

## Step-by-Step Migration

### 1. Export from Qdrant

```python
from qdrant_client import QdrantClient
import json

qdrant = QdrantClient("http://localhost:6333")

for collection in qdrant.get_collections().collections:
    name = collection.name

    # Scroll through all points
    points = []
    offset = None
    while True:
        result = qdrant.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False  # We'll re-embed in VectorDB
        )
        batch, next_offset = result

        for point in batch:
            record = {
                "id": str(point.id),
                "doc": point.payload.get("text", point.payload.get("content", "")),
                "meta": {k: str(v) for k, v in point.payload.items()
                         if k not in ("text", "content")},
                "collection": name,
            }
            if record["doc"]:  # skip empty docs
                points.append(record)

        if next_offset is None:
            break
        offset = next_offset

    with open(f"{name}.jsonl", "w") as f:
        for p in points:
            f.write(json.dumps(p) + "\n")

    print(f"Exported {len(points)} points from '{name}'")
```

### 2. Import into VectorDB

```bash
./vectordb-server &

for file in *.jsonl; do
    vectordb-cli import --file "$file" --batch-size 500
done
```

### 3. Update Application Code

**Qdrant (before):**
```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient("http://localhost:6333")

# Insert
client.upsert(
    collection_name="docs",
    points=[PointStruct(
        id=1,
        vector=[0.1, 0.2, ...],  # pre-computed embedding
        payload={"text": "Hello world", "source": "web"}
    )]
)

# Search
results = client.search(
    collection_name="docs",
    query_vector=[0.1, 0.2, ...],
    query_filter=Filter(must=[
        FieldCondition(key="source", match=MatchValue(value="web"))
    ]),
    limit=5,
)
```

**VectorDB (after):**
```python
from vectordb import VectorDBClient

client = VectorDBClient("http://localhost:8080")

# Insert (auto-embeds text)
client.insert(
    doc="Hello world",
    meta={"source": "web"},
    collection="docs"
)

# Search (auto-embeds query)
results = client.search(
    query="Hello world",
    top_k=5,
    collection="docs",
    meta={"source": "web"}
)
```

### 4. Filter Translation

| Qdrant Filter | VectorDB equivalent |
|---------------|---------------------|
| `must: [FieldCondition(key, match)]` | `"meta": {"key": "value"}` |
| `should: [...]` | `"meta_any": [{"k": "v1"}, {"k": "v2"}]` |
| `must_not: [...]` | `"meta_not": {"key": "value"}` |
| `Range(gte=5, lte=10)` | `"meta_ranges": [{"key": "field", "min": 5, "max": 10}]` |

## Key Differences

1. **No pre-computed vectors needed**: VectorDB embeds text server-side (ONNX bge-small-en or external). No need to manage embedding pipelines.
2. **String IDs**: VectorDB uses string IDs (auto-generated UUIDs if omitted). Qdrant uses integer or UUID point IDs.
3. **Metadata types**: VectorDB metadata values are strings. For numeric filtering, use `meta_ranges`.
4. **No gRPC**: VectorDB is HTTP-only. For high-throughput, use batch endpoints and Cowrie encoding.
5. **Multi-tenancy**: VectorDB has built-in JWT-based multi-tenancy with per-tenant rate limiting and quotas.
