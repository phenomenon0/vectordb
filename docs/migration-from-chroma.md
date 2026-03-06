# Migrating from ChromaDB to VectorDB

## Overview

ChromaDB is a Python-first embedding database. VectorDB offers comparable features with a Go-native server, multi-tenancy, and hybrid search. This guide covers the migration path.

## Concept Mapping

| ChromaDB | VectorDB | Notes |
|----------|----------|-------|
| `Collection` | Collection | Same concept |
| `collection.add()` | `POST /insert` or `POST /batch_insert` | VectorDB auto-embeds text |
| `collection.query()` | `POST /query` | Same semantic search |
| `collection.delete()` | `POST /delete` | By ID |
| `collection.update()` | `POST /insert` with `upsert: true` | Upsert mode |
| `where` filter | `meta` / `meta_any` / `meta_not` | Similar metadata filtering |
| `where_document` | Not directly supported | Use metadata instead |
| Embedding function | Built-in ONNX or external | Set via server config |
| `PersistentClient` | Always persistent | WAL + snapshots by default |
| Tenant/database | Multi-tenancy with JWT | Per-tenant isolation |

## Step-by-Step Migration

### 1. Export from ChromaDB

```python
import chromadb
import json

client = chromadb.PersistentClient(path="./chroma-data")

for collection in client.list_collections():
    col = client.get_collection(collection.name)
    results = col.get(include=["documents", "metadatas", "embeddings"])

    with open(f"{collection.name}.jsonl", "w") as f:
        for i in range(len(results["ids"])):
            record = {
                "id": results["ids"][i],
                "doc": results["documents"][i] if results["documents"] else "",
                "meta": results["metadatas"][i] if results["metadatas"] else {},
                "collection": collection.name,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Exported {len(results['ids'])} docs from '{collection.name}'")
```

### 2. Import into VectorDB

```bash
# Start VectorDB
./vectordb-server &

# Import each collection
for file in *.jsonl; do
    vectordb-cli import --file "$file"
done
```

Or using the Python client:
```python
from vectordb import VectorDBClient

client = VectorDBClient("http://localhost:8080")

import json
with open("my-collection.jsonl") as f:
    docs = [json.loads(line) for line in f]

client.insert_batch(docs)
```

### 3. Update Application Code

**ChromaDB (before):**
```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("docs")

# Insert
collection.add(
    documents=["Hello world"],
    metadatas=[{"source": "web"}],
    ids=["doc1"]
)

# Query
results = collection.query(
    query_texts=["search term"],
    n_results=5,
    where={"source": "web"}
)
```

**VectorDB (after):**
```python
from vectordb import VectorDBClient

client = VectorDBClient("http://localhost:8080")

# Insert
client.insert(
    doc="Hello world",
    meta={"source": "web"},
    id="doc1",
    collection="docs"
)

# Query
results = client.search(
    query="search term",
    top_k=5,
    collection="docs",
    meta={"source": "web"}
)
```

### 4. Filter Translation

| ChromaDB `where` | VectorDB equivalent |
|-------------------|---------------------|
| `{"field": "value"}` | `"meta": {"field": "value"}` |
| `{"$and": [...]}` | `"meta": {"key1": "v1", "key2": "v2"}` (AND is default) |
| `{"$or": [...]}` | `"meta_any": [{"key1": "v1"}, {"key2": "v2"}]` |
| `{"field": {"$ne": "x"}}` | `"meta_not": {"field": "x"}` |
| `{"field": {"$gt": 5}}` | `"meta_ranges": [{"key": "field", "min": 5}]` |

## Key Differences

1. **Embedding**: ChromaDB requires you to pass an embedding function. VectorDB has built-in ONNX embedding (bge-small-en) — just send text.
2. **Authentication**: VectorDB supports JWT multi-tenancy out of the box.
3. **Hybrid search**: VectorDB supports dense + sparse (BM25) hybrid search natively.
4. **Performance**: VectorDB uses HNSW with SIMD-accelerated distance computation.
5. **Persistence**: VectorDB uses WAL + snapshots (no SQLite dependency).
