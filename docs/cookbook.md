# DeepData Cookbook

Practical recipes for common use cases.

---

## 1. Building a RAG System

Retrieval-Augmented Generation: store documents, retrieve relevant chunks, feed to an LLM.

### Ingest Documents

```python
from deepdata import DeepDataClient

client = DeepDataClient("http://localhost:8080")

# Split documents into chunks (400-800 tokens each)
chunks = [
    {"doc": "Python is a programming language...", "meta": {"source": "wiki", "topic": "python"}},
    {"doc": "Go is a statically typed language...", "meta": {"source": "wiki", "topic": "go"}},
    {"doc": "Rust focuses on memory safety...", "meta": {"source": "wiki", "topic": "rust"}},
]

# Batch insert
client.batch_insert(chunks, collection="knowledge-base")
```

### Retrieve & Generate

```python
def rag_query(question: str, llm_client) -> str:
    # 1. Retrieve relevant chunks
    results = client.search(
        query=question,
        top_k=5,
        collection="knowledge-base",
    )

    # 2. Build context
    context = "\n\n".join(r.doc for r in results)

    # 3. Generate answer
    response = llm_client.chat([
        {"role": "system", "content": f"Answer using this context:\n\n{context}"},
        {"role": "user", "content": question}
    ])

    return response
```

---

## 2. Hybrid Search (Dense + Sparse)

Combine semantic (dense) search with keyword (BM25) matching for best recall.

### Setup with V2 API

```bash
# Create a hybrid collection
curl -X POST http://localhost:8080/v2/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "articles",
    "fields": [
      {"name": "embedding", "type": 0, "dim": 384,
       "index": {"type": 0, "params": {"m": 16, "ef_construction": 200}}},
      {"name": "keywords", "type": 1, "dim": 30000,
       "index": {"type": 4}}
    ]
  }'
```

### Query with Hybrid Scoring

```python
results = client.search(
    query="machine learning optimization",
    top_k=10,
    collection="articles",
    score_mode="hybrid",    # Combine dense + lexical scores
    hybrid_alpha=0.7        # 70% semantic, 30% keyword
)
```

### Tuning Alpha

| Use Case | Alpha | Why |
|----------|-------|-----|
| General search | 0.7 | Semantic-heavy, keyword backup |
| Technical docs | 0.5 | Balance (exact terms matter) |
| Code search | 0.3 | Keywords dominate (function names, etc.) |
| Creative writing | 0.9 | Semantic meaning over exact words |

---

## 3. Multi-Tenant SaaS

Isolate data per customer with JWT authentication.

### Setup

```bash
JWT_SECRET="$(openssl rand -hex 32)" JWT_REQUIRED=true ./deepdata serve
```

### Provision a Customer

```bash
# Generate customer token
TOKEN=$(deepdata-cli gentoken --tenant customer-42 \
  --permissions read,write \
  --collections customer-42-docs \
  --expires 8760h \
  --secret "$JWT_SECRET" --json | jq -r .token)

# Set quota
curl -X POST http://localhost:8080/admin/quota/set \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"tenant_id": "customer-42", "max_vectors": 50000}'

# Set rate limit
curl -X POST http://localhost:8080/admin/ratelimit/set \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"tenant_id": "customer-42", "rps": 50}'
```

### Customer Usage

```python
from deepdata import DeepDataClient

# Customer's application uses their token
client = DeepDataClient(
    "http://deepdata.yourservice.com",
    api_key=customer_token
)

# They can only access their collection
client.insert("My document", collection="customer-42-docs")
results = client.search("find this", collection="customer-42-docs")

# Access to other collections is denied (403)
```

---

## 4. HNSW Parameter Tuning

### Parameters

| Parameter | Default | Effect | Trade-off |
|-----------|---------|--------|-----------|
| `m` | 16 | Edges per node | Higher = better recall, more RAM |
| `ef_construction` | 200 | Build-time search width | Higher = better index quality, slower build |
| `ef_search` | 50 | Query-time search width | Higher = better recall, slower queries |

### Benchmarking

```bash
# Low recall, fast queries (real-time search)
ef_search=20   # ~95% recall, <2ms

# Balanced (default)
ef_search=50   # ~98% recall, ~5ms

# High recall (accuracy-critical)
ef_search=200  # ~99.5% recall, ~15ms

# Maximum recall (near-exact)
ef_search=500  # ~99.9% recall, ~40ms
```

### Per-Query Override

```python
results = client.search(
    query="precision search",
    top_k=10,
    ef_search=200  # Override for this query only
)
```

### When to Rebuild the Index

Rebuild with higher `m` and `ef_construction` if:
- Recall drops below 95% at your target latency
- Dataset grows 10x from initial index creation
- You need sub-millisecond latency (reduce `m` to 8)

```bash
curl -X POST http://localhost:8080/compact
```

---

## 5. Embedding Modes

### Local (Ollama)

Default mode. Requires Ollama running locally:

```bash
VECTORDB_MODE=local ./deepdata serve
```

### Pro (OpenAI)

Uses OpenAI's `text-embedding-3-small` (~$0.02/1M tokens):

```bash
VECTORDB_MODE=pro OPENAI_API_KEY=sk-... ./deepdata serve
```

### Hash Embedder (Benchmarking)

Deterministic hash — zero network calls, zero cost. Not semantically meaningful, but useful for benchmarking insert/query throughput:

```bash
USE_HASH_EMBEDDER=1 ./deepdata serve
```

### Bring Your Own Embeddings

Embed client-side and send raw vectors:

```python
import openai

def embed(text):
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

# Insert with pre-computed vector
import requests
requests.post("http://localhost:8080/insert", json={
    "doc": "Hello world",
    "vector": embed("Hello world"),
    "collection": "docs"
})
```

### Direct Embedding Endpoint

Get embeddings without inserting:

```bash
curl -X POST http://localhost:8080/api/embed -d '{"text":"hello world"}'
```

---

## 6. Backup & Disaster Recovery

### Automated Snapshots

```bash
SNAPSHOT_EXPORT_PATH=/backup/deepdata EXPORT_INTERVAL_MIN=30 ./deepdata serve
```

### Manual Backup

```bash
# Export
curl -s http://localhost:8080/export > deepdata-backup-$(date +%Y%m%d).bin

# Restore
curl -X POST http://localhost:8080/import --data-binary @deepdata-backup-20260101.bin
```

### Backup Strategy

| Strategy | RPO | Method |
|----------|-----|--------|
| WAL replay | ~seconds | WAL files persist all writes |
| Periodic snapshot | 30-60min | `EXPORT_INTERVAL_MIN` |
| External backup | daily | CronJob + object storage |
| Streaming replication | ~real-time | WAL shipping to replicas |

### Testing Restore

```bash
# Start fresh instance
DATA_DIR=/tmp/restore-test PORT=8081 ./deepdata serve &

# Import backup
curl -X POST http://localhost:8081/import --data-binary @backup.bin

# Verify
curl http://localhost:8081/health | jq .total
```

---

## 7. gRPC Usage

DeepData exposes gRPC on port 50051 (configurable via `GRPC_PORT`). Same operations as the HTTP API, protobuf-encoded for lower overhead.

### Go Client via gRPC

```go
import (
    pb "github.com/phenomenon0/vectordb/api/proto/deepdata/v1"
    "google.golang.org/grpc"
)

conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
client := pb.NewDeepDataServiceClient(conn)

// Insert
resp, err := client.Insert(ctx, &pb.InsertRequest{
    Doc:        "Hello world",
    Collection: "docs",
})

// Search
results, err := client.Search(ctx, &pb.SearchRequest{
    Query: "Hello",
    TopK:  5,
})
```

### When to Use gRPC vs HTTP

| | HTTP | gRPC |
|---|---|---|
| Ease of use | curl, any language | Needs protobuf codegen |
| Throughput | Good | ~2x higher |
| Payload size | JSON overhead | Compact binary |
| Streaming | No | Yes |
| Browser support | Yes | Needs grpc-web proxy |
