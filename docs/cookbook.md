# VectorDB Cookbook

Practical recipes for common use cases.

---

## 1. Building a RAG System

Retrieval-Augmented Generation: store documents, retrieve relevant chunks, feed to an LLM.

### Ingest Documents

```python
from vectordb import VectorDBClient

client = VectorDBClient("http://localhost:8080")

# Split documents into chunks (400-800 tokens each)
chunks = [
    {"doc": "Python is a programming language...", "meta": {"source": "wiki", "topic": "python"}},
    {"doc": "Go is a statically typed language...", "meta": {"source": "wiki", "topic": "go"}},
    {"doc": "Rust focuses on memory safety...", "meta": {"source": "wiki", "topic": "rust"}},
]

# Batch insert
client.insert_batch(chunks, collection="knowledge-base")
```

### Retrieve & Generate

```python
def rag_query(question: str, llm_client) -> str:
    # 1. Retrieve relevant chunks
    results = client.search(
        query=question,
        top_k=5,
        collection="knowledge-base",
        include_meta=True
    )

    # 2. Build context
    context = "\n\n".join(results.docs)

    # 3. Generate answer
    response = llm_client.chat([
        {"role": "system", "content": f"Answer using this context:\n\n{context}"},
        {"role": "user", "content": question}
    ])

    return response
```

### With LangChain

```python
from vectordb.integrations.langchain import VectorDBVectorStore
from langchain.chains import RetrievalQA

vectorstore = VectorDBVectorStore(
    url="http://localhost:8080",
    collection="knowledge-base"
)

qa = RetrievalQA.from_chain_type(
    llm=your_llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

answer = qa.run("What is Rust?")
```

---

## 2. Hybrid Search (Dense + Sparse)

Combine semantic (dense) search with keyword (BM25) matching for best recall.

### Setup with v2 API

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
JWT_SECRET="$(openssl rand -hex 32)" JWT_REQUIRED=true ./vectordb-server
```

### Provision a Customer

```bash
# Generate customer token
TOKEN=$(vectordb-cli gentoken --tenant customer-42 \
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
# Customer's application uses their token
client = VectorDBClient(
    "http://vectordb.yourservice.com",
    api_key=customer_token
)

# They can only access their collection
client.insert(doc="My document", collection="customer-42-docs")
results = client.search(query="find this", collection="customer-42-docs")

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

## 5. Embedding Model Selection

### Built-in: bge-small-en (Default)

- **Dimension**: 384
- **Speed**: 3-8ms/query on CPU
- **Quality**: Good for English text
- **Size**: ~33MB ONNX model

Best for: General English text, low-resource deployments.

### Using External Embeddings

If you need multilingual, code, or domain-specific embeddings:

```bash
# Disable built-in embedder
USE_HASH_EMBEDDER=1 ./vectordb-server
```

Then embed client-side and use the raw vector API:

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

### Model Comparison

| Model | Dim | Speed | Quality | Use Case |
|-------|-----|-------|---------|----------|
| bge-small-en | 384 | 3-8ms | Good | English text, low resource |
| bge-base-en | 768 | 10-20ms | Better | English, higher quality |
| e5-large | 1024 | 30-50ms | Best | Maximum quality |
| text-embedding-3-small | 1536 | API call | Great | Multilingual, API-based |
| nomic-embed-text | 768 | 5-15ms | Good | Open source, multilingual |

---

## 6. Backup & Disaster Recovery

### Automated Snapshots

```bash
# Export every 30 minutes to a backup path
SNAPSHOT_EXPORT_PATH=/backup/vectordb
EXPORT_INTERVAL_MIN=30
./vectordb-server
```

### Manual Backup

```bash
# Export
curl -s http://localhost:8080/export > vectordb-backup-$(date +%Y%m%d).bin

# Restore
curl -X POST http://localhost:8080/import --data-binary @vectordb-backup-20250101.bin
```

### Backup Strategy

| Strategy | RPO | Method |
|----------|-----|--------|
| WAL replay | ~seconds | WAL files persist all writes |
| Periodic snapshot | 30-60min | `EXPORT_INTERVAL_MIN` |
| External backup | daily | CronJob + object storage |

### Testing Restore

```bash
# Start fresh instance
DATA_DIR=/tmp/restore-test PORT=8081 ./vectordb-server &

# Import backup
curl -X POST http://localhost:8081/import --data-binary @backup.bin

# Verify
curl http://localhost:8081/health | jq .total
```
