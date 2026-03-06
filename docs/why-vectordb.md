# Why VectorDB

VectorDB is a pure-Go embeddable vector search engine built for teams that want fast, simple, self-hosted similarity search without the operational overhead of managed services or Python runtimes.

## Design Goals

**1. Single binary, zero dependencies**

No Python, no Docker required, no external model APIs. Download one binary, run it. Embedding happens in-process via ONNX Runtime (or use your own embeddings). The server is a single Go process that handles HTTP, indexing, persistence, and search.

**2. CPU-first performance**

Designed for CPU-bound workloads on commodity hardware (Hetzner CX32, any 8-core box). SIMD-accelerated distance functions (AVX2+FMA), HNSW with tunable ef_search, and Product Quantization (PQ4/PQ-ADC) for memory-constrained deployments. Sub-10ms P50 query latency at 1M vectors.

**3. Batteries included**

Out of the box: HNSW, DiskANN, IVF-PQ, sparse inverted index, hybrid search (dense+BM25 with RRF fusion), multi-tenancy with JWT/RBAC, WAL persistence, snapshot export/import, Prometheus metrics, and a CLI. No assembly required.

**4. Embeddable**

Import as a Go library. No HTTP overhead, no serialization tax. Use `VectorStore` directly in your application for in-process search with the same HNSW/PQ indices the server uses.

## When to Use VectorDB

| Scenario | VectorDB | Alternatives |
|----------|----------|-------------|
| Self-hosted RAG on a single node | Best fit — single binary, embedded ONNX | Chroma (Python), Qdrant (Rust) |
| Go application needing vector search | Native Go library, no FFI/CGO | pgvector (requires Postgres) |
| Multi-tenant SaaS with collection isolation | Built-in JWT + RBAC + per-tenant rate limits | Pinecone (managed), Weaviate |
| Edge/IoT with memory constraints | PQ4 compression: 16 bytes/vector vs 512 raw | Faiss (C++, needs Python bindings) |
| Hybrid keyword + semantic search | Dense + sparse + RRF fusion in one engine | Vespa (Java, heavy), Elasticsearch |

## When NOT to Use VectorDB

- **Billions of vectors**: VectorDB is optimized for up to ~10M vectors per node. For billion-scale, consider Milvus or Pinecone.
- **Managed service**: If you don't want to run infrastructure, use a managed offering.
- **GPU-accelerated training**: VectorDB is an inference/search engine, not a training framework.

## Comparison

| Feature | VectorDB | Chroma | Qdrant | Pinecone |
|---------|----------|--------|--------|----------|
| Language | Go | Python | Rust | Managed |
| Self-hosted | Yes | Yes | Yes | No |
| Embeddable | Yes (Go lib) | Yes (Python) | No (server only) | No |
| HNSW | Yes | Yes | Yes | Yes |
| DiskANN | Yes | No | No | No |
| PQ Compression | PQ4 + PQ-ADC | No | Yes (scalar) | Yes |
| Sparse/BM25 | Yes | No | Yes | Yes |
| Hybrid Search | RRF fusion | No | RRF fusion | Yes |
| Multi-tenancy | JWT + RBAC | No | API keys | Namespaces |
| WAL Persistence | Yes | Yes | Yes | Managed |
| Binary Size | ~15MB | ~200MB+ (Python) | ~30MB | N/A |
| Dependencies | None | Python, pip | None | API key |

## Architecture at a Glance

```
Client (Go/Python/TS/curl)
    │
    ▼
┌─────────────────────────────┐
│  HTTP Server (:8080)        │
│  JWT auth · rate limiting   │
│  Prometheus /metrics        │
├─────────────────────────────┤
│  Embedder (ONNX / hash)    │
│  Reranker (cross-encoder)  │
├─────────────────────────────┤
│  Index Layer                │
│  HNSW · DiskANN · IVF-PQ  │
│  Sparse · PQ4 · PQ-ADC    │
├─────────────────────────────┤
│  VectorStore (flat buffer)  │
│  Collection · Tenant maps   │
│  Metadata · Tombstones      │
├─────────────────────────────┤
│  Persistence                │
│  WAL · Snapshots · Export   │
└─────────────────────────────┘
```

## Getting Started

```bash
# Install
go install github.com/phenomenon0/Agent-GO/vectordb@latest

# Or Docker
docker run -p 8080:8080 -v vectordb-data:/data vectordb

# Insert
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{"doc": "VectorDB is fast", "collection": "docs"}'

# Search
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "speed", "top_k": 5, "collection": "docs"}'
```

See [README.md](../README.md) for full documentation links.
