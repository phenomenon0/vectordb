# Why DeepData

DeepData is a pure-Go vector database built for teams that want fast, self-hosted similarity search without managed service lock-in or Python runtimes.

## Design Goals

**1. Single binary, zero dependencies**

No Python, no Docker required, no external model APIs. Download one binary, run it. Embedding happens in-process via Ollama or OpenAI, with a hash embedder for zero-dependency benchmarking. The server is a single Go process that handles HTTP, gRPC, indexing, persistence, and search.

**2. CPU-first performance**

Designed for CPU-bound workloads on commodity hardware. SIMD-accelerated distance functions (AVX2+FMA), HNSW with tunable ef_search, and Product Quantization (PQ4/PQ-ADC) for memory-constrained deployments. Sub-10ms P50 query latency at 1M vectors.

**3. Batteries included**

Out of the box: five index types (HNSW, DiskANN, IVF, Flat, Sparse/BM25), hybrid search with RRF fusion, multi-tenancy with JWT/RBAC, WAL persistence, streaming replication, encryption at rest, audit logging, gRPC and HTTP APIs, Prometheus metrics, OpenTelemetry tracing, Grafana dashboards, and a CLI. No assembly required.

**4. Embeddable**

Import as a Go library. No HTTP overhead, no serialization tax. Use `VectorStore` directly in your application for in-process search with the same HNSW/PQ indices the server uses.

## When to Use DeepData

| Scenario | DeepData | Alternatives |
|----------|----------|-------------|
| Self-hosted RAG on a single node | Best fit — single binary, built-in embedders | Chroma (Python), Qdrant (Rust) |
| Go application needing vector search | Native Go library, no FFI/CGO | pgvector (requires Postgres) |
| Multi-tenant SaaS with collection isolation | Built-in JWT + RBAC + per-tenant rate limits | Pinecone (managed), Weaviate |
| Edge/IoT with memory constraints | PQ4 compression: 16 bytes/vector vs 512 raw | Faiss (C++, needs Python bindings) |
| Hybrid keyword + semantic search | Dense + sparse + RRF fusion in one engine | Vespa (Java, heavy), Elasticsearch |
| Billion-scale on commodity hardware | DiskANN memory-maps graphs to disk | Milvus (distributed), Pinecone |

## When NOT to Use DeepData

- **Managed service**: If you don't want to run infrastructure, use Pinecone or similar.
- **GPU-accelerated training**: DeepData is an inference/search engine, not a training framework.

## Comparison

| Feature | DeepData | Chroma | Qdrant | Pinecone |
|---------|----------|--------|--------|----------|
| Language | Go | Python | Rust | Managed |
| Self-hosted | Yes | Yes | Yes | No |
| Embeddable | Yes (Go lib) | Yes (Python) | No (server only) | No |
| HNSW | Yes | Yes | Yes | Yes |
| DiskANN | Yes | No | No | No |
| PQ Compression | PQ4 + PQ-ADC | No | Scalar | Yes |
| Sparse/BM25 | Yes | No | Yes | Yes |
| Hybrid Search | RRF fusion | No | RRF fusion | Yes |
| gRPC API | Yes | No | Yes | No |
| Multi-tenancy | JWT + RBAC | No | API keys | Namespaces |
| Encryption at Rest | AES-256-GCM | No | No | Managed |
| Audit Logging | Yes | No | No | Managed |
| WAL Persistence | Yes | Yes | Yes | Managed |
| Binary Size | ~15MB | ~200MB+ | ~30MB | N/A |
| Dependencies | None | Python, pip | None | API key |

## Architecture at a Glance

```
Client (Go / Python / curl / gRPC)
    │
    ▼
┌─────────────────────────────────┐
│  HTTP (:8080) + gRPC (:50051)   │
│  JWT auth · RBAC · rate limiting│
│  Prometheus · OpenTelemetry     │
├─────────────────────────────────┤
│  Embedder (Ollama / OpenAI)     │
│  Hash embedder (benchmarks)     │
├─────────────────────────────────┤
│  Index Layer                    │
│  HNSW · DiskANN · IVF · Flat   │
│  Sparse/BM25 · PQ4 · PQ-ADC    │
├─────────────────────────────────┤
│  VectorStore                    │
│  Collections · Tenant maps      │
│  Metadata · Tombstones          │
├─────────────────────────────────┤
│  Persistence + Security         │
│  WAL · Snapshots · Replication  │
│  Encryption at rest · Audit log │
└─────────────────────────────────┘
```

## Getting Started

```bash
# Build from source
go build -o deepdata ./cmd/deepdata && ./deepdata serve

# Or Docker
docker compose up

# Insert
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{"doc": "DeepData is fast", "collection": "docs"}'

# Search
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "speed", "top_k": 5, "collection": "docs"}'
```

See [README.md](../README.md) for full documentation links.
