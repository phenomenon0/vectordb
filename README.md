# DeepData

**High-performance vector database. Single binary. Zero dependencies.**

[![Go](https://img.shields.io/badge/Go-1.24+-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/phenomenon0/vectordb)](https://github.com/phenomenon0/vectordb/releases/latest)

DeepData is a vector database written in Go. One binary, no runtime deps. Five index types (HNSW, IVF, DiskANN, Flat, Sparse/BM25), hybrid dense+sparse search, gRPC and HTTP APIs, quantization down to binary, and streaming replication. Runs on a laptop. Scales to a cluster.

## Download

Pre-built binaries — no Go toolchain required:

| Platform | Download |
|----------|----------|
| Linux x86_64 | [deepdata-linux-amd64](https://github.com/phenomenon0/vectordb/releases/latest/download/deepdata-linux-amd64) |
| macOS x86_64 | [deepdata-darwin-amd64](https://github.com/phenomenon0/vectordb/releases/latest/download/deepdata-darwin-amd64) |
| macOS ARM (Apple Silicon) | [deepdata-darwin-arm64](https://github.com/phenomenon0/vectordb/releases/latest/download/deepdata-darwin-arm64) |
| Windows x86_64 | [deepdata-windows-amd64.exe](https://github.com/phenomenon0/vectordb/releases/latest/download/deepdata-windows-amd64.exe) |

```bash
# Linux/macOS: download, make executable, run
chmod +x deepdata-linux-amd64
./deepdata-linux-amd64 serve
```

## Quick Start

```bash
go build ./cmd/deepdata && ./deepdata serve
# Web UI → http://localhost:8080
```

Or with Docker:

```bash
docker compose up
```

## Usage

Insert a document:

```bash
curl -X POST http://localhost:8080/insert \
  -d '{"doc": "vector databases use ANN for fast retrieval", "id": "doc-1", "meta": {"topic": "databases"}}'
```

Query it back:

```bash
curl -X POST http://localhost:8080/query \
  -d '{"query": "how do vector databases work?", "top_k": 5, "mode": "ann", "include_meta": true}'
```

Create a V2 collection (multi-vector, typed):

```bash
curl -X POST http://localhost:8080/v2/collections \
  -d '{"name": "papers", "dimension": 384, "distance": "cosine"}'
```

Metadata filtering with `meta` (AND), `meta_any` (OR), `meta_not` (NOT).

## Why DeepData

**Search.** Dense vectors, sparse BM25, or both fused via RRF. Reranking, pagination, and metadata filters — range, time, numeric. The query planner picks the fastest path.

**Indexes.** Five types. HNSW for low-latency recall. IVF for memory-constrained workloads. DiskANN when your data doesn't fit in RAM — it memory-maps the graph and streams from disk. Flat scan for small collections where building an index isn't worth it. Sparse inverted index for keyword/BM25 search. Quantization (FP16, Uint8, PQ8, PQ4, Binary) trades precision for 2x–32x memory savings. SIMD-accelerated distance kernels, optional CUDA.

**Scale.** WAL with CRC checksums for crash safety. Streaming replication from primary to replicas via WAL log shipping. Streaming snapshots (disk-backed, gzip-compressed) for bootstrapping new nodes without OOM. DiskANN handles billion-scale on commodity hardware. All scale limits — max collections, max tenants, rate limits — are env-configurable.

**Security.** JWT authentication, RBAC with fine-grained permissions, TLS and mutual TLS, encryption at rest (AES-256-GCM or ChaCha20-Poly1305 with Argon2id key derivation), and audit logging with 25+ event types covering auth, vector ops, admin, and cluster actions.

**Observability.** Prometheus metrics endpoint, OpenTelemetry traces (stdout or OTLP export), Grafana dashboard with 40+ panels and alerting rules, health endpoint for liveness probes.

**Multi-tenant.** Per-tenant collections with isolation, quotas, and rate limiting. One server, many tenants.

**Embedders.** Ollama and OpenAI built in, hot-swappable at runtime. Hash embedder for zero-cost benchmarking and testing. Set `VECTORDB_MODE=local` for Ollama, `VECTORDB_MODE=pro` for OpenAI.

**GraphRAG.** Entity extraction feeds a graph index that boosts hybrid search with relationship-aware scoring. Feedback loops let you feed relevance signals back to improve ranking over time.

## Architecture

```
DeepData/
  cmd/
    deepdata/       # Main server + embedded web UI + gRPC
    cli/            # Command-line client
    gentoken/       # JWT token generator
  api/proto/        # gRPC protobuf definitions
  client/           # Go client library
  sdk/python/       # Python client (pip install deepdata)
  internal/
    index/          # HNSW, IVF, DiskANN, flat, sparse — SIMD kernels
    collection/     # Collection manager, filtered search
    cluster/        # Election, replication, sharding, snapshots
    security/       # JWT, RBAC, TLS, encryption at rest, audit
    telemetry/      # Prometheus metrics, OpenTelemetry traces
    hybrid/         # Dense+sparse fusion (RRF)
    sparse/         # Inverted index, BM25
    graph/          # GraphRAG entity-aware index
    feedback/       # Relevance feedback loop
    storage/        # Persistence (gob, cowrie/SJSON)
    wal/            # Write-ahead log
    extraction/     # LLM-powered metadata extraction
    filter/         # Metadata filter engine
  benchmarks/       # Python benchmark suite
  desktop/          # Tauri v2 native wrapper
  docs/             # Guides, migration, benchmarks
```

## API at a Glance

| | Endpoint | What it does |
|---|---|---|
| **POST** | `/insert` | Add or upsert a document |
| **POST** | `/query` | Search with filters |
| **POST** | `/delete` | Delete by ID |
| **POST** | `/batch/insert` | Bulk insert (up to 10K) |
| **GET** | `/scroll` | Paginated iteration |
| **POST** | `/api/embed` | Get embedding for text |
| **POST** | `/v2/collections` | Create / list / delete collections |
| **POST** | `/v2/insert` | Insert into a V2 collection |
| **POST** | `/v2/search` | Search a V2 collection |
| **POST** | `/v2/recommend` | Recommend similar items |
| **POST** | `/v2/discover` | Discovery search |
| **GET** | `/health` | Stats + liveness |
| **GET** | `/metrics` | Prometheus endpoint |

gRPC on `:50051` (configurable via `GRPC_PORT`). Same operations, protobuf-efficient. Max message size 64MB.

Full API reference in [`internal/collection/API.md`](internal/collection/API.md).

## Modes

| Mode | Embedder | Cost |
|------|----------|------|
| `local` | Ollama | Free |
| `pro` | OpenAI `text-embedding-3-small` | ~$0.02/1M tokens |
| `hash` | Deterministic hash | Free (benchmarks/testing) |

Set with `--mode local` or `VECTORDB_MODE=local`.

## Configuration

Key environment variables:

| Variable | Default | What it controls |
|----------|---------|------------------|
| `VECTORDB_MODE` | `local` | Embedder backend (local/pro) |
| `API_TOKEN` | — | Simple bearer token auth |
| `JWT_SECRET` | — | JWT signing key |
| `GRPC_PORT` | `50051` | gRPC listener port |
| `MAX_COLLECTIONS` | `10000` | Collection count limit |
| `MAX_TENANTS` | `100000` | Tenant count limit |
| `TENANT_RPS` | `100` | Per-tenant rate limit (req/s) |
| `STORAGE_FORMAT` | `gob` | Serialization: gob, cowrie, cowrie-zstd |
| `ENCRYPTION_ENABLED` | `false` | Enable encryption at rest |
| `ENCRYPTION_ALGORITHM` | `aes-gcm` | aes-gcm or chacha20 |
| `AUDIT_LOG` | `false` | Enable audit logging |
| `TLS_ENABLED` | `false` | Enable TLS/mTLS |

## Clients

| Language | Install |
|----------|---------|
| Go | `import "github.com/phenomenon0/vectordb/client"` |
| Python | `pip install deepdata` |

## Desktop App

The `desktop/` directory wraps DeepData in a Tauri v2 native window with dynamic port allocation and native controls. Build with:

```bash
cd desktop && npm install && npm run tauri build
```

## Docs

- [Installation](docs/installation.md) — Docker, source, systemd
- [Cookbook](docs/cookbook.md) — RAG, hybrid search, multi-tenancy
- [Security](docs/security.md) — JWT, TLS/mTLS, RBAC, encryption
- [Distributed Architecture](docs/distributed-architecture.md) — Replication, sharding
- [Benchmarks](docs/benchmarks.md) — Latency and throughput numbers
- [Kubernetes](docs/kubernetes.md) — StatefulSet, Ingress, backups
- [Troubleshooting](docs/troubleshooting.md) — Common issues
- [Grafana Dashboard](docs/grafana/) — Prometheus dashboard + alerts
- [Contributing](docs/contributing.md)
- [Changelog](CHANGELOG.md)

## License

MIT
