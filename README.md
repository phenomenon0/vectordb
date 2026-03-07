# DeepData

High-performance vector database written in Go. Single binary, embedded web UI, optional desktop app.

## Features

- **Multiple index types**: HNSW (default), IVF, DiskANN, flat scan
- **Hybrid search**: dense + sparse vector fusion (RRF, weighted, linear)
- **Quantization**: FP16, Uint8, PQ8, PQ4 — trade memory for speed
- **Multi-tenancy**: tenant isolation, per-tenant quotas, collection-level ACLs
- **Persistence**: WAL with CRC checksums, automatic snapshots, gob/SJSON formats
- **Security**: JWT auth, API key rotation, RBAC, TLS/mTLS, audit logging
- **Embedders**: Ollama (local), OpenAI, ONNX runtime, hash (testing) — hot-swappable at runtime
- **Observability**: Prometheus metrics, Grafana dashboard, structured logging
- **Desktop app**: Tauri wrapper with native window controls (Linux/macOS/Windows)

## Quick Start

```bash
# Run with Go
go run . serve --port 8080

# Or with Docker
docker compose up

# Or build the binary
go build -o deepdata .
./deepdata serve --port 8080
```

The embedded web UI is served at the root (`http://localhost:8080`).

## Modes

| Mode | Embedder | Dimension | Cost |
|------|----------|-----------|------|
| **local** | Ollama `nomic-embed-text` / ONNX `bge-small` | 768 / 384 | Free |
| **pro** | OpenAI `text-embedding-3-small` | 1536 | ~$0.02/1M tokens |

Set with `--mode local` or `VECTORDB_MODE=local`.

## API

Server listens on `:8080` by default. All endpoints accept/return JSON.

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/insert` | Add or upsert a document |
| `POST` | `/query` | Search with optional metadata filters |
| `POST` | `/delete` | Delete by ID (tombstone) |
| `GET/POST` | `/scroll` | Paginated document iteration |
| `POST` | `/batch/insert` | Bulk insert (up to 10K docs) |

### Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Stats: total, active, deleted vectors |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/integrity` | Checksum validation |
| `POST` | `/compact` | Rebuild index, drop tombstones |
| `GET` | `/export` | Download snapshot |
| `POST` | `/import` | Load snapshot |
| `GET` | `/metrics` | Prometheus metrics |

### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/api/config/embedder` | View or hot-swap the embedder |
| `GET/POST` | `/api/config/keys` | Manage LLM API keys (in-memory) |
| `GET` | `/api/mode` | Current mode info |
| `GET` | `/api/costs` | Cost tracking (pro mode) |

### Insert

```bash
curl -X POST http://localhost:8080/insert \
  -d '{"doc": "vector databases use ANN for fast retrieval", "id": "doc-1", "meta": {"topic": "databases"}}'
```

### Query

```bash
curl -X POST http://localhost:8080/query \
  -d '{"query": "how do vector databases work?", "top_k": 5, "mode": "ann", "include_meta": true}'
```

Supports metadata filtering: `meta` (AND), `meta_any` (OR), `meta_not` (NOT).

## CLI Flags

```
deepdata serve [flags]

  --port            HTTP port (env: PORT, default: 8080)
  --mode            Engine mode: local or pro (env: VECTORDB_MODE)
  --data-dir        Data directory (env: VECTORDB_DATA_DIR)
  --dimension       Embedding dimension (env: EMBED_DIM)
  --embedder        Embedder: ollama, openai, hash (env: EMBEDDER_TYPE)
  --embedder-model  Model name (env: OLLAMA_EMBED_MODEL)
  --embedder-url    Embedder URL (env: OLLAMA_URL)
```

## Desktop App

The `desktop/` directory contains a Tauri v2 wrapper that launches the Go server as a sidecar and loads the web UI in a native window.

```bash
cd desktop
npm install
npm run tauri build
```

Features: dynamic port allocation, draggable titlebar, native minimize/maximize/close controls.

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](INSTALLATION.md) | Docker, source build, SDKs, systemd |
| [Troubleshooting](TROUBLESHOOTING.md) | 20+ common issues with solutions |
| [Cookbook](docs/cookbook.md) | RAG, hybrid search, multi-tenancy, tuning |
| [Security](docs/security.md) | JWT, TLS/mTLS, RBAC, encryption, audit |
| [Kubernetes](docs/kubernetes.md) | StatefulSet, Ingress, backup CronJob |
| [Benchmarks](docs/benchmarks.md) | Latency, throughput, scalability |
| [Grafana Dashboard](grafana/) | Prometheus dashboard + alerting rules |
| [Changelog](CHANGELOG.md) | Release history |

### Migration Guides

- [From ChromaDB](docs/migration-from-chroma.md)
- [From Qdrant](docs/migration-from-qdrant.md)
- [From Pinecone](docs/migration-from-pinecone.md)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP listen port |
| `VECTORDB_MODE` | `local` | `local` or `pro` |
| `VECTORDB_DATA_DIR` | `./data` | Data storage directory |
| `EMBEDDER_TYPE` | `ollama` | `ollama`, `openai`, `onnx`, `hash` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OPENAI_API_KEY` | — | OpenAI API key (pro mode) |
| `JWT_SECRET` | — | JWT signing secret (required if using auth) |
| `EMBED_DIM` | auto | Override embedding dimension |
| `WAL_MAX_BYTES` | `5242880` | WAL size before rotation |
| `COMPACT_INTERVAL_MIN` | — | Auto-compact interval (minutes) |

## License

MIT
