# DeepData

**High-performance vector database. Single binary. Zero dependencies.**

[![Go](https://img.shields.io/badge/Go-1.24+-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/phenomenon0/vectordb)](https://github.com/phenomenon0/vectordb/releases/latest)

DeepData is a vector database written in Go that ships as a single binary with an embedded web UI. HNSW, IVF, DiskANN indexes. Hybrid dense+sparse search. SIMD-accelerated distance functions. Runs locally, scales to clusters.

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

Metadata filtering with `meta` (AND), `meta_any` (OR), `meta_not` (NOT).

## Why DeepData

- **Single binary** — no runtime deps, no JVM, no Python. `go build` and ship
- **Hybrid search** — dense + sparse fusion via RRF or weighted scoring
- **4 index types** — HNSW, IVF, DiskANN, flat scan. Pick your tradeoff
- **Quantization** — FP16, Uint8, PQ8, PQ4 to trade memory for speed
- **SIMD + GPU** — AVX2 distance kernels, optional CUDA acceleration
- **Multi-tenant** — collection isolation, per-tenant quotas, RBAC
- **Embedders built in** — Ollama, OpenAI, ONNX runtime, hot-swappable
- **Production ready** — WAL with CRC checksums, snapshots, Prometheus metrics, JWT auth, TLS/mTLS

## Architecture

```
DeepData/
  cmd/
    deepdata/       # Main server + embedded web UI
    cli/            # Command-line client
    gentoken/       # JWT token generator
  internal/
    index/          # HNSW, IVF, DiskANN, flat — SIMD kernels
    collection/     # Collection manager, filtered search
    cluster/        # Distributed: election, replication, sharding
    security/       # JWT auth, RBAC, TLS, encryption, audit
    hybrid/         # Dense+sparse fusion
    sparse/         # Inverted index, BM25
    storage/        # Persistence (gob, cowrie/SJSON)
    wal/            # Write-ahead log
    extraction/     # LLM-powered metadata extraction
    feedback/       # Relevance feedback loop
    filter/         # Metadata filter engine
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
| **GET** | `/health` | Stats + liveness |
| **GET** | `/metrics` | Prometheus endpoint |

Full API reference in [`internal/collection/API.md`](internal/collection/API.md).

## Modes

| Mode | Embedder | Cost |
|------|----------|------|
| `local` | Ollama / ONNX | Free |
| `pro` | OpenAI `text-embedding-3-small` | ~$0.02/1M tokens |

Set with `--mode local` or `VECTORDB_MODE=local`.

## Desktop App

The `desktop/` directory wraps DeepData in a Tauri v2 native window with dynamic port allocation and native controls. Build with:

```bash
cd desktop && npm install && npm run tauri build
```

## Docs

- [Installation](docs/installation.md) — Docker, source, systemd
- [Cookbook](docs/cookbook.md) — RAG, hybrid search, multi-tenancy
- [Security](docs/security.md) — JWT, TLS/mTLS, RBAC, encryption
- [Benchmarks](docs/benchmarks.md) — Latency and throughput numbers
- [Kubernetes](docs/kubernetes.md) — StatefulSet, Ingress, backups
- [Troubleshooting](docs/troubleshooting.md) — Common issues
- [Grafana Dashboard](docs/grafana/) — Prometheus dashboard + alerts
- [Contributing](docs/contributing.md)
- [Changelog](CHANGELOG.md)

## License

MIT
