# DeepData Benchmark Configuration Guide

## Why These Settings Matter

When benchmarking DeepData, several environment variables critically affect performance and recall. Getting them wrong can produce misleading results.

## Critical Settings for Benchmarking

### Must-Set for Fair Benchmarks

| Variable | Benchmark Value | Default | Why It Matters |
|---|---|---|---|
| `VECTORDB_BASE_DIR` | `/tmp/deepdata-bench` | `~/.vectordb` | **Must use fresh dir** — old data pollutes results |
| `EMBEDDER_TYPE` | `hash` | auto (ollama>hash) | Avoids Ollama network overhead during insert |
| `API_RPS` | `100000` | `100` | Default rate limit (100/min) throttles benchmarks |
| `TENANT_RPS` | `100000` | `100` | Per-tenant rate limit also throttles |
| `TENANT_BURST` | `100000` | `100` | Burst capacity for rate limiter |
| `SCAN_THRESHOLD` | `0` | `500` | Forces index usage even for small collections |
| `PORT` | `8080` | `8080` | HTTP listen port |

### HNSW Index Tuning

| Variable | Benchmark Value | Default | Effect |
|---|---|---|---|
| `HNSW_M` | `16` | `16` | Graph connectivity — higher = better recall, more memory |
| `HNSW_EFSEARCH` | `128` | `64` | Search beam width — higher = better recall, slower |
| `HNSW_ML` | `0.25` | `0.25` | Level multiplier — controls graph depth |

### Recommended Benchmark Startup Command

```bash
# Fresh data, no rate limits, hash embedder, HNSW tuned
API_RPS=100000 TENANT_RPS=100000 TENANT_BURST=100000 \
HNSW_M=16 HNSW_EFSEARCH=128 SCAN_THRESHOLD=0 \
EMBEDDER_TYPE=hash \
VECTORDB_BASE_DIR=/tmp/deepdata-bench \
PORT=8080 \
./deepdata-server
```

## Common Pitfalls

### 1. Old Database Instance (Low Recall)
**Symptom**: Recall@10 < 0.7 when competitors show 0.9+
**Cause**: Server loaded existing snapshot from `~/.vectordb/local` with stale vectors
**Fix**: Set `VECTORDB_BASE_DIR` to a fresh temporary directory

### 2. Rate Limiting (429 Errors)
**Symptom**: `429 Too Many Requests` during search/insert
**Cause**: Default `API_RPS=100` limits to 100 requests/minute
**Fix**: Set `API_RPS=100000` (or higher)

### 3. Wrong Embedder (Ollama Overhead)
**Symptom**: Slow inserts, `EMBEDDER_TYPE` ignored
**Cause**: `EMBEDDER_TYPE=hash` wasn't handled — Ollama was auto-detected
**Fix**: Now fixed in code; set `EMBEDDER_TYPE=hash` for benchmarks

### 4. VECTORDB_DATA_DIR vs VECTORDB_BASE_DIR
**Symptom**: `VECTORDB_DATA_DIR=/tmp/x` but server still uses `~/.vectordb/local`
**Cause**: `VECTORDB_DATA_DIR` sets mode config but `GetDataDirectory()` reads from base dir
**Fix**: Use `VECTORDB_BASE_DIR` instead — it controls the actual path resolution

## Full Environment Variable Reference

### Core
| Variable | Default | Description |
|---|---|---|
| `VECTORDB_MODE` | `local` | Mode: `local` or `pro` |
| `VECTORDB_BASE_DIR` | `~/.vectordb` | Base directory for all data |
| `PORT` | `8080` | HTTP listen port |
| `EMBEDDER_TYPE` | auto | Embedder: `hash`, `ollama`, `openai`, `onnx` |
| `EMBED_DIM` | varies | Embedding dimension (384/768/1536) |
| `STORAGE_FORMAT` | `gob` | Serialization: `gob`, `cowrie`, `cowrie-zstd` |

### HNSW Index
| Variable | Default | Description |
|---|---|---|
| `HNSW_M` | `16` | Graph degree (connections per node) |
| `HNSW_ML` | `0.25` | Level multiplier |
| `HNSW_EFSEARCH` | `64` | Search beam width |
| `SCAN_THRESHOLD` | `500` | Min vectors before using HNSW (0=always use index) |

### Rate Limiting
| Variable | Default | Description |
|---|---|---|
| `API_RPS` | `100` | Global rate limit (requests/minute) |
| `TENANT_RPS` | `100` | Per-tenant rate limit |
| `TENANT_BURST` | `100` | Per-tenant burst capacity |

### Security
| Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | none | JWT signing key (enables auth) |
| `API_TOKEN` | none | Simple bearer token |
| `REQUIRE_AUTH` | `0` | Force authentication |
| `TRUST_PROXY` | `0` | Trust X-Forwarded-For |
| `CORS_ALLOWED_ORIGINS` | `*` | CORS allowlist |

### WAL & Persistence
| Variable | Default | Description |
|---|---|---|
| `WAL_MAX_BYTES` | `5242880` | Max WAL size before rotation (5MB) |
| `WAL_MAX_OPS` | `1000` | Max WAL operations before flush |
| `VECTOR_CAPACITY` | `1000000` | Initial vector store capacity |
| `COMPACT_INTERVAL_MIN` | `60` | Compaction interval (minutes) |
| `COMPACT_TOMBSTONE_THRESHOLD` | `10` | % tombstones triggering compaction |

### Embedder Providers
| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OPENAI_API_KEY` | none | OpenAI API key (required for pro mode) |

### Observability
| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `info` | Log level |
| `LOG_FORMAT` | `json` | Log format: `json` or `text` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | none | OpenTelemetry endpoint |

### Obsidian Sync
| Variable | Default | Description |
|---|---|---|
| `OBSIDIAN_VAULT` | none | Vault path (enables sync) |
| `OBSIDIAN_COLLECTION` | `obsidian` | Target collection |
| `OBSIDIAN_INTERVAL` | `5m` | Sync interval |
