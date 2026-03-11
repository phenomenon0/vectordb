# Installation

## Requirements

- **Go 1.24+** (for building from source)
- **Linux, macOS, or Windows** (amd64/arm64)
- **RAM**: 256MB minimum, 1GB+ recommended for large datasets
- **Disk**: Depends on dataset size; ~100 bytes per vector (768d with HNSW)

## Quick Start (Docker)

```bash
git clone https://github.com/phenomenon0/vectordb.git
cd vectordb
docker compose up -d

# Verify
curl http://localhost:8080/health
```

## Build from Source

```bash
git clone https://github.com/phenomenon0/vectordb.git
cd vectordb

# Build server
go build -o deepdata ./cmd/deepdata

# Build CLI
go build -o deepdata-cli ./cmd/cli

# Run
./deepdata serve
```

## Pre-built Binaries

Download from [GitHub Releases](https://github.com/phenomenon0/vectordb/releases/latest). No Go toolchain required:

```bash
chmod +x deepdata-linux-amd64
./deepdata-linux-amd64 serve
```

## Go Client Library

```bash
go get github.com/phenomenon0/vectordb/client
```

```go
import "github.com/phenomenon0/vectordb/client"

c := client.New("http://localhost:8080")
resp, err := c.Insert(ctx, client.InsertRequest{
    Doc:        "Hello world",
    Collection: "docs",
})
```

## Python Client

```bash
pip install deepdata
```

```python
from deepdata import DeepDataClient

client = DeepDataClient("http://localhost:8080")
client.insert("Hello world", collection="docs")
results = client.search("Hello", top_k=5, collection="docs")
```

Async variant:

```python
from deepdata import AsyncDeepDataClient

async with AsyncDeepDataClient("http://localhost:8080") as client:
    results = await client.search("Hello", top_k=5)
```

## Configuration

DeepData is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP server port |
| `GRPC_PORT` | `50051` | gRPC server port |
| `DATA_DIR` | `./data` | Data directory for persistence |
| `LOG_LEVEL` | `info` | Log level: debug, info, warn, error |
| `LOG_FORMAT` | `json` | Log format: json or text |
| `VECTORDB_MODE` | `local` | Embedder mode: local (Ollama), pro (OpenAI) |
| `JWT_SECRET` | _(empty)_ | JWT signing secret (enables auth) |
| `JWT_REQUIRED` | `false` | Require JWT for all requests |
| `API_TOKEN` | _(empty)_ | Simple bearer token auth |
| `MAX_COLLECTIONS` | `10000` | Maximum number of collections |
| `MAX_TENANTS` | `100000` | Maximum number of tenants |
| `TENANT_RPS` | `100` | Per-tenant rate limit (requests/sec) |
| `TENANT_BURST` | `100` | Per-tenant burst allowance |
| `STORAGE_FORMAT` | `gob` | Serialization: gob, cowrie, cowrie-zstd |
| `WAL_MAX_BYTES` | `5242880` | WAL file size limit (5MB) |
| `WAL_MAX_OPS` | `1000` | WAL ops before rotation |
| `VECTOR_CAPACITY` | `1000` | Initial vector store capacity |
| `USE_HASH_EMBEDDER` | `0` | Use hash embedder (low-memory/benchmarks) |
| `COMPACT_INTERVAL_MIN` | _(disabled)_ | Auto-compact interval in minutes |
| `SNAPSHOT_EXPORT_PATH` | _(empty)_ | Auto-export snapshot path |
| `ENCRYPTION_ENABLED` | `false` | Enable encryption at rest |
| `ENCRYPTION_PASSPHRASE` | _(empty)_ | Encryption passphrase |
| `ENCRYPTION_ALGORITHM` | `aes-gcm` | aes-gcm or chacha20 |
| `AUDIT_LOG` | `false` | Enable audit logging |
| `AUDIT_LOG_FILE` | _(empty)_ | Audit log file path |
| `TLS_ENABLED` | `false` | Enable TLS |
| `TLS_CERT_FILE` | _(empty)_ | TLS certificate file |
| `TLS_KEY_FILE` | _(empty)_ | TLS private key file |
| `TLS_CLIENT_AUTH` | _(empty)_ | mTLS client auth mode (require) |
| `TLS_MIN_VERSION` | `1.2` | Minimum TLS version |

## systemd Service

```ini
# /etc/systemd/system/deepdata.service
[Unit]
Description=DeepData Vector Database
After=network.target

[Service]
Type=simple
User=deepdata
ExecStart=/usr/local/bin/deepdata serve
Environment=PORT=8080
Environment=GRPC_PORT=50051
Environment=DATA_DIR=/var/lib/deepdata
Environment=LOG_LEVEL=info
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now deepdata
```

## Verify Installation

```bash
# Health check
curl -s http://localhost:8080/health | jq .

# Liveness probe
curl -s http://localhost:8080/healthz

# Readiness probe
curl -s http://localhost:8080/readyz

# Insert a test document
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{"doc": "test document", "collection": "test"}'

# Query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'

# Using the CLI
deepdata-cli health
deepdata-cli insert --doc "test document" --collection test
deepdata-cli query --query "test" --top-k 3
```
