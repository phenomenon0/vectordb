# Installation

## Requirements

- **Go 1.24+** (for building from source)
- **Linux, macOS, or Windows** (amd64/arm64)
- **RAM**: 256MB minimum, 1GB+ recommended for large datasets
- **Disk**: Depends on dataset size; ~100 bytes per vector (768d with HNSW)

## Quick Start (Docker)

```bash
# Clone and start
git clone https://github.com/phenomenon0/Agent-GO.git
cd Agent-GO/vectordb
docker compose up -d

# Verify
curl http://localhost:8080/health
```

## Build from Source

```bash
# Clone
git clone https://github.com/phenomenon0/Agent-GO.git
cd Agent-GO

# Build server
go build -o vectordb-server ./vectordb/

# Build CLI
go build -o vectordb-cli ./vectordb/cmd/vectordb-cli/

# Run
./vectordb-server
```

## Go Client Library

```bash
go get github.com/phenomenon0/Agent-GO/vectordb/client
```

```go
import "github.com/phenomenon0/Agent-GO/vectordb/client"

c := client.New("http://localhost:8080")
resp, err := c.Insert(ctx, client.InsertRequest{
    Doc: "Hello world",
    Collection: "docs",
})
```

## Python Client

```bash
pip install vectordb-client
```

```python
from vectordb import VectorDBClient

client = VectorDBClient("http://localhost:8080")
client.insert(doc="Hello world", collection="docs")
```

## TypeScript/JavaScript Client

```bash
npm install vectordb-js
```

```typescript
import { VectorDBClient } from 'vectordb-js';

const client = new VectorDBClient('http://localhost:8080');
await client.insert({ doc: 'Hello world', collection: 'docs' });
```

## Configuration

VectorDB is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP server port |
| `DATA_DIR` | `./data` | Data directory for persistence |
| `LOG_LEVEL` | `info` | Log level: debug, info, warn, error |
| `LOG_FORMAT` | `json` | Log format: json or text |
| `JWT_SECRET` | _(empty)_ | JWT signing secret (enables auth) |
| `JWT_REQUIRED` | `false` | Require JWT for all requests |
| `WAL_MAX_BYTES` | `5242880` | WAL file size limit (5MB) |
| `WAL_MAX_OPS` | `1000` | WAL ops before rotation |
| `VECTOR_CAPACITY` | `1000` | Initial vector store capacity |
| `USE_HASH_EMBEDDER` | `0` | Use hash embedder (low-memory) |
| `COMPACT_INTERVAL_MIN` | _(disabled)_ | Auto-compact interval in minutes |
| `SNAPSHOT_EXPORT_PATH` | _(empty)_ | Auto-export snapshot path |

## systemd Service

```ini
# /etc/systemd/system/vectordb.service
[Unit]
Description=VectorDB Server
After=network.target

[Service]
Type=simple
User=vectordb
ExecStart=/usr/local/bin/vectordb-server
Environment=PORT=8080
Environment=DATA_DIR=/var/lib/vectordb
Environment=LOG_LEVEL=info
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vectordb
```

## Verify Installation

```bash
# Health check
curl -s http://localhost:8080/health | jq .

# Insert a test document
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{"doc": "test document", "collection": "test"}'

# Query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'

# Using the CLI
vectordb-cli health
vectordb-cli insert --doc "test document" --collection test
vectordb-cli query --query "test" --top-k 3
```
