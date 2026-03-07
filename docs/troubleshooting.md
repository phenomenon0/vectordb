# Troubleshooting

## Server Won't Start

### Port already in use
```
listen tcp :8080: bind: address already in use
```
**Fix**: Another process is using port 8080. Either stop it or change the port:
```bash
PORT=8081 ./vectordb-server
```

### Permission denied on data directory
```
failed to create data directory: permission denied
```
**Fix**: Ensure the user has write access:
```bash
sudo mkdir -p /var/lib/vectordb
sudo chown $USER /var/lib/vectordb
DATA_DIR=/var/lib/vectordb ./vectordb-server
```

### Out of memory at startup
**Fix**: Reduce initial capacity and use hash embedder for minimal memory:
```bash
VECTOR_CAPACITY=100 USE_HASH_EMBEDDER=1 ./vectordb-server
```

### ONNX embedder fails to load
```
failed to initialize embedder: onnx session error
```
**Fix**: Either install the ONNX model or fall back to hash embedder:
```bash
# Option 1: Download model
wget https://huggingface.co/BAAI/bge-small-en/resolve/main/model.onnx -O models/bge-small-en.onnx

# Option 2: Use hash embedder
USE_HASH_EMBEDDER=1 ./vectordb-server
```

---

## Connection Issues

### Client can't connect
```
vectordb: connection failed: dial tcp 127.0.0.1:8080: connect: connection refused
```
**Fix**: Verify the server is running and the URL is correct:
```bash
curl http://localhost:8080/health
# If using a different port:
VECTORDB_URL=http://localhost:8081 vectordb-cli health
```

### Request timeout
```
vectordb: request timed out
```
**Fix**: The server is overloaded or the query is too complex:
- Reduce `top_k` value
- Use `mode: "ann"` instead of `"scan"`
- Increase client timeout: `client.New(url, client.WithTimeout(60*time.Second))`

### 401 Unauthorized
**Fix**: Authentication is enabled. Set your token:
```bash
# CLI
VECTORDB_TOKEN=your-token vectordb-cli health

# Go client
c := client.New(url, client.WithToken("your-token"))
```

### 429 Too Many Requests
**Fix**: You're being rate-limited. The client auto-retries with backoff. To increase limits:
```bash
TENANT_RPS=500 ./vectordb-server
```

---

## Query Issues

### Empty results
- Verify documents exist: `vectordb-cli stats`
- Check collection name matches: queries default to the `"default"` collection
- If using metadata filters, verify metadata was set during insert
- For ANN mode, the HNSW index needs at least a few vectors to work

### Low quality results / wrong documents returned
- **Hash embedder**: Only useful for testing. Use ONNX or external embedder for semantic search
- **Dimension mismatch**: Ensure query and stored vectors use the same embedding model
- **Try scan mode**: `"mode": "scan"` does exact search (slower but guaranteed correct)
- **Increase ef_search**: `"ef_search": 200` improves recall at cost of latency

### Scores are all 0 or 1
- Scores depend on the distance metric. Cosine similarity returns 0-1
- Hash embedder produces random-ish embeddings — scores won't be meaningful

---

## Data Issues

### WAL corruption
```
failed to replay WAL: unexpected EOF
```
**Fix**: The WAL file was truncated (crash, disk full). The server will skip corrupted entries and continue. To force a clean start:
```bash
# Backup existing data
cp -r data/ data-backup/
# Remove WAL (will lose un-snapshotted changes)
rm data/*.wal
```

### Snapshot won't load
```
failed to load snapshot: gob decode error
```
**Fix**: The snapshot format may have changed between versions. Export data and re-import:
```bash
# If the old binary can still start:
vectordb-cli export --output backup.jsonl
# Start with new binary
vectordb-cli import --file backup.jsonl
```

### Disk full
**Fix**: Enable auto-compaction to reclaim deleted vector space:
```bash
COMPACT_INTERVAL_MIN=60 ./vectordb-server
```
Or trigger manual compaction:
```bash
curl -X POST http://localhost:8080/compact
```

---

## Performance Issues

### Slow inserts
- Use **batch insert** (`/batch_insert`) instead of single inserts — 10-100x faster
- CLI: `vectordb-cli import --file data.jsonl --batch-size 500`
- Reduce WAL rotation frequency: `WAL_MAX_OPS=5000`

### Slow queries
- Use **ANN mode** (default) instead of scan mode
- Reduce `top_k` — smaller k = faster
- Increase HNSW `ef_construction` for better index quality (requires rebuild)
- Enable query caching (enabled by default)

### High memory usage
- **Expected**: HNSW index keeps all vectors in RAM
- **Estimate**: ~(dim × 4 + 200) bytes per vector. For 768d: ~3.3KB per vector
- **1M vectors at 768d**: ~3.3GB RAM
- **Mitigation**: Use PQ4 quantization for 32x compression, or DiskANN for disk-backed index
- Reduce initial capacity: `VECTOR_CAPACITY=1000`

### CPU spikes during compaction
Compaction rebuilds the HNSW index. This is CPU-intensive but temporary:
- Schedule compaction during low-traffic periods
- Set `COMPACT_INTERVAL_MIN` to a longer interval

---

## Docker Issues

### Container exits immediately
Check logs:
```bash
docker compose logs vectordb
```
Common causes: port conflict, missing volume mount, environment misconfiguration.

### Data lost after container restart
Ensure a volume is mounted:
```yaml
volumes:
  - vectordb-data:/data
```

### Health check failing
The health check hits `/health`. If the server takes time to start:
```yaml
healthcheck:
  start_period: 30s  # give more time for large datasets
```

---

## Authentication Issues

### Generate a test token
```bash
# Set JWT_SECRET on the server
JWT_SECRET=my-secret ./vectordb-server

# Generate token
curl -X POST http://localhost:8080/admin/tokens \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "test", "permissions": ["read", "write"]}'
```

### Token expired
Generate a new token with longer expiration. Default is 24h.

### Collection access denied
Your token may not have access to the requested collection. Check token claims:
```bash
# Decode JWT (base64)
echo "YOUR_TOKEN" | cut -d. -f2 | base64 -d | jq .
```

---

## Build Issues

### `go build` fails with missing dependencies
```bash
go mod download
go mod tidy
```

### ONNX build tag issues
ONNX embedder requires the `onnx` build tag:
```bash
go build -tags onnx -o vectordb-server ./vectordb/
```
Without it, the hash embedder is used as fallback (no error, just different embedding quality).

### CGO issues
VectorDB can be built without CGO for maximum portability:
```bash
CGO_ENABLED=0 go build -o vectordb-server ./vectordb/
```
