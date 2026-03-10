# Benchmark Implementation Log

Issues discovered and fixes applied during the real-world VDB benchmark run.

## DeepData Issues

### 1. v2 API Schema Uses Go Struct Field Names (No JSON Tags)
- **Problem**: `POST /v2/collections` expects `{"Name":"...", "Fields":[{"Name":"embedding","Type":0,"Dim":1536}]}` — PascalCase Go field names, not lowercase JSON conventions
- **Impact**: Collection creation fails with unhelpful errors like "must have at least one vector field"
- **Fix**: Use Go naming in HTTP payloads
- **Recommendation**: Add `json:"name"` struct tags to `CollectionSchema` and `VectorField` in `internal/collection/types.go` for API consistency

### 2. VectorType is an int enum, not a string
- **Problem**: API expects `"Type": 0` for dense vectors, not `"Type": "dense"` or `"Type": "vector"`
- **Impact**: Confusing for SDK/API users who don't know the int mappings
- **Recommendation**: Accept both string ("dense"/"sparse") and int (0/1) in the HTTP handler via custom unmarshal, or at minimum document the enum values in API docs

### 3. Rate Limiting on v2 Insert (429 Too Many Requests)
- **Problem**: Default `API_RPS=100` rate limiter throttles batch inserts hard — each insert is a separate HTTP call
- **Impact**: Benchmark insert throughput capped by rate limiter, not actual DB performance. Real throughput with `API_RPS=10000` was 689 docs/sec
- **Fix**: Set `API_RPS=10000` env var; added retry-with-backoff in adapter
- **Recommendation**: Add a `/v2/batch_insert` endpoint that accepts multiple documents in one request to avoid per-doc rate limiting

### 4. v2 Insert Returns uint64 IDs, Not Client-Provided String IDs
- **Problem**: v2 insert auto-assigns uint64 IDs; no way to pass a custom string ID
- **Impact**: Must maintain an ID mapping table in the adapter to compute recall correctly
- **Recommendation**: Support client-provided string IDs in v2 (like the v1 `/insert` endpoint does)

### 5. v2 Search Response Uses Go Struct Field Names
- **Problem**: Response contains `{"ID": 1, "Metadata": {...}}` not `{"id": 1, "metadata": {...}}`
- **Impact**: Same PascalCase issue as collection creation
- **Recommendation**: Add json tags to `Document` struct

### 6. No v2 Pre-computed Vector Search
- **Problem**: v1 `/query` accepts text and re-embeds it. v2 `/v2/search` accepts raw vectors via `queries` field.
- **Status**: v2 search works correctly for vector-in queries — this is good.

### 7. Search Latency ~22ms vs Competitors ~1.5ms
- **Problem**: DeepData search p50=21.6ms while Weaviate=1.5ms, Qdrant=1.8ms, ChromaDB=1.7ms
- **Root cause**: Likely HTTP overhead in Go's net/http + JSON encoding/decoding + HNSW traversal overhead
- **Note**: DeepData is running as a local process, competitors in containers. The gap suggests room for optimization in the HTTP/search path
- **Recommendation**: Profile search hot path, consider connection pooling, binary protocol, or batch query support

### 8. Recall@100 = 0.96 (vs competitors at 0.99-1.0)
- **Problem**: DeepData's recall@100 is slightly lower than competitors
- **Root cause**: May be HNSW parameter differences or the ef_search not being applied via v2 API
- **Recommendation**: Verify ef_search parameter is being honored in v2 search path

## Competitor Observations

### Weaviate
- Solid recall (1.0 at all k) and best QPS (1809)
- Hybrid search recall@10 = 0.82 — keyword component hurts when searching code (code != natural language)
- Insert: 44.8 docs/sec — slow due to per-doc gRPC overhead in batch
- `vectorizer_config` is deprecated in weaviate-client v4; use `vector_config` instead

### Milvus
- Perfect recall (1.0 at all k), consistent with HNSW
- Very slow insert: 4.8 docs/sec — the 3-service stack (etcd+minio+milvus) adds massive overhead
- Search latency p50=4.5ms, p99=729ms (cold start spike on first query)
- Hybrid falls back to dense-only in our adapter (proper sparse needs separate BM25 setup)
- 3-service deployment (etcd + minio + milvus) is a major ops burden

### Qdrant
- Fastest insert: 2314 docs/sec and best search latency (p50=1.8ms)
- Perfect recall (1.0 at all k) with matching client version
- **Critical bug found**: qdrant-client v1.17 `query_points()` silently returns wrong results against v1.12 server — always pin client to match server version
- Another bug: using sequential per-batch int IDs caused data overwrite — must use global counter for point IDs

### ChromaDB
- Good all-around: 464 docs/sec insert, recall@10=0.998, p50=1.7ms
- chromadb server 0.6.x is incompatible with Python 3.14 (pydantic v1). Must use server 1.0+ with client 1.5+
- No hybrid search support
- Simplest API of all competitors

## Podman/Container Notes

- Podman requires full registry paths (`docker.io/...`) for image names — short names fail with "short-name resolution enforced"
- Port 8000 often already in use — ChromaDB remapped to 8010
- `docker compose` not available — use `podman-compose` instead
- Milvus depends on etcd + minio, takes ~30s to become healthy

## General Architectural Notes

- DeepData's single-binary deployment is a significant advantage vs Milvus (3 services)
- Graph-boosted search is genuinely unique — no competitor has equivalent
- The v2 API is more capable but needs API polish (json tags, string enums, batch endpoints)
- For fair benchmarking, pre-computed vectors eliminate embedder variance across VDBs
- DeepData's insert throughput (689/sec) is competitive but search latency (22ms) needs optimization
- All competitors achieve ~1-5ms search latency; DeepData at 22ms is 10-15x slower — this is the main gap to close
