# Changelog

All notable changes to VectorDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **CLI tool** (`vectordb-cli`): health, insert, query, delete, collections, stats, import/export (JSONL), compact, gentoken
- **Go client improvements**: `Compact()`, `ListCollections()`, `Scroll()` methods, typed error hierarchy (`APIError`, sentinel errors), exponential backoff retry with jitter
- **Scroll endpoint** (`GET/POST /scroll`): paginated document iteration with tenant/collection filtering
- **Grafana dashboard**: 16-panel template with query latency, throughput, error rates, shard health, and alerting rules
- **Cowrie codec support**: `Accept: application/cowrie` for ~48% smaller float32 array responses
- **Docker deployment**: multi-stage Dockerfile + docker-compose.yml
- **Documentation**: Installation guide, Troubleshooting (20+ issues), Cookbook (6 recipes), Security guide, Kubernetes deployment, Benchmarks, migration guides (ChromaDB, Qdrant, Pinecone), "Why VectorDB" positioning page

### Fixed
- **Sparse search scoring**: replaced placeholder `0.5` score with actual cosine similarity computation
- **SIMD safety**: added `Safe` variants of distance functions that return errors instead of panicking on dimension mismatch
- **NaN/Inf rejection**: `validateVector()` on insert path prevents corrupt vectors from entering the index
- **PQ serialization**: implemented `Export`/`Import` for PQ4 and PQ-ADC indices (was returning "not implemented")
- **Missing strconv import**: scroll endpoint wouldn't compile

### Changed
- `HTTPError` is now a type alias for `APIError` (backward compatible)
- Client retries enabled by default (3 retries, exponential backoff, jitter)
- Distributed architecture docs marked as experimental with warning banners

## [0.1.0] - 2024-12-17

### Added

#### Core Features
- **HNSW Index**: High-performance approximate nearest neighbor search with O(log n) complexity
- **IVF Index**: Clustering-based search with configurable nlist/nprobe parameters
- **DiskANN Index**: Memory-mapped disk-backed index for large-scale datasets
- **Flat Index**: Brute-force exact search for small datasets or ground truth

#### Vector Types
- Dense vectors (float32)
- Sparse vectors (BM25/SPLADE compatible)
- Binary vectors (planned)

#### Quantization
- Float16 quantization (~50% memory reduction)
- Uint8 quantization (~75% memory reduction)  
- Product Quantization (PQ8) for extreme compression
- 4-bit Product Quantization (PQ4) for maximum compression

#### Hybrid Search
- Dense + sparse vector fusion
- Reciprocal Rank Fusion (RRF) strategy
- Weighted fusion with configurable alpha
- Linear combination fusion

#### Multi-tenancy
- Tenant isolation with separate namespaces
- Per-tenant quotas and rate limits
- Collection-level ACLs

#### Persistence
- Write-Ahead Logging (WAL) with CRC checksums
- Automatic snapshots with configurable thresholds
- SJSON binary format (~48% smaller than JSON)
- Gob format for backward compatibility

#### API
- HTTP REST API with JSON/SJSON support
- Batch insert (up to 10K documents)
- Filtered search with metadata predicates
- Prometheus metrics endpoint
- Health and readiness probes (Kubernetes compatible)

#### Security
- JWT authentication
- API key rotation
- Role-Based Access Control (RBAC)
- TLS support
- Audit logging

#### Embeddings
- Built-in hash embedder for testing
- OpenAI API integration
- Ollama local model support
- ONNX runtime support (optional)

### Experimental
- **Distributed Mode**: Sharding, replication, and query routing
  - WARNING: Not recommended for production use
  - Incomplete quorum safety checks
  - Limited snapshot sync for replicas

### Security Fixes
- Removed hardcoded default JWT secret (now fails fast if not configured)
- Fixed potential panic in MAC address generation
- Replaced production panics with proper error handling/logging

### Bug Fixes
- Fixed audit log rotation not triggering on sync writes
- Fixed HNSW race condition in lock yielding during bulk insert
- Fixed collection cleanup memory leak on delete
- Fixed response encoding errors being silently ignored

### Performance
- Reduced test suite runtime with `-short` flag support
- Optimized k-means training dataset sizes for CI

### Developer Experience
- Added comprehensive Makefile with common targets
- Enhanced CI workflow with race detection
- Separate VectorDB test job in CI pipeline

## [0.0.1] - 2024-11-01

### Added
- Initial development release
- Basic HNSW implementation
- Simple HTTP API
- In-memory storage only

---

## Versioning

VectorDB uses semantic versioning:
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Upgrade Guide

### 0.0.x to 0.1.0

1. **JWT Configuration**: If using JWT authentication, you MUST now set the `JWT_SECRET` environment variable. The default secret has been removed for security.

2. **API Response Handling**: Response encoding errors are now logged. Check your logs for any `failed to encode response` warnings.

3. **Distributed Mode**: If using distributed mode, note that it is now marked experimental. Consider using single-node mode for production until v1.0.

4. **Collection Deletion**: Collections now properly clean up resources on deletion. This is automatic and requires no changes.
