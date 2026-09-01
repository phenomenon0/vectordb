# DeepData architecture

DeepData at HEAD is a single-node, tenant-aware vector database: six journaled mutations (`internal/collection/durable_store.go:18-23`), eleven gRPC RPCs on the `deepdata.v3.DeepData` service (`api/proto/deepdata/v3/deepdata.proto:240-252`) mirrored one-to-one by eleven HTTP routes under `/v3/tenants/` (`cmd/deepdata/collection_http.go:389`, `handleTenantRoutes`), and caller-supplied vectors only — the binary hard-wires `NewHashEmbedder(1)` at `cmd/deepdata/main.go:3275`, so server-side embedding is not live; it is a plan under CTL-02. This file is the map of the layers and of where each layer's truth lives. It carries no status: gate status lives only in `tasks/gates.json` and its render `docs/PRE_RELEASE_STATUS.md`.

## The tower (bottom → top; source: tasks/journal/2026-09-01-redesign-architects.md §5.1)

```
L6 Truth       tasks/gates.json ─render→ docs/PRE_RELEASE_STATUS.md (scripts/gates.py) · docs/ARCHITECTURE.md (machine-read non-goals) · docs/decisions/ · scripts/check_docs_contract.py R1–R12
L5 Agent       cmd/deepdata-mcp — at HEAD 5 untyped tools; planned 6 memory verbs whose inputSchemas ARE the L2 files + resources deepdata://contract, deepdata://status (CTL-03)
L4 Clients     sdk/python (pydantic models, CI-tested) · api/gen gRPC stubs (generated from the proto, diff-gated)
L3 Transports  HTTP /v3 + gRPC deepdata.v3 — thin; planned shared internal/apierror (CTL-01), cmd/deepdata/embed_text.go (CTL-02), serverRuntime for auth+limits (SYS-01)
L2 Contract    at HEAD: the proto + Go request structs + Canonical* limits; planned api/contract/v3/{schemas/*.json, operations.json, CONTRACT.md} as THE source (CTL-03), GET /v3/status as its runtime projection (CTL-04)
L1 Engine      internal/collection: Collection (hnsw|flat|inverted, fusion, filters, UsageTracker) · typed sentinels in limits.go · vectors only
L0 Durability  DurableStore: journal + snapshot v2 (class A) · usage.json sidecar (class B, planned CTL-05) · indexes (class C)
```

## Layers: truth, proof, docs, plan

| Layer | Truth file at HEAD | Proven by | Documented in | Planned change |
|---|---|---|---|---|
| L0 Durability | `internal/collection/journal.go` (checksummed frames, `:878-880`), `internal/collection/snapshot.go` (`writeUnifiedCollectionSnapshotV2` `:276`) | `internal/collection/journal_test.go`, `internal/collection/snapshot_stream_test.go`; fault latch `internal/collection/durable_store.go:245` → `cmd/deepdata/server.go:1566` | [bounded recovery journal](../tasks/journal/2026-08-28-bounded-recovery.md), [ADR 0002](decisions/0002-no-online-compactor.md) | usage.json sidecar written with each snapshot and loaded per class B (CTL-05) |
| L1 Engine | `internal/collection/durable_store.go` (single mutex boundary `:88`), `internal/collection/limits.go` (sentinels `:30-37`), `internal/collection/usage.go` | `internal/collection/durable_store_test.go`, `internal/collection/limits_test.go`, `internal/collection/agent_retrieval_test.go` | `internal/collection/API.md`, [ADR 0001](decisions/0001-narrow-rc-to-single-node-caller-vectors.md) | one IndexTypes vocabulary (SYS-02); typed not-found / exists sentinels beside `ErrInvalidSearchArgument` (CTL-01) |
| L2 Contract | `api/proto/deepdata/v3/deepdata.proto`, request structs + `Canonical*` consts in `internal/collection/limits.go` | `TestCanonicalGRPCDescriptorExcludesAdvancedMethods` (`cmd/deepdata/canonical_surface_test.go:516`), `scripts/check_proto_generated.sh` | `internal/collection/API.md` | api/contract/v3 JSON Schema as the single source (CTL-03); GET /v3/status as its runtime projection (CTL-04) |
| L3 Transports | `cmd/deepdata/collection_http.go`, `cmd/deepdata/collection_grpc.go`, wiring in `cmd/deepdata/main.go` (`canonicalOnly` `:3198`, gRPC registration `:3433`) | `cmd/deepdata/canonical_surface_test.go`, `cmd/deepdata/agent_retrieval_http_test.go`, `cmd/deepdata/agent_retrieval_grpc_test.go` | `internal/collection/API.md`, `docs/security.md` | internal/apierror envelope (CTL-01); embed_text.go texts path (CTL-02); serverRuntime extraction (SYS-01); routes subcommand (CTL-04, DOC-03) |
| L4 Clients | `sdk/python/deepdata/models.py` (`TenantIndexConfig` `:26`), `api/gen/deepdata/v3/deepdata_grpc.pb.go` | `sdk/python/tests/test_tenant_v3.py:128` (`.github/workflows/ci.yml:134`), `scripts/check_proto_generated.sh` (`.github/workflows/ci.yml:45`) | `sdk/python/README.md` | pydantic model_fields drift test against the L2 schemas (CTL-04) |
| L5 Agent | `cmd/deepdata-mcp/main.go` (5 tools, `:231-281`; tenant from `DEEPDATA_TENANT` `:102`) | `cmd/deepdata-mcp/main_test.go` — in no `.github/workflows/ci.yml` package list (CI-05) | [docs/mcp.md](mcp.md) | six memory verbs (deepdata_recall, remember, forget, get, collections, create_collection) on api/contract (CTL-03) |
| L6 Truth | `tasks/gates.json` | `scripts/gates.py` check, `scripts/check_docs_contract.py` — both run in `.github/workflows/ci.yml:48-51` | `tasks/PROTOCOL.md`, `docs/PRE_RELEASE_STATUS.md` | the docs-linter gate (DOC-01); generated route table (DOC-03) |

## Durability classes (journal §5.2)

| Class | Contents | On corruption | Where at HEAD |
|---|---|---|---|
| A canonical | six journal mutations, snapshot v2, schema (plus the embedding binding, CTL-02) | fail closed: `TestDurableStoreRejectsCorruptJournalFrame` (`internal/collection/durable_store_test.go:730`), `TestUnifiedCollectionSnapshotRejectsChecksumCorruption` (`internal/collection/snapshot_test.go:87`) | `internal/collection/journal.go`, `internal/collection/snapshot.go` |
| B accreted signal | UsageTracker frecency (later: feedback weights, MEM-02) | loud discard: error log + a status signal, the collection stays up (CTL-05) | in-memory only (`internal/collection/usage.go:30`); usage.json sidecar is CTL-05 |
| C derived | HNSW / flat / inverted indexes | rebuilt from A on load (`internal/collection/snapshot.go:76`; `TestUnifiedCollectionSnapshotV2RebuildsDenseAndSparseIndexes` `internal/collection/snapshot_stream_test.go:201`) | memory |

Why B ≠ A: a ranking hint that fails closed takes correct-by-similarity answers down with it.

## Invariant: one source, N projections, every projection checked in CI

At HEAD the checks are `scripts/check_proto_generated.sh` (proto → `api/gen`, `.github/workflows/ci.yml:45`), `scripts/check_version_contract.py` (`internal/releaseinfo/version.txt` → SDK, Helm, Dockerfile, `.github/workflows/ci.yml:47`), the gRPC method-set test at `cmd/deepdata/canonical_surface_test.go:516`, the HTTP allowlist test `TestCanonicalRCSurfaceRejectsUnsupportedHandlers` (`cmd/deepdata/canonical_surface_test.go:41`), and `sdk/python/tests/test_tenant_v3.py:128` (SDK limits vs server admission).
Planned drift tests, each direction against the L2 schemas (CTL-04): Go json tags and Canonical limits vs schema properties and maxima; operations.json grpc_rpc set vs proto methods; pydantic model_fields vs schema properties; MCP tool-name set vs the contract.
`scripts/check_docs_contract.py` (DOC-01) and `scripts/gates.py` check run in CI after the version-parity step (`.github/workflows/ci.yml:48-51`); nothing is hand-maintained in two places once every projection above has its check.

Out of scope for the RC (each line is label | regex, machine-read by `scripts/check_docs_contract.py` rules R8 and R11):
<!-- non-goals -->
DiskANN | \bdiskann\b
IVF | \bivf(?:[_-]?(?:flat|pq|hnsw|sq))?\b|\binverted[- ]file\b
PQ/binary/scalar quantization | \b(?:product|binary|scalar|int8|uint8|fp16|float16)[- ]quantiz\w*|\bPQ\b|\bquantiz(?:ation|ed|er|e)\b
replication/cluster/shard/failover | \b(?:replication|clustering|shards?|sharding|sharded|failover|follower restore|multi[- ]node|high[- ]availability|distributed mode)\b|(?<!one )(?<!single )(?<!\d )\breplicas?\b|(?<!kubernetes )(?<!kind )(?<!k8s )\bclusters?\b(?![- ]level)
GraphRAG/graph reranking | \bgraph[- ]?rag\b|\bgraph[- ](?:rerank\w*|boost\w*)\b|\bknowledge[- ]graph\b|\bpagerank\b
Tauri/desktop app | \btauri\b|\bdesktop (?:app|application|installer|package|build|client)s?\b
web UI | \bweb[- ]?ui\b|\bbrowser[- ]based (?:ui|dashboard|console)\b|\balpine\.?js\b|\bvite\b
CUDA/GPU | \bcuda\b|\bgpus?\b|\bgpu-accelerated\b|\bnvidia\b
built-in TLS/encryption-at-rest | \bencryption[- ]at[- ]rest\b|\b(?:built[- ]in|native|server[- ]side)\s+(?:m?tls|encryption)\b|\btls (?:cert\w*|key|listener|config\w*)\b
server-managed embeddings (until CTL-02) | \bserver[- ]managed embedd\w*|\bembedding providers?\b|\b(?:ollama|openai|gemini|voyage|jina|cohere|mistral)[- ]embedd\w*|\bONNX\b|\bbge[- ]small\b|/api/embed\b
<!-- /non-goals -->
