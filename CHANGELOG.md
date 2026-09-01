# DeepData changelog

This changelog describes the headless, persistent, single-node DeepData
server. The current candidate is `0.2.0-rc.1` (`internal/releaseinfo/version.txt`),
licensed under Apache-2.0 (root `LICENSE`, b1f28be). Push, tag, and package
publication are tracked by gates PUB-01, PUB-02, and PUB-03 in `tasks/gates.json`.

## 0.2.0-rc.1 — unreleased single-node release candidate

Entries with a commit sha landed after the upsert/get-document feature at
a99fe53 (2026-08-07); entries without one describe the candidate as first
assembled.

### Added

- Upsert and get-document by caller-supplied ID across the durable engine, V3
  gRPC (`Upsert`, `GetDoc`), V3 HTTP (`PUT`/`GET`
  `/v3/tenants/{tenant_id}/collections/{name}/docs/{doc_id}`), and the Python
  client (a99fe53).
- Agent retrieval contract: `score_floor`, `fallback`, and `usage_boost`
  request fields with `weak_match`, `best_score`, and `fell_back_to` response
  fields on HTTP, gRPC, and the Python client, plus the `cmd/deepdata-mcp`
  stdio server for MCP hosts (bde4f94).
- Benchmark harness: `benchmarks/ddload`, a stdlib-only Go load client, and the
  V3 port of `benchmarks/recall_test.py` (50fe829); `benchmarks/ddload-qdrant`,
  an equal-effort comparison client whose results are in `docs/BENCHMARKS.md`
  (6cc5ef0).
- `scripts/backup_restore_drill.sh`: a bare-process offline backup/restore
  drill with strict post-restore assertions (bc1fa27).
- Apache-2.0 `LICENSE` at the repository root, declared by the Helm chart and
  the Python SDK metadata (b1f28be).
- Tenant-aware HTTP V3 and matching unary `deepdata.v3.DeepData` gRPC
  contracts for collection create/delete, single and atomic batch insert,
  document delete, collection get/list, tenant info, and search.
- One durable canonical mutation journal with sequence numbers, checksummed
  frames, synchronized acknowledgements, checkpoint rotation, strict replay,
  and narrowly scoped torn-tail recovery.
- Dense HNSW/Flat, sparse inverted/BM25, and two-field hybrid retrieval using
  caller-supplied vectors.
- Static administrative bearer authentication and scoped HS256 JWT policy
  shared by HTTP and gRPC.
- A deliberately small tenant-aware Python client, strict typing, package
  build checks, and authenticated live restart integration coverage.
- Hardened Linux amd64 Docker, Compose, and Helm contracts with non-root
  identity, read-only root filesystem, persistent storage, immutable image
  digest requirements, and public liveness/readiness probes.

### Changed

- `CollectionInfo` marshals snake_case keys (`name`, `fields`, `description`,
  `metadata`, `doc_count`) on canonical V3 responses instead of PascalCase
  (bc1fa27; internal/collection/manager.go:181-187).
- Journal appends keep a persistent descriptor and sync the parent directory
  once per open writer instead of on every record (6ce44e5).
- The non-RC `internal/graph` package computes CSR/PageRank locally instead of
  importing the retired cowrie gnn module; GraphRAG stays outside the RC and is
  not a supported surface (442cdb6).
- Recovery: journal replay streams records with bounded memory and snapshot
  coverage verification reads headers only (gates RCV-01, RCV-02); HNSW
  segment topology is persisted in the collection schema through the
  `segments` index parameter and never derived from GOMAXPROCS
  (internal/collection/collection.go:153-165); `DEEPDATA_BIND_HOST` binds both
  listeners to one IP literal. Bounded-memory unified snapshot load is gate
  RCV-03 (39bfe42).
- Status is derived from evidence: `tasks/gates.json` truth ledger driven by
  `scripts/gates.py`, the generated `docs/PRE_RELEASE_STATUS.md`, and the
  `scripts/check_docs_contract.py` docs linter (a443cd0).
- Errors are a structured envelope: `internal/apierror` maps the engine
  sentinels to one code table; HTTP writes `{code, message, hint, field,
  request_id, retryable, retry_after_ms, docs}` as JSON, gRPC attaches
  `ErrorInfo` and `RetryInfo` details, the MCP server forwards the envelope as
  `structuredContent`, and the Python client exposes it on `APIError` and
  retries on the server's `retryable` verdict. Tenant and collection limits
  answer 409 `quota_exceeded` / `FailedPrecondition` instead of 429; 429 is
  reserved for `rate_limited` and carries `Retry-After`; the search-shape and
  `hybrid_params`/`fallback` checks moved from the transports into the engine
  (gate CTL-01).
- Text in, text out: a `VectorField.embedding` binding (`provider`, `model`)
  journaled with the schema; `texts` on insert, batch insert, upsert and
  search across V3 HTTP, gRPC, the Python client and `cmd/deepdata-mcp`,
  resolved above the engine (`cmd/deepdata/embed_text.go`) by the one
  process embedder `DEEPDATA_EMBEDDER` names (`none` by default; `ollama`,
  `openai`, `onnx`, or the explicit-only `hash`); search responses carry
  `embedded_by`; new codes `embedding_mismatch` (409) and
  `embedder_unavailable` (503); `/readyz` reports `embedder`; sparse `bm25`
  bindings use the deterministic `TextToSparse` (gate CTL-02).
- `api/contract`: the v3 agent contract as embedded JSON Schema, one
  `{input, output}` file per MCP verb plus the error envelope, and
  `CONTRACT.md`. `cmd/deepdata-mcp` rewritten on it: six memory verbs
  (`deepdata_recall`, `deepdata_remember`, `deepdata_forget`, `deepdata_get`,
  `deepdata_collections`, `deepdata_create_collection`) whose
  `inputSchema`/`outputSchema` are the contract files verbatim, `outputSchema`
  plus `structuredContent` and safety annotations on every tool, resources
  `deepdata://contract` and `deepdata://status`, text routed to every bound
  field with a dense→sparse fallback by default, a `max_chars` token budget
  with `truncated` plus a steering hint, `DEEPDATA_COLLECTION`, and a place in
  the CI vet and test lists (gates CTL-03, CI-05). The five untyped tools
  `search`, `insert`, `upsert`, `get_document`, `list_collections` are gone.
- Normal startup exposes only the canonical V3 HTTP routes and V3 gRPC
  service. Legacy root/V2 mutation APIs and advanced recommendation,
  discovery, embedding-provider, GraphRAG, extraction, and feedback handlers
  are outside the RC and are not registered.
- Persistent runtime support is explicitly Linux amd64 for this candidate.
  Other OS/architecture jobs are compile proofs, not support claims.
- Authentication now requires exactly one credential of at least 32 bytes,
  rejects surrounding whitespace and URL query tokens, compares static tokens
  in constant time, and pins JWT validation to HS256.
- Historical `deepdata.v1` protobuf descriptors remain frozen; the breaking
  tenant-aware contract moved to `deepdata.v3`.

### Removed

- Legacy V2 HTTP surface: the handlers (bde4f94) and their registration with
  `/v2/insert`, `/v2/import`, `/v2/recommend`, `/v2/discover`,
  `/v2/collections`, and the associated benchmark harnesses; only the
  canonical V3 handlers can be registered (9d1672b, breaking).
- cmd/recalltest, superseded by `benchmarks/ddload` (39bfe42).

### Fixed

- Sparse index import fails closed on unparsable document IDs instead of
  collapsing them to ID 0 (59dc110).
- Weighted hybrid fusion normalization scans min/max explicitly so unsorted
  input cannot skew scores (9f45151).
- Snapshot serialization now preserves tenant, vector, index, metadata, ID,
  and recovery semantics and fails closed on corrupt/incompatible state.
- WAL and checkpoint ordering now preserve acknowledged mutations across
  crash, rotation, and restart scenarios without duplicate replay.
- Request cancellation after a durable append can no longer strand memory
  behind the journal or fault the store until restart.
- Python mutations are single-attempt after an ambiguous transport failure;
  only safe reads and explicitly read-only search requests are retried.

### Security and operational notes

- Built-in TLS and encryption at rest are not provided; both remain deployment
  responsibilities.
- Distributed/HA behavior, compliance audit logging, the web UI, and desktop
  packaging are not release gates or supported production surfaces.
- Python distribution metadata uses `deepdata-client` (the import remains
  `deepdata`) because the `deepdata` project on PyPI is unrelated. Claiming or
  publishing the selected name still requires explicit release authority.
- The license is Apache-2.0 (root `LICENSE`, b1f28be; gate REL-02). Tag,
  registry, and signing authority are tracked by gates PUB-01..PUB-03.

## Historical tag warning

Repository tags `v1.0.0` and `v1.0.1` describe an unrelated Atlas Runtime
artifact and must not be interpreted as DeepData releases. Older `v0.1.x`
source predates this canonical durability contract. Future DeepData tags and
migration guarantees will be documented explicitly before publication.
