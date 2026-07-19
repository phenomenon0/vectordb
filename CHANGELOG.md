# DeepData changelog

This changelog describes the headless, persistent, single-node DeepData
server. The current candidate is `0.2.0-rc.1`; it does not yet have a legal
license or published artifact.

## Unreleased single-node release candidate

### Added

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

### Fixed

- Snapshot serialization now preserves tenant, vector, index, metadata, ID,
  and recovery semantics and fails closed on corrupt/incompatible state.
- WAL and checkpoint ordering now preserve acknowledged mutations across
  crash, rotation, and restart scenarios without duplicate replay.
- Request cancellation after a durable append can no longer strand memory
  behind the journal or fault the store until restart.
- Python mutations are single-attempt after an ambiguous transport failure;
  only safe reads and explicitly read-only search requests are retried.

### Security and operational notes

- TLS termination and encryption at rest remain deployment responsibilities.
- Distributed/HA behavior, compliance audit logging, the web UI, and desktop
  packaging are not release gates or supported production surfaces.
- Python distribution metadata uses `deepdata-client` (the import remains
  `deepdata`) because the `deepdata` project on PyPI is unrelated. Claiming or
  publishing the selected name still requires explicit release authority.
- Legal ownership and license text must be resolved before publication.

## Historical tag warning

Repository tags `v1.0.0` and `v1.0.1` describe an unrelated Atlas Runtime
artifact and must not be interpreted as DeepData releases. Older `v0.1.x`
source predates this canonical durability contract. Future DeepData tags and
migration guarantees will be documented explicitly before publication.
