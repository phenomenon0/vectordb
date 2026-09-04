# 0001 — Narrow the RC to a single-node, caller-vectors server

- Date: 2026-07-17
- Status: accepted
- Supersedes: —
- Evidence: tasks/todo.md:22-63 at a443cd0 ("Frozen RC Scope", written at a5720b8); cmd/deepdata/main.go:3198 (`const canonicalOnly = true`, since 355fbf3); cmd/deepdata/server.go:3414-3423 (`canonicalRCSurface`)

## Context
The pre-RC tree served root, V2 and advanced HTTP surfaces, a historical V1 gRPC service, and extraction and feedback handlers on top of one store. Rather than give each its own durability and authorization implementation, the RC disables them by default; only the V3 tenant surface gets the crash-durable engine, tenant authorization and the test matrix.

## Decision
The first RC is a Linux amd64, headless, single-node server: V3 tenant HTTP plus the mirrored deepdata.v3 gRPC, six journaled mutations, dense/sparse/hybrid search over caller-supplied vectors with HNSW, Flat and inverted indexes. `canonicalOnly` is a compile-time constant, not a flag; `canonicalRCSurface` answers 404 to every path outside /v3/tenants/ and the healthz/readyz/livez/metrics paths before CORS, OTel tracing and the router (cmd/deepdata/server.go:3379-3385); only the request-ID, panic-recovery and timeout wrappers sit outside it.

## Consequences
Everything excluded is listed in the non-goals block of docs/ARCHITECTURE.md and machine-checked by scripts/check_docs_contract.py (rules R8 and R11). Excluded code stays compiled but unreachable until ADR 0006 settles its disposition. ADR 0006 proposes the re-expansion; this decision stays in force for the RC tag.
