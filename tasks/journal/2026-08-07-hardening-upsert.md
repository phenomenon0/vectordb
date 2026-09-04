# 2026-08-07 — hardening-upsert

> Archived 2026-09-01 from docs/PRE_RELEASE_STATUS.md at 5dafec2 (that page is now generated from tasks/gates.json); unedited below this note. Claims below are dated evidence, not current status.

## Upsert / Get-Document Batch (a99fe53, 2026-08-07)

A single-semantic-commit feature pass on top of the 043ad5d hardening batch, adding the
two missing document operations to the canonical surface:

- **Caller-supplied-ID upsert (`PUT /docs/{id}` / gRPC `Upsert`):** replaces an existing
  document atomically under the collection write lock — no count inflation, live dense and
  sparse postings evicted then re-added, HNSW tombstones resurrected, and the scan cursor
  preserved by threading the prepared `nextID` through the replace path.
- **Durable journaling:** `upsert_document` is a new journal mutation with encode/decode,
  replay prepare/apply, and a shared-barrier `getDocument` read, so upserts survive
  restart with exactly-once replay (focused crash/replay + restart-persistence tests).
- **Get-document (`GET /docs/{id}` / gRPC `GetDoc`):** a shared-barrier single-document
  read; 404 when the id does not exist. Batch and legacy routes are untouched.
- **Surface:** `deepdata.v3` proto regenerated (9 -> 11 unary RPCs), canonical surface test
  updated, HTTP adds numeric-id PUT/GET with 404 on non-numeric segments.
- **Python SDK:** sync/async tenants gain `upsert(id=..., vectors=..., metadata=...)` and
  `get_document(...)` with strict typing and live integration coverage (upsert, get,
  restart persistence).

Verification for this batch: `go build ./...`, `go vet`, the full `cmd/deepdata` suite,
`-race` over `internal/collection`, and python unit (57 pass) + mypy + build all pass;
official check-runner receipts (`go-storage`, `go-vet-cgo0`, `go-short`, `go-race`,
`python-unit`, `python-mypy`, `python-build`) are bound to `a99fe53`.

## Hardening Batch (043ad5d, 2026-08-07)

A single-semantic-commit hardening pass on top of the network-isolation gate, from an
adversarial re-review of the in-process surfaces:

- **Strict dense-vector decode:** `decodeDenseVectorFast` now rejects elements whose value
  is finite outside `float32` range (which previously became `+Inf`/`-Inf` and poisoned
  distance/similarity), requires the closing `]`, and rejects trailing bytes after the
  array (a second bracketed array or garbage could previously be silently ignored).
- **UTF-8-safe request-ID truncation:** a client-supplied `X-Request-ID` longer than
  128 bytes is truncated on a rune boundary, so the echoed header and structured log
  field stay valid UTF-8 instead of a mid-rune cut.
- **Auth-gated `/metrics`:** the Prometheus surface is now behind the same
  `REQUIRE_AUTH` guard as the API routes (operation/volume detail no longer leaks when
  auth is enabled); probes stay public.
- **FLAT count correctness:** re-adding an existing ID (including a tombstone
  resurrection) replaces the vector without inflating `Count`.
- **Atomic HNSW batches:** `BatchAdd`/`BatchAddNoCopy` defer metadata registration and
  graph insertion until after the graph is fully built, and roll back newly registered
  nodes / resurrected tombstones / the count on a cancelled context or storage error —
  a failed batch leaves no partial state and every ID in it is re-addable.
- **Sparse overwrite correctness:** the cosine norm is recomputed from the stored vector
  on every `Add` (an all-zero overwrite no longer keeps a stale norm), and
  inverted-list postings emptied by an overwrite are dropped instead of lingering.

All of these are covered by new focused unit tests. Verification for this batch:
`go build ./...`, `go vet`, full `go test -short ./...`, and `-race` over
`internal/index`, `cmd/deepdata`, `internal/collection` all pass; official
check-runner receipts (`go-storage`, `go-vet-cgo0`, `go-short`, `go-race`) are bound
to `043ad5d`. One test-noise flake
(`benchmarks/review/TestDevOpsSREReview`, P99/P50 latency-ratio) was observed once and
passes on the controlled rerun (100%, 6/6); both logs are retained.

