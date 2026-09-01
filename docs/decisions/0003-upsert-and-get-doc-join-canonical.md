# 0003 — Upsert and get-document join the canonical surface

- Date: 2026-08-07
- Status: accepted
- Supersedes: the exclusion of upsert and single-document read from the RC — tasks/todo.md:29-30 at a443cd0 (the frozen mutation list) and docs/why-vectordb.md:43 at a443cd0 ("Upsert/update, document fetch/scan" under "When it is not a good fit", pre-2026-09-01 wording)
- Evidence: a99fe53 (feat(collection): caller-supplied-ID upsert and get-document); tasks/journal/2026-08-07-hardening-upsert.md:5-28; api/proto/deepdata/v3/deepdata.proto:250-251 (rpc Upsert, rpc GetDoc)

## Context
The frozen scope listed create collection, delete collection, insert, batch insert and delete document, and no single-document read. An agent that stores memories must replace a record under its own id and read it back. The two operations were absent, not unsafe.

## Decision
Caller-supplied-id upsert (PUT /docs/{id}, gRPC Upsert) is a journaled mutation: atomic replace under the collection write lock, no count inflation, exactly-once replay. Get-document (GET /docs/{id}, gRPC GetDoc) is a shared-barrier read, 404 when the id does not exist. Both land on HTTP, gRPC and the Python SDK in one commit.

## Consequences
The canonical surface is 11 unary RPCs and six mutations; smaller historical counts in docs are drift that linter rules R1 and R2 catch. Batch and legacy routes are untouched. DUR-01 to DUR-04 cover the journaled path.
