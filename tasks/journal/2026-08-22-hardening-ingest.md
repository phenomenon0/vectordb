# Session 2026-08-22 — hardening batch, publication gates, and ingest-bottleneck evidence

> Archived 2026-09-01 from tasks/todo.md:357-390 at a443cd0; unedited below this note.

## Session 2026-08-22 — hardening batch, publication gates, and ingest-bottleneck evidence

**Commits 59dc110..bc1fa27 (+benchmarks/license docs). Remote RC CI green on this branch for the first time since the candidate moved past 14d1442.**

Closed pre-existing open items:
- License chosen (Apache-2.0) and committed; PRE_RELEASE_STATUS publication gate updated. Remaining publication authority (tag, registries, signing) still separately gated.
- Cowrie/go mod tidy breaker resolved: internal/graph no longer imports the retired private cowrie/gnn module; local CSR+PageRank vendored behind identical call sites. `go mod tidy` exits clean.
- Offline backup/restore rehearsal executed for real (`scripts/backup_restore_drill.sh`): seed 500 docs -> graceful stop -> whole-root copy + manifest -> destroy live state -> restore -> strict verify (schema, landmark doc with metadata, full searchability). DRILL PASSED.

Hardening fixes (each with tests):
- sparse import fail-closed on unparsable doc IDs (was silent collapse to 0)
- legacy V2 HTTP surface deleted end to end (handlers, registration, benchmark harnesses); only /v3/tenants is served
- HNSW parallel-insert worker RNG now crypto/rand-seeded per worker
- Collection.Schema() can no longer leak live state on clone failure
- weighted fusion normalizes via explicit min/max scan (no sorted-input assumption)
- IVF nprobe request-configurable (`n_probe`, zero value = default 10)
- CollectionInfo marshals snake_case (was PascalCase leak); one Go test updated to canonical keys
- canonical request paths use boxing-free dense decode (decodeCanonicalVectorRaw); sparse keeps exact error contract
- journal append keeps a persistent descriptor and syncs the parent dir once per writer instead of per record — A/B: durable single insert 20,725 -> 8,059 ns/op (2.6x); crash/restart matrices and race suites green

New evidence tooling:
- `benchmarks/ddload` and `benchmarks/ddload-qdrant`: stdlib Go clients so numbers measure servers, not Python serialization.
- Equal-effort head-to-head vs Qdrant 1.19 published in docs/BENCHMARKS.md with caveats. Headlines: time-to-searchable gap ~9x (our weakest axis); DeepData serves 7x serial / 33x more concurrent searches against the same collection (hot-query caveat stated).

**Ingest bottleneck identified by measurement — roadmap correction.**
BenchmarkDurableInsertBatch100HNSW vs FLAT at engine level:
- FLAT dim4 batch100: 784,506 ns/op (~7.8 us/doc)
- HNSW m16 efc300 dim128 batch100: 15,524,123 ns/op (~155 us/doc)
End-to-end durable ingest (ddload, same schema): ~277 us/doc. Conclusion: single-threaded HNSW construction under the collection lock dominates ingest; journal fsync is already amortized per batch. Group commit was deprioritized accordingly — it helps many-concurrent-small-writer patterns but would not move the benchmarked batch-ingest number.

Re-prioritized next levers (replacing "group commit" as Wave 3B):
1. Parallel index construction across segments (Qdrant-style segmented builds) — largest win, largest scope.
2. Allocation reduction in the HNSW build path (817 allocs/doc measured).
3. Group commit for concurrent small-writer workloads (deferred until 1 lands or workload demands it).
