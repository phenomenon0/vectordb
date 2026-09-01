# 0005 — Ingest levers: segments, then allocations, then group commit

- Date: 2026-08-21
- Status: accepted
- Supersedes: the ordering in docs/BENCHMARKS.md:42-43 at a443cd0, which lists group commit ahead of parallel index construction
- Evidence: 6ce44e5 (perf(journal): persistent descriptor and once-per-writer namespace sync); the 2026-08-22 session notes at tasks/todo.md:357-390 at a443cd0, moving to tasks/journal/2026-08-22-hardening-ingest.md in this rewrite; docs/BENCHMARKS.md:23-30 at a443cd0

## Context
Engine benchmarks put FLAT dim4 batch100 at 784,506 ns/op (~7.8 us/doc) and HNSW m16 efc300 dim128 batch100 at 15,524,123 ns/op (~155 us/doc); end-to-end durable ingest measured ~277 us/doc. Journal fsync was already amortized per batch, and 6ce44e5 cut a durable single insert from 20,725 to 8,059 ns/op. Single-threaded HNSW construction under the collection lock dominates; time-to-searchable is the ~9x gap against Qdrant 1.19 (BENCHMARKS.md:28, :37-43 at a443cd0).

## Decision
Levers in this order: (1) parallel index construction across segments; (2) allocation reduction in the HNSW build path (817 allocs/doc measured); (3) group commit, deferred until (1) lands or a concurrent small-writer workload demands it — it would not move the benchmarked batch-ingest number.

## Consequences
Segmented builds exist at internal/index/segmented.go (39bfe42) and are governed by ADR 0007; RCV-06 measures them under the cold-start memory envelope before they count. Group commit has no gate. BENCHMARKS.md prose that still leads with group commit is drift for the docs rewrite to correct.
