# Mega Benchmark: Wide-Sweep All-VDB Autonomous Run

> This checklist covers the completed benchmark subtask. Overall product readiness is tracked in [`docs/PRE_RELEASE_STATUS.md`](../docs/PRE_RELEASE_STATUS.md).

## Plan
- [x] Build self-contained Python script `benchmarks/mega_bench.py`
- [x] Use gRPC for the canonical DeepData search path; retain HTTP as a separately configured diagnostic baseline
- [x] Complete 3 datasets × 6 ef_search levels × 5 VDB systems, plus the DeepData HTTP diagnostic and failure modes (108 cells)
- [x] Checkpoint every measurement with atomic keyed upserts and resume validation
- [x] Generate the final report and open it in VelvetMD
- [x] Commit the completed benchmark repair

## Review

Completed: 2026-07-17

- Canonicalized the legacy checkpoint from 109 rows/108 keys to 108 unique rows and made the newest keyed result authoritative.
- Fixed Milvus ground-truth identity by inserting explicit zero-based primary keys.
- Moved Milvus HNSW creation after insert/flush, waited for complete indexing, verified indexed row counts, and recorded effective ef values.
- Reran all 18 Milvus cells; final checkpoint has 108/108 unique completed cells, zero failures, and 7/7 passing DeepData failure-mode checks.
- Added atomic checkpoint replacement, a single-run lock, scoped reruns, dependency/dataset manifests, pending-only service startup, and final completeness invariants.
- Added seven regression tests for checkpoint normalization/upserts, non-destructive scoped reruns, overwrite guards, dependency drift, and sparse-matrix report coverage.
- The report discloses that the gRPC row uses float16 quantization while the HTTP diagnostic uses full precision, and distinguishes top-100 recall/latency from top-10 steady-state QPS. Ninety retained legacy cells predate per-cell timestamps; the 18 corrected Milvus cells include timestamps and lifecycle metadata.
