# 0007 — Persisted segment count, never derived from GOMAXPROCS

- Date: 2026-09-01
- Status: accepted
- Supersedes: —
- Evidence: 39bfe42 (wip(recovery): bounded-memory journal replay and coverage streaming; adds internal/index/segmented.go); internal/index/segmented.go:85-87 (segments must be >= 1) and :106-114 (SegmentsForNewCollection returns 1); tasks/lessons.md:109-113

## Context
SegmentedIndex splits a dense index into N independent segments owned by id modulo N, so builds and searches fan out. A missing segments parameter could be filled from GOMAXPROCS — but the same journal would then rebuild into a different graph layout on another host, and high-core machines would silently multiply build, search and export concurrency.

## Decision
Index topology is durable semantics. The segment count is an explicit, persisted schema parameter (IndexConfig.Params["segments"]); schemas that omit it get the historical one-graph default, 1, on every host.

## Consequences
Replay is reproducible across hosts. Parallel segmented builds (ADR 0005, lever 1) are opt-in per collection, never ambient. RCV-06 tracks measuring them under the cold-start memory envelope; internal/collection/segmented_topology_test.go is in its scope.
