// Package graph provides GraphRAG integration for DeepData.
//
// It maintains a knowledge graph built from extraction pipeline output,
// stored as CSR adjacency for efficient PageRank computation.
// Graph importance scores are used as a third signal in hybrid fusion
// alongside dense (vector) and sparse (BM25) results.
//
// Tower layer: L1 engine, were it live.
//
// Status: dormant, not part of the RC surface; nothing registers it when
// canonicalOnly is set. It returns only as a class-C in-memory signal per
// ADR 0008 (accepted 2026-09-01),
// docs/decisions/0008-graph-and-extraction-return-as-signals-over-text-in.md
// (gate MEM-01).
package graph
