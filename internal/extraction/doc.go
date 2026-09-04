// Package extraction provides LLM-based entity and relationship extraction
// for building knowledge graphs from unstructured text.
//
// This package implements "cognify" functionality similar to Cognee,
// automatically extracting entities and relationships from text chunks
// and storing them in the existing EntityGraph infrastructure.
//
// Tower layer: none — were it live it would sit above L4, as a client.
//
// Status: dormant, not part of the RC surface; its handlers are registered
// only when canonicalOnly is off. It returns only as an out-of-band client
// per ADR 0008 (accepted 2026-09-01),
// docs/decisions/0008-graph-and-extraction-return-as-signals-over-text-in.md
// (gate MEM-01).
package extraction
