// Package index is the vector index abstraction the engine builds on: the
// registry and factory for the registered index types (hnsw, flat, sparse),
// their segmented and payload-filtered variants, and the shared distance,
// quantization and config helpers they use.
//
// Tower layer: L1 engine.
package index
