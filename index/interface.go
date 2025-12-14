package index

import (
	"context"

	"agentscope/vectordb/filter"
)

// Index is the core interface for vector indexes in VectorDB.
// All index implementations (HNSW, IVF, FLAT, DiskANN) must implement this interface.
//
// Design Philosophy:
// - Simple: Minimal required methods for core operations
// - Pluggable: Easy to swap index implementations
// - Efficient: Supports both in-memory and disk-backed indexes
// - Observable: Returns metrics and statistics
//
// Implementations should be thread-safe or document their concurrency requirements.
type Index interface {
	// Name returns the human-readable name of the index type (e.g., "HNSW", "IVF", "FLAT")
	Name() string

	// Add inserts a vector into the index with the given ID.
	// Vector dimensions must match the index's configured dimension.
	//
	// Returns:
	//   - error if vector dimensions mismatch, ID already exists, or index is full
	//
	// Thread-safety: Implementation-dependent (document in concrete types)
	Add(ctx context.Context, id uint64, vector []float32) error

	// Search finds the k nearest neighbors to the query vector.
	// Returns up to k results sorted by ascending distance (closest first).
	//
	// Parameters:
	//   - query: Vector to search for (must match index dimension)
	//   - k: Number of nearest neighbors to return
	//   - params: Search-specific parameters (e.g., ef_search for HNSW, nprobe for IVF)
	//
	// Returns:
	//   - []Result: Neighbors sorted by distance (closest first)
	//   - error: If query dimensions mismatch or search fails
	//
	// Thread-safety: Safe for concurrent reads (writes may block depending on implementation)
	Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error)

	// Delete marks a vector as deleted (tombstone).
	// Deleted vectors are excluded from search results.
	// Actual removal may be deferred until compaction.
	//
	// Returns:
	//   - error if ID does not exist or deletion fails
	//
	// Thread-safety: Implementation-dependent
	Delete(ctx context.Context, id uint64) error

	// Stats returns index statistics (size, count, memory usage, etc.)
	// Useful for monitoring, debugging, and capacity planning.
	Stats() IndexStats

	// Export serializes the index to bytes for persistence or replication.
	// Format is index-specific but should be self-contained (include metadata).
	//
	// Returns:
	//   - []byte: Serialized index data
	//   - error: If serialization fails
	//
	// Thread-safety: Should be safe to call concurrently with Search (snapshot semantics)
	Export() ([]byte, error)

	// Import loads a previously exported index from bytes.
	// This may replace the current index state entirely.
	//
	// Returns:
	//   - error: If data is invalid or incompatible with index type
	//
	// Thread-safety: Must NOT be called concurrently with other operations
	Import(data []byte) error
}

// SearchParams holds search-specific parameters for different index types.
// Index implementations can type-assert to their specific parameter types.
//
// Examples:
//   - HNSW: ef_search (beam width)
//   - IVF: nprobe (clusters to search)
//   - FLAT: no parameters (exhaustive search)
type SearchParams interface {
	// Type returns the parameter type name for validation
	Type() string
}

// DefaultSearchParams provides sensible defaults for search
type DefaultSearchParams struct{}

func (DefaultSearchParams) Type() string { return "default" }

// HNSWSearchParams are parameters specific to HNSW index search
type HNSWSearchParams struct {
	EfSearch int            // Beam width for search (higher = more accurate, slower)
	Filter   filter.Filter  // Optional metadata filter (nil = no filtering)
}

func (HNSWSearchParams) Type() string { return "hnsw" }

// IVFSearchParams are parameters specific to IVF index search
type IVFSearchParams struct {
	NProbe int            // Number of clusters to search (higher = more accurate, slower)
	Filter filter.Filter  // Optional metadata filter (nil = no filtering)
}

func (IVFSearchParams) Type() string { return "ivf" }

// Result represents a single search result (ID + distance/score)
type Result struct {
	ID       uint64                 // Vector ID
	Distance float32                // Distance from query (lower = more similar)
	Score    float32                // Optional: Normalized score (higher = more similar)
	Metadata map[string]interface{} // Optional: Vector metadata (for filtered search)
}

// IndexStats provides statistics about the index state
type IndexStats struct {
	// Index identity
	Name string // Index type name (e.g., "HNSW")
	Dim  int    // Vector dimensions

	// Capacity and usage
	Count   int // Total vectors in index
	Deleted int // Tombstoned vectors (not yet compacted)
	Active  int // Active vectors (Count - Deleted)

	// Memory usage (bytes)
	MemoryUsed int64 // Approximate memory usage
	DiskUsed   int64 // Disk usage (0 for in-memory indexes)

	// Performance metrics (optional, implementation-specific)
	Extra map[string]interface{} // Additional index-specific stats
}

// Factory creates index instances by type name
// This will be implemented in factory.go
type Factory interface {
	// Create creates a new index of the specified type
	Create(indexType string, dim int, config map[string]interface{}) (Index, error)

	// SupportedTypes returns the list of supported index types
	SupportedTypes() []string
}
