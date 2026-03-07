// Package storage provides pluggable storage formats for VectorStore persistence.
// Supports gob (default, backward compatible) and cowrie (optimized for embeddings).
package storage

import (
	"io"
	"time"
)

// Payload is the serializable snapshot of a VectorStore.
// This struct is shared by all storage formats for consistency.
type Payload struct {
	FormatVersion int                          // NEW: Format version for backward compatibility (0=legacy, 2=with vector types)
	Dim           int
	Data          []float32                    // Main embedding data - biggest optimization target (LEGACY - use VectorData)
	VectorType    int                          // NEW: Default vector type for this collection (0=float32, 1=float16, 2=uint8, 3=sparse_coo)
	VectorData    map[uint64][]byte            // NEW: Per-vector encoded data (serialized VectorData)
	Docs          []string
	IDs           []string
	Meta          map[uint64]map[string]string
	Deleted       map[uint64]bool
	Coll          map[uint64]string
	TenantID      map[uint64]string
	Next          int64
	Count         int
	HNSW          []byte                       // Legacy serialized HNSW graph (deprecated)
	Indexes       map[string][]byte            // Index abstraction - collection -> serialized index
	Checksum      string
	LastSaved     time.Time
	LexTF         map[uint64]map[string]int
	DocLen        map[uint64]int
	DF            map[string]int
	SumDocL       int
	NumMeta       map[uint64]map[string]float64
	TimeMeta      map[uint64]map[string]time.Time
}

// Format defines the interface for storage format implementations.
type Format interface {
	// Name returns the format identifier (e.g., "gob", "cowrie")
	Name() string

	// Extension returns the file extension (e.g., ".gob", ".cowrie")
	Extension() string

	// Save writes the payload to the writer
	Save(w io.Writer, p *Payload) error

	// Load reads the payload from the reader
	Load(r io.Reader) (*Payload, error)
}

// registry holds available storage formats
var registry = make(map[string]Format)

// Register adds a format to the registry
func Register(f Format) {
	registry[f.Name()] = f
}

// Get retrieves a format by name, returns nil if not found
func Get(name string) Format {
	return registry[name]
}

// Default returns the default storage format (cowrie with compression for efficiency).
// For backward compatibility with existing gob files, use Get("gob").
func Default() Format {
	if f := registry["cowrie-zstd"]; f != nil {
		return f
	}
	// Fallback to uncompressed cowrie, then gob
	if f := registry["cowrie"]; f != nil {
		return f
	}
	return registry["gob"]
}

// List returns all registered format names
func List() []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	return names
}
