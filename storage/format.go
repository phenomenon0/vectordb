// Package storage provides pluggable storage formats for VectorStore persistence.
// Supports gob (default, backward compatible) and sjson (optimized for embeddings).
package storage

import (
	"io"
	"time"
)

// Payload is the serializable snapshot of a VectorStore.
// This struct is shared by all storage formats for consistency.
type Payload struct {
	Dim       int
	Data      []float32 // Main embedding data - biggest optimization target
	Docs      []string
	IDs       []string
	Meta      map[uint64]map[string]string
	Deleted   map[uint64]bool
	Coll      map[uint64]string
	Next      int64
	Count     int
	HNSW      []byte // Serialized HNSW graph
	Checksum  string
	LastSaved time.Time
	LexTF     map[uint64]map[string]int
	DocLen    map[uint64]int
	DF        map[string]int
	SumDocL   int
	NumMeta   map[uint64]map[string]float64
	TimeMeta  map[uint64]map[string]time.Time
}

// Format defines the interface for storage format implementations.
type Format interface {
	// Name returns the format identifier (e.g., "gob", "sjson")
	Name() string

	// Extension returns the file extension (e.g., ".gob", ".sjson")
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

// Default returns the default storage format (gob for backward compatibility)
func Default() Format {
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
