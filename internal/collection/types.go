package collection

import (
	"fmt"
)

// VectorType defines the type of vector stored in a field.
type VectorType int

const (
	// VectorTypeDense represents dense float32 vectors (e.g., embeddings).
	VectorTypeDense VectorType = iota

	// VectorTypeSparse represents sparse vectors (e.g., BM25, SPLADE).
	VectorTypeSparse

	// VectorTypeBinary represents binary vectors (future).
	VectorTypeBinary
)

func (vt VectorType) String() string {
	switch vt {
	case VectorTypeDense:
		return "dense"
	case VectorTypeSparse:
		return "sparse"
	case VectorTypeBinary:
		return "binary"
	default:
		return "unknown"
	}
}

// ParseVectorType converts a string to VectorType.
func ParseVectorType(s string) (VectorType, error) {
	switch s {
	case "dense":
		return VectorTypeDense, nil
	case "sparse":
		return VectorTypeSparse, nil
	case "binary":
		return VectorTypeBinary, nil
	default:
		return 0, fmt.Errorf("unknown vector type: %s", s)
	}
}

// IndexType defines the type of index used for a vector field.
type IndexType int

const (
	// IndexTypeHNSW is a graph-based index for dense vectors.
	IndexTypeHNSW IndexType = iota

	// IndexTypeIVF is a clustering-based index for dense vectors.
	IndexTypeIVF

	// IndexTypeFLAT is brute-force search (exact).
	IndexTypeFLAT

	// IndexTypeDiskANN is a disk-backed hybrid index.
	IndexTypeDiskANN

	// IndexTypeInverted is an inverted index for sparse vectors.
	IndexTypeInverted
)

func (it IndexType) String() string {
	switch it {
	case IndexTypeHNSW:
		return "hnsw"
	case IndexTypeIVF:
		return "ivf"
	case IndexTypeFLAT:
		return "flat"
	case IndexTypeDiskANN:
		return "diskann"
	case IndexTypeInverted:
		return "inverted"
	default:
		return "unknown"
	}
}

// ParseIndexType converts a string to IndexType.
func ParseIndexType(s string) (IndexType, error) {
	switch s {
	case "hnsw":
		return IndexTypeHNSW, nil
	case "ivf":
		return IndexTypeIVF, nil
	case "flat":
		return IndexTypeFLAT, nil
	case "diskann":
		return IndexTypeDiskANN, nil
	case "inverted":
		return IndexTypeInverted, nil
	default:
		return 0, fmt.Errorf("unknown index type: %s", s)
	}
}

// IndexConfig holds configuration for a specific index.
type IndexConfig struct {
	Type   IndexType
	Params map[string]interface{}
}

// VectorField defines a single vector field in a collection.
//
// A collection can have multiple vector fields, each with its own
// type, dimension, and index configuration.
//
// Example:
//   - Field "embedding": Dense, 384-dim, HNSW index
//   - Field "keywords": Sparse, 10000-dim, Inverted index
type VectorField struct {
	Name   string      // Field name (e.g., "embedding", "keywords")
	Type   VectorType  // Dense, Sparse, or Binary
	Dim    int         // Vector dimension
	Index  IndexConfig // Index configuration
}

// Validate checks if the vector field configuration is valid.
func (vf *VectorField) Validate() error {
	if vf.Name == "" {
		return fmt.Errorf("field name cannot be empty")
	}

	if vf.Dim <= 0 {
		return fmt.Errorf("dimension must be positive, got %d", vf.Dim)
	}

	// Validate index type matches vector type
	switch vf.Type {
	case VectorTypeDense:
		if vf.Index.Type == IndexTypeInverted {
			return fmt.Errorf("inverted index not supported for dense vectors")
		}
	case VectorTypeSparse:
		if vf.Index.Type != IndexTypeInverted {
			return fmt.Errorf("sparse vectors require inverted index, got %s", vf.Index.Type)
		}
	case VectorTypeBinary:
		return fmt.Errorf("binary vectors not yet supported")
	default:
		return fmt.Errorf("unknown vector type: %d", vf.Type)
	}

	return nil
}

// CollectionSchema defines the schema for a multi-vector collection.
type CollectionSchema struct {
	Name        string                 // Collection name
	Fields      []VectorField          // Vector fields
	Metadata    map[string]interface{} // Collection-level metadata
	Description string                 // Human-readable description
}

// Validate checks if the collection schema is valid.
func (cs *CollectionSchema) Validate() error {
	if cs.Name == "" {
		return fmt.Errorf("collection name cannot be empty")
	}

	if len(cs.Fields) == 0 {
		return fmt.Errorf("collection must have at least one vector field")
	}

	// Check for duplicate field names
	fieldNames := make(map[string]bool)
	for _, field := range cs.Fields {
		if fieldNames[field.Name] {
			return fmt.Errorf("duplicate field name: %s", field.Name)
		}
		fieldNames[field.Name] = true

		// Validate each field
		if err := field.Validate(); err != nil {
			return fmt.Errorf("field %s: %v", field.Name, err)
		}
	}

	return nil
}

// GetField returns a field by name, or nil if not found.
func (cs *CollectionSchema) GetField(name string) *VectorField {
	for i := range cs.Fields {
		if cs.Fields[i].Name == name {
			return &cs.Fields[i]
		}
	}
	return nil
}

// HasField checks if a field with the given name exists.
func (cs *CollectionSchema) HasField(name string) bool {
	return cs.GetField(name) != nil
}

// FieldCount returns the number of vector fields.
func (cs *CollectionSchema) FieldCount() int {
	return len(cs.Fields)
}

// Document represents a single document with multiple vector fields.
type Document struct {
	ID       uint64                            // Document ID
	Vectors  map[string]interface{}            // Field name -> vector data
	Metadata map[string]interface{}            // Document metadata
}

// Validate checks if the document matches the collection schema.
func (d *Document) Validate(schema *CollectionSchema) error {
	if d.ID == 0 {
		return fmt.Errorf("document ID cannot be zero")
	}

	// Check all required fields are present
	for _, field := range schema.Fields {
		if _, exists := d.Vectors[field.Name]; !exists {
			return fmt.Errorf("missing required vector field: %s", field.Name)
		}
	}

	// Check for extra fields
	for fieldName := range d.Vectors {
		if !schema.HasField(fieldName) {
			return fmt.Errorf("unknown vector field: %s", fieldName)
		}
	}

	return nil
}

// GetVector retrieves a vector field by name.
func (d *Document) GetVector(fieldName string) (interface{}, bool) {
	vec, ok := d.Vectors[fieldName]
	return vec, ok
}

// SetVector sets a vector field.
func (d *Document) SetVector(fieldName string, vector interface{}) {
	if d.Vectors == nil {
		d.Vectors = make(map[string]interface{})
	}
	d.Vectors[fieldName] = vector
}

// GetMetadata retrieves a metadata field.
func (d *Document) GetMetadata(key string) (interface{}, bool) {
	if d.Metadata == nil {
		return nil, false
	}
	val, ok := d.Metadata[key]
	return val, ok
}

// SetMetadata sets a metadata field.
func (d *Document) SetMetadata(key string, value interface{}) {
	if d.Metadata == nil {
		d.Metadata = make(map[string]interface{})
	}
	d.Metadata[key] = value
}

// SearchRequest represents a multi-vector search request.
type SearchRequest struct {
	// Collection to search
	CollectionName string

	// Vector queries (field name -> query vector)
	Queries map[string]interface{}

	// Top-k results to return
	TopK int

	// Metadata filters (optional)
	Filters map[string]interface{}

	// Hybrid search parameters (optional)
	HybridParams *HybridSearchParams
}

// HybridSearchParams configures hybrid search across multiple vector fields.
type HybridSearchParams struct {
	// Fusion strategy: "rrf", "weighted", or "linear"
	Strategy string

	// Field weights (for weighted fusion)
	Weights map[string]float32

	// RRF constant (default: 60)
	RRFConstant float32
}

// DefaultHybridParams returns recommended hybrid search parameters.
func DefaultHybridParams() *HybridSearchParams {
	return &HybridSearchParams{
		Strategy:    "rrf",
		RRFConstant: 60.0,
		Weights: map[string]float32{
			"embedding": 0.7,
			"keywords":  0.3,
		},
	}
}

// SearchResponse represents the results of a multi-vector search.
type SearchResponse struct {
	// Matched documents
	Documents []Document

	// Scores for each document
	Scores []float32

	// Query execution time (milliseconds)
	QueryTimeMs float64

	// Number of candidates examined
	CandidatesExamined int
}
