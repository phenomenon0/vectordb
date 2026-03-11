package collection

import (
	"encoding/json"
	"fmt"
	"strings"
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

// MarshalJSON outputs the string form of VectorType.
func (vt VectorType) MarshalJSON() ([]byte, error) {
	return json.Marshal(vt.String())
}

// UnmarshalJSON accepts both integer (0,1,2) and string ("dense","sparse","binary") forms.
func (vt *VectorType) UnmarshalJSON(data []byte) error {
	// Try int first
	var n int
	if err := json.Unmarshal(data, &n); err == nil {
		if n < 0 || n > 2 {
			return fmt.Errorf("unknown vector type: %d", n)
		}
		*vt = VectorType(n)
		return nil
	}
	// Try string
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("vector type must be int or string, got: %s", string(data))
	}
	switch strings.ToLower(s) {
	case "dense":
		*vt = VectorTypeDense
	case "sparse":
		*vt = VectorTypeSparse
	case "binary":
		*vt = VectorTypeBinary
	default:
		return fmt.Errorf("unknown vector type: %s", s)
	}
	return nil
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

// MarshalJSON outputs the string form of IndexType.
func (it IndexType) MarshalJSON() ([]byte, error) {
	return json.Marshal(it.String())
}

// UnmarshalJSON accepts both integer and string forms.
func (it *IndexType) UnmarshalJSON(data []byte) error {
	var n int
	if err := json.Unmarshal(data, &n); err == nil {
		if n < 0 || n > int(IndexTypeInverted) {
			return fmt.Errorf("unknown index type: %d", n)
		}
		*it = IndexType(n)
		return nil
	}

	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("index type must be int or string, got: %s", string(data))
	}

	parsed, err := ParseIndexType(strings.ToLower(s))
	if err != nil {
		return err
	}
	*it = parsed
	return nil
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
	Type   IndexType              `json:"type"`
	Params map[string]interface{} `json:"params,omitempty"`
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
	Name  string      `json:"name"`  // Field name (e.g., "embedding", "keywords")
	Type  VectorType  `json:"type"`  // Dense, Sparse, or Binary
	Dim   int         `json:"dim"`   // Vector dimension
	Index IndexConfig `json:"index"` // Index configuration
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
	Name        string                 `json:"name"`                  // Collection name
	Fields      []VectorField          `json:"fields"`                // Vector fields
	Metadata    map[string]interface{} `json:"metadata,omitempty"`    // Collection-level metadata
	Description string                 `json:"description,omitempty"` // Human-readable description
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
	ID       uint64                 `json:"id"`                 // Document ID
	Vectors  map[string]interface{} `json:"vectors,omitempty"`  // Field name -> vector data
	Metadata map[string]interface{} `json:"metadata,omitempty"` // Document metadata
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
	CollectionName string `json:"collection_name"`

	// Vector queries (field name -> query vector)
	Queries map[string]interface{} `json:"queries"`

	// Top-k results to return
	TopK int `json:"top_k"`

	// HNSW ef_search override (0 = use server default)
	EfSearch int `json:"ef_search,omitempty"`

	// Whether to include vectors in the response (nil = default true)
	IncludeVectors *bool `json:"include_vectors,omitempty"`

	// Metadata filters (optional)
	Filters map[string]interface{} `json:"filters,omitempty"`

	// Hybrid search parameters (optional)
	HybridParams *HybridSearchParams `json:"hybrid_params,omitempty"`
}

// HybridSearchParams configures hybrid search across multiple vector fields.
type HybridSearchParams struct {
	// Fusion strategy: "rrf", "weighted", or "linear"
	Strategy string `json:"strategy"`

	// Field weights (for weighted fusion)
	Weights map[string]float32 `json:"weights,omitempty"`

	// RRF constant (default: 60)
	RRFConstant float32 `json:"rrf_constant,omitempty"`
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
	Documents []Document `json:"documents"`

	// Scores for each document
	Scores []float32 `json:"scores"`

	// Query execution time (milliseconds)
	QueryTimeMs float64 `json:"query_time_ms"`

	// Number of candidates examined
	CandidatesExamined int `json:"candidates_examined"`
}

// RecommendRequest represents a recommendation request using positive/negative examples.
type RecommendRequest struct {
	CollectionName string   `json:"collection"`
	FieldName      string   `json:"field"`
	PositiveIDs    []uint64 `json:"positive_ids"`
	NegativeIDs    []uint64 `json:"negative_ids"`
	NegativeWeight float32  `json:"negative_weight"`
	TopK           int      `json:"top_k"`
	EfSearch       int      `json:"ef_search"`
	Filters        map[string]interface{} `json:"filters,omitempty"`
}

// ContextPair represents a positive/negative document pair for discovery search.
type ContextPair struct {
	PositiveID uint64 `json:"positive_id"`
	NegativeID uint64 `json:"negative_id"`
}

// DiscoverRequest represents a context-based discovery search request.
type DiscoverRequest struct {
	CollectionName string        `json:"collection"`
	FieldName      string        `json:"field"`
	TargetID       uint64        `json:"target_id"`
	TargetVector   []float32     `json:"target_vector"`
	Context        []ContextPair `json:"context"`
	TopK           int           `json:"top_k"`
	EfSearch       int           `json:"ef_search"`
	Filters        map[string]interface{} `json:"filters,omitempty"`
}
