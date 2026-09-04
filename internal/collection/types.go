package collection

import (
	"encoding/json"
	"fmt"
	"strings"
)

// IsValidCanonicalIdentifier reports whether a tenant/collection identifier is
// safe as one unescaped canonical URL path segment and stable across clients.
func IsValidCanonicalIdentifier(value string) bool {
	if len(value) == 0 || len(value) > 64 {
		return false
	}
	for _, char := range value {
		if !((char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') ||
			(char >= '0' && char <= '9') || char == '_' || char == '-') {
			return false
		}
	}
	return true
}

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

// IndexTypes is the index vocabulary of the release candidate, in wire order.
// ParseIndexType, UnmarshalJSON, createDenseIndex, createSparseIndex, both
// schema validators, capabilities.index_types on GET /v3/status and the
// contract enum in deepdata_create_collection.json all derive from this slice,
// so an index type cannot exist for one layer and not another.
//
// IndexTypeIVF and IndexTypeDiskANN are deliberately absent: ADR 0001 narrows
// the RC to HNSW, Flat and Inverted. Their iota values stay because index
// types are journaled; renumbering would reinterpret every persisted schema.
// String keeps naming them so a v2-era journal or client fails by name.
var IndexTypes = []IndexType{IndexTypeHNSW, IndexTypeFLAT, IndexTypeInverted}

// Vectors reports which vector kind an index type serves, and whether it is a
// member of IndexTypes at all. It is the per-type property the validators and
// the index constructors split IndexTypes on, so "dense takes hnsw or flat,
// sparse takes inverted" is stated exactly once.
func (it IndexType) Vectors() (VectorType, bool) {
	switch it {
	case IndexTypeHNSW, IndexTypeFLAT:
		return VectorTypeDense, true
	case IndexTypeInverted:
		return VectorTypeSparse, true
	default:
		return 0, false
	}
}

// IndexTypeNames returns the wire names of IndexTypes in order. GET /v3/status
// publishes it and cmd/deepdata/contract_test.go compares the contract enum
// against it, so clients and the engine cannot disagree silently.
func IndexTypeNames() []string {
	names := make([]string, len(IndexTypes))
	for i, it := range IndexTypes {
		names[i] = it.String()
	}
	return names
}

// indexTypeNamesFor returns the wire names of the index types that serve vt,
// for error messages that tell a caller what it may send instead.
func indexTypeNamesFor(vt VectorType) []string {
	names := make([]string, 0, len(IndexTypes))
	for _, it := range IndexTypes {
		if kind, ok := it.Vectors(); ok && kind == vt {
			names = append(names, it.String())
		}
	}
	return names
}

// validateFieldIndexType is the one place the index vocabulary meets a schema.
// A field whose index type is not in IndexTypes, or is in it but indexes the
// other vector kind, would validate and journal yet have no constructor on
// replay: the collection comes back without that index after a restart.
func validateFieldIndexType(field VectorField) error {
	if kind, ok := field.Index.Type.Vectors(); !ok || kind != field.Type {
		return fmt.Errorf("%w: field %s uses index %s; %s fields accept only %s",
			ErrInvalidArgument, field.Name, field.Index.Type, field.Type,
			strings.Join(indexTypeNamesFor(field.Type), " or "))
	}
	return nil
}

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
		if _, ok := IndexType(n).Vectors(); !ok {
			return fmt.Errorf("%w: unknown index type: %d", ErrInvalidArgument, n)
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

// ParseIndexType converts a wire string to IndexType. Only members of
// IndexTypes parse; the retired names are refused by name so a v2-era journal
// or client fails loud instead of being reinterpreted as a live type.
func ParseIndexType(s string) (IndexType, error) {
	for _, it := range IndexTypes {
		if it.String() == s {
			return it, nil
		}
	}
	if s == IndexTypeIVF.String() || s == IndexTypeDiskANN.String() {
		return 0, fmt.Errorf("%w: index type %s was retired; the release supports %s",
			ErrInvalidArgument, s, strings.Join(IndexTypeNames(), ", "))
	}
	return 0, fmt.Errorf("%w: unknown index type: %s", ErrInvalidArgument, s)
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
	// Embedding binds the field to a text embedder so callers may send
	// `texts` instead of vectors. Journaled with the schema; nil means the
	// field only accepts vectors. Dense fields name the server embedder
	// (provider:model); sparse fields may bind only the deterministic
	// "bm25" term hash (TextToSparse).
	Embedding *EmbeddingConfig `json:"embedding,omitempty"`
}

// EmbeddingConfig names the embedder a field's texts are resolved with.
// The vector dimension is the field's Dim; there is no second copy.
type EmbeddingConfig struct {
	Provider string `json:"provider"`
	Model    string `json:"model,omitempty"`
}

// EmbeddingProviderBM25 is the only provider a sparse field may bind: the
// deterministic term-hash path any client can reproduce.
const EmbeddingProviderBM25 = "bm25"

// Validate checks if the vector field configuration is valid.
func (vf *VectorField) Validate() error {
	if vf.Name == "" {
		return fmt.Errorf("field name cannot be empty")
	}

	if vf.Dim <= 0 {
		return fmt.Errorf("dimension must be positive, got %d", vf.Dim)
	}

	if vf.Embedding != nil {
		if vf.Embedding.Provider == "" {
			return fmt.Errorf("embedding.provider cannot be empty")
		}
		isBM25 := vf.Embedding.Provider == EmbeddingProviderBM25
		if vf.Type == VectorTypeSparse && (!isBM25 || vf.Embedding.Model != "") {
			return fmt.Errorf("sparse fields may bind only embedding {provider: %q} without a model", EmbeddingProviderBM25)
		}
		if vf.Type == VectorTypeDense && isBM25 {
			return fmt.Errorf("embedding provider %q is for sparse fields; dense fields bind a text embedder", EmbeddingProviderBM25)
		}
	}

	// Validate index type matches vector type
	switch vf.Type {
	case VectorTypeDense, VectorTypeSparse:
		if err := validateFieldIndexType(*vf); err != nil {
			return err
		}
	case VectorTypeBinary:
		return fmt.Errorf("binary vectors not yet supported")
	default:
		return fmt.Errorf("unknown vector type: %d", vf.Type)
	}

	return nil
}

// Durability classes a collection's documents can be created with (ADR 0009).
// The collection's existence is class A either way — create and delete are
// journaled — but an ephemeral collection's documents are memory only: they
// write no journal record and are gone after restart, and its upstream
// rebuilds them.
const (
	DurabilityDurable   = "durable"
	DurabilityEphemeral = "ephemeral"
)

// normalizeDurability maps the persisted zero value onto the default class so
// every read reports a concrete one. Schemas written before ADR 0009 have no
// durability and are durable.
func normalizeDurability(durability string) string {
	if durability == "" {
		return DurabilityDurable
	}
	return durability
}

// CollectionSchema defines the schema for a multi-vector collection.
type CollectionSchema struct {
	Name        string                 `json:"name"`                  // Collection name
	Fields      []VectorField          `json:"fields"`                // Vector fields
	Metadata    map[string]interface{} `json:"metadata,omitempty"`    // Collection-level metadata
	Description string                 `json:"description,omitempty"` // Human-readable description
	Durability  string                 `json:"durability,omitempty"`  // "durable" (default) or "ephemeral"
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
			// %w, not %v: the index-type rejection carries ErrInvalidArgument
			// and the transports map that sentinel to a 400 rather than a 500.
			return fmt.Errorf("field %s: %w", field.Name, err)
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

const (
	// MaxSearchFields bounds the deliberately small RC hybrid surface.
	MaxSearchFields = 2
	// MaxSearchTopK bounds result allocation and response size.
	MaxSearchTopK = 1000
	// MaxSearchEf bounds the caller ef_search override so one request
	// cannot force a full-graph HNSW scan; large recall needs stay below the
	// cost of an unbounded beam.
	MaxSearchEf = 4096
	// MaxBatchDocuments bounds one atomic journaled batch.
	MaxBatchDocuments = 10_000
)

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

	// Whether to include vectors in the response (nil = default false)
	IncludeVectors *bool `json:"include_vectors,omitempty"`

	// Metadata filters (optional)
	Filters map[string]interface{} `json:"filters,omitempty"`

	// Hybrid search parameters (optional)
	HybridParams *HybridSearchParams `json:"hybrid_params,omitempty"`

	// ScoreFloor is a confidence filter on the returned raw scores.
	// Direction follows the field metric: on dense (distance) fields it is
	// a maximum acceptable distance (keep score <= floor); on sparse (BM25)
	// and fused hybrid scores it is a minimum acceptable score (keep score
	// >= floor). 0 disables it. It is also the weak-match trigger:
	// WeakMatch is reported when the floor is set and nothing survives.
	// The floor is caller-relative and not transferable across fields with
	// different metrics.
	ScoreFloor float64 `json:"score_floor,omitempty"`

	// Fallback configures the auto-fallback ladder (optional). Mutually
	// exclusive with HybridParams. Requires exactly the two named query
	// fields.
	Fallback *FallbackParams `json:"fallback,omitempty"`

	// UsageBoost blends frecency into ranking (0 = disabled, max 1).
	// 1.0 is rejected: a pure usage ranking would discard the similarity
	// signal entirely. The re-ordering only changes result order; reported
	// scores remain the raw per-field scores.
	UsageBoost float64 `json:"usage_boost,omitempty"`
}

// FallbackParams configures the auto-fallback ladder for a two-field
// request: the primary field is searched first; if it yields no results —
// or, when Threshold > 0, if its best score is worse than the threshold in
// the primary field's score direction (best distance > threshold on dense
// fields, best score < threshold on sparse fields) — the secondary field
// is searched and its results are returned with SearchResponse.FellBackTo
// set. With Threshold == 0 the ladder degrades to "only fall back on zero
// hits". The decision uses the primary answer after ScoreFloor has been
// applied, so a primary that yields no confident result at all (zero hits,
// or wiped out by the floor) is also treated as weak.
type FallbackParams struct {
	Primary   string  `json:"primary"`
	Secondary string  `json:"secondary"`
	Threshold float64 `json:"threshold,omitempty"`
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

	// BestScore is the best raw score among the returned documents (the
	// minimum on distance fields, the maximum on score fields; 0 when
	// there are none). It lets callers calibrate ScoreFloor.
	BestScore float32 `json:"best_score,omitempty"`

	// WeakMatch is true when ScoreFloor > 0 and no document survived it.
	// Agents should treat a weak-match response as "no confident answer"
	// rather than consuming the results.
	WeakMatch bool `json:"weak_match"`

	// FellBackTo names the secondary field used when the fallback ladder
	// fired (empty when the primary field answered the query).
	FellBackTo string `json:"fell_back_to,omitempty"`

	// ScoreDirection tells the caller how to read Scores and BestScore:
	// "lower_is_better" on dense distance fields, "higher_is_better" on
	// sparse BM25 fields and on fused hybrid contributions. It names the
	// direction of the answer actually returned, so a fallback response
	// reports the secondary field's direction.
	ScoreDirection string `json:"score_direction,omitempty"`
}

// Score directions reported by SearchResponse.ScoreDirection and
// FieldInfo.ScoreDirection.
const (
	ScoreDirectionLowerIsBetter  = "lower_is_better"
	ScoreDirectionHigherIsBetter = "higher_is_better"
)
