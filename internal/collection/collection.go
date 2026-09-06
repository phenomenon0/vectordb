package collection

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sync"
	"time"

	"github.com/phenomenon0/vectordb/internal/filter"
	"github.com/phenomenon0/vectordb/internal/hybrid"
	"github.com/phenomenon0/vectordb/internal/index"
	"github.com/phenomenon0/vectordb/internal/index/simd"
	"github.com/phenomenon0/vectordb/internal/sparse"
)

// DefaultEfSearch is the ef_search a new Collection starts with. The server
// sets it once at startup from its configuration, before any collection exists.
var DefaultEfSearch = 200

// Collection manages multiple vector indexes for a single collection.
//
// A collection can have multiple vector fields, each with its own index:
//   - Dense fields use the dense members of IndexTypes (HNSW, FLAT)
//   - Sparse fields use InvertedIndex
//
// Example:
//
//	collection with "embedding" (HNSW) + "keywords" (Inverted)
type Collection struct {
	schema  CollectionSchema
	indexes map[string]index.Index // field name -> index instance
	sparse  map[string]*sparse.InvertedIndex
	mu      sync.RWMutex

	// Document storage
	documents map[uint64]*Document // doc_id -> document
	nextID    uint64

	// Default ef_search for HNSW, taken from DefaultEfSearch at creation time.
	defaultEfSearch int

	// usage is an in-memory, non-durable frecency signal over documents
	// this collection has returned or fetched. Session signal only: it
	// resets on restart and must only nudge ranking, never replace it.
	usage *UsageTracker

	// durableReadOnly prevents a Collection pointer obtained from a durable V2
	// manager or tenant read API from bypassing the canonical tenant WAL.
	durableReadOnly bool

	// pendingCommit holds IDs reserved by commitPrepared (already visible in
	// documents) whose index insert has not finished yet. deleteDocumentDirect
	// waits on commitCond until an ID clears pendingCommit before touching the
	// indexes; otherwise it can hit an index whose insert for that ID hasn't
	// started yet, get a spurious "not found" from idx.Delete, and abort with
	// the document (and any postings the loop already reached) left stuck.
	pendingCommit map[uint64]struct{}
	commitCond    *sync.Cond
}

// NewCollection creates a new multi-vector collection.
func NewCollection(schema CollectionSchema) (*Collection, error) {
	if err := schema.Validate(); err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}

	c := &Collection{
		schema:          schema,
		indexes:         make(map[string]index.Index),
		sparse:          make(map[string]*sparse.InvertedIndex),
		documents:       make(map[uint64]*Document),
		nextID:          1,
		defaultEfSearch: DefaultEfSearch,
		usage:           NewUsageTracker(),
		pendingCommit:   make(map[uint64]struct{}),
	}
	c.commitCond = sync.NewCond(&c.mu)

	// Initialize indexes for each field
	for _, field := range schema.Fields {
		if err := c.createIndex(field); err != nil {
			return nil, fmt.Errorf("failed to create index for field %s: %w", field.Name, err)
		}
	}

	return c, nil
}

// createIndex creates an index instance for a vector field.
func (c *Collection) createIndex(field VectorField) error {
	switch field.Type {
	case VectorTypeDense:
		return c.createDenseIndex(field)
	case VectorTypeSparse:
		return c.createSparseIndex(field)
	case VectorTypeBinary:
		return fmt.Errorf("binary vectors not yet supported")
	default:
		return fmt.Errorf("unknown vector type: %d", field.Type)
	}
}

// createDenseIndex creates a dense vector index: one constructor per dense
// member of IndexTypes.
// segmentCountFromParam coerces the schema's "segments" value (decoded as
// float64 from JSON) into a validated segment count.
func segmentCountFromParam(raw interface{}) (int, error) {
	switch v := raw.(type) {
	case int:
		if v < 1 || v > 64 {
			return 0, fmt.Errorf("segments must be in [1,64], got %d", v)
		}
		return v, nil
	case float64:
		if v != math.Trunc(v) || v < 1 || v > 64 {
			return 0, fmt.Errorf("segments must be an integer in [1,64], got %v", v)
		}
		return int(v), nil
	default:
		return 0, fmt.Errorf("segments must be a number in [1,64]")
	}
}

func (c *Collection) createDenseIndex(field VectorField) error {
	config := field.Index.Params
	if config == nil {
		config = make(map[string]interface{})
	}

	var idx index.Index
	var err error

	switch field.Index.Type {
	case IndexTypeHNSW:
		// Set defaults if not provided
		if _, ok := config["m"]; !ok {
			config["m"] = 16
		}
		if _, ok := config["ef_construction"]; !ok {
			config["ef_construction"] = 200
		}

		// Segment topology is part of the persisted schema, not an ambient host
		// tuning knob. A missing parameter preserves the historical single HNSW
		// graph; callers opt into deterministic segmented routing explicitly.
		// This also prevents a journal from replaying into a different topology
		// merely because GOMAXPROCS changed between hosts or restarts.
		segments := index.SegmentsForNewCollection()
		if raw, ok := config["segments"]; ok {
			parsed, convErr := segmentCountFromParam(raw)
			if convErr != nil {
				return fmt.Errorf("field %s: %w", field.Name, convErr)
			}
			segments = parsed
		}

		if segments > 1 {
			// Each segment is an independent HNSW graph built from the same
			// configuration; docs route deterministically by document ID.
			// The wrapper hides them behind the single-index interfaces, so
			// search/delete/export paths below need no segmentation awareness.
			hnswConfig := make(map[string]interface{}, len(config))
			for k, v := range config {
				if k == "segments" {
					continue // wrapper-level knob, not an HNSW parameter
				}
				hnswConfig[k] = v
			}
			dim := field.Dim
			idx, err = index.NewSegmentedIndex(segments, func() (index.Index, error) {
				return index.NewHNSWIndex(dim, hnswConfig)
			})
			if err != nil {
				return fmt.Errorf("failed to create segmented HNSW index (%d segments): %w", segments, err)
			}
		} else {
			idx, err = index.NewHNSWIndex(field.Dim, config)
			if err != nil {
				return fmt.Errorf("failed to create HNSW index: %w", err)
			}
		}
	case IndexTypeFLAT:
		idx, err = index.NewFLATIndex(field.Dim, config)
		if err != nil {
			return fmt.Errorf("failed to create FLAT index: %w", err)
		}

	default:
		// Outside the vocabulary, or inside it but sparse-only.
		if err := validateFieldIndexType(field); err != nil {
			return err
		}
		// In IndexTypes and dense, yet no constructor above: the schema would
		// validate and journal, then fail to rebuild on replay.
		return fmt.Errorf("%w: no dense index constructor for %s", ErrInvalidArgument, field.Index.Type)
	}

	c.indexes[field.Name] = idx
	return nil
}

// createSparseIndex creates a sparse vector index (Inverted).
func (c *Collection) createSparseIndex(field VectorField) error {
	if err := validateFieldIndexType(field); err != nil {
		return err
	}

	// Extract BM25 parameters
	k1 := float32(1.2)
	b := float32(0.75)
	if field.Index.Params != nil {
		if val, ok := field.Index.Params["k1"].(float64); ok {
			k1 = float32(val)
		}
		if val, ok := field.Index.Params["b"].(float64); ok {
			b = float32(val)
		}
	}

	idx := sparse.NewInvertedIndex(field.Dim)
	idx.SetBM25Params(k1, b)
	c.sparse[field.Name] = idx
	return nil
}

// Add adds a document to the collection.
//
// The document must have vectors for all fields defined in the schema.
// Takes *Document so that assigned IDs are visible to the caller.
func (c *Collection) Add(ctx context.Context, doc *Document) error {
	if c.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
	if doc == nil {
		return fmt.Errorf("document cannot be nil")
	}
	c.mu.Lock()
	normalized, nextID, err := c.prepareDocumentsLockedVariant([]Document{*doc}, false)
	if err != nil {
		c.mu.Unlock()
		return err
	}
	c.reserveDocumentsLocked(normalized, nextID)
	c.markPendingCommitLocked(normalized)
	c.mu.Unlock()

	err = c.addPreparedToIndexes(ctx, normalized)
	c.mu.Lock()
	c.clearPendingCommitLocked(normalized)
	if err != nil {
		c.rollbackDocumentsLocked(ctx, normalized)
		c.mu.Unlock()
		return err
	}
	c.mu.Unlock()
	doc.ID = normalized[0].ID
	return nil
}

// normalizeDocumentVectorTypes checks that every vector field on doc matches
// its schema's vector kind and dimension. Vector is already typed by the time
// this runs (JSON decode produces Dense or Sparse directly), so this is a
// validation pass, not a conversion.
//
// Snapshot V2 load and durable-journal replay share this helper as their
// cold-start validation step.
func normalizeDocumentVectorTypes(doc *Document, schema *CollectionSchema) error {
	for fieldName, vector := range doc.Vectors {
		field := schema.GetField(fieldName)
		if field == nil {
			return fmt.Errorf("unknown vector field %s", fieldName)
		}
		switch field.Type {
		case VectorTypeDense:
			if vector.Dense == nil {
				return fmt.Errorf("field %s: expected dense vector", fieldName)
			}
			if len(vector.Dense) != field.Dim {
				return fmt.Errorf("field %s dimension mismatch: got %d, want %d", fieldName, len(vector.Dense), field.Dim)
			}
		case VectorTypeSparse:
			if vector.Sparse == nil {
				return fmt.Errorf("field %s: expected sparse vector", fieldName)
			}
			normalized, err := sparse.NewSparseVector(vector.Sparse.Indices, vector.Sparse.Values, vector.Sparse.Dim)
			if err != nil {
				return fmt.Errorf("field %s: %w", fieldName, err)
			}
			if normalized.Dim != field.Dim {
				return fmt.Errorf("field %s dimension mismatch: got %d, want %d", fieldName, normalized.Dim, field.Dim)
			}
			vector.Sparse = normalized
			doc.Vectors[fieldName] = vector
		default:
			return fmt.Errorf("unsupported vector type %s for field %s", field.Type, fieldName)
		}
	}
	return nil
}

// addToIndex adds a vector to the appropriate index.
func (c *Collection) addToIndex(ctx context.Context, field VectorField, docID uint64, vector Vector) error {
	switch field.Type {
	case VectorTypeDense:
		if vector.Dense == nil {
			return fmt.Errorf("invalid dense field %s: expected dense vector", field.Name)
		}

		idx, ok := c.indexes[field.Name]
		if !ok {
			return fmt.Errorf("index not found for field: %s", field.Name)
		}

		return idx.Add(ctx, docID, vector.Dense)

	case VectorTypeSparse:
		if vector.Sparse == nil {
			return fmt.Errorf("invalid sparse field %s: expected sparse vector", field.Name)
		}

		idx, ok := c.sparse[field.Name]
		if !ok {
			return fmt.Errorf("sparse index not found for field: %s", field.Name)
		}

		return idx.Add(ctx, docID, vector.Sparse)

	default:
		return fmt.Errorf("unsupported vector type: %d", field.Type)
	}
}

// setIndexMetadata sets metadata for a document ID in the appropriate index.
// This is used for filtered search support.
func (c *Collection) setIndexMetadata(field VectorField, docID uint64, metadata map[string]interface{}) error {
	switch field.Type {
	case VectorTypeDense:
		idx, ok := c.indexes[field.Name]
		if !ok {
			return fmt.Errorf("index not found for field: %s", field.Name)
		}

		// Check if the index has a SetMetadata method (via type assertion)
		type metadataSetter interface {
			SetMetadata(id uint64, metadata map[string]interface{}) error
		}

		if setter, ok := idx.(metadataSetter); ok {
			return setter.SetMetadata(docID, metadata)
		}

		// Index doesn't support metadata (e.g., legacy binary-backed indexes),
		// silently skip.
		return nil

	case VectorTypeSparse:
		// Sparse indexes don't currently support metadata filtering
		// Could be added in the future if needed
		return nil

	default:
		return fmt.Errorf("unsupported vector type: %d", field.Type)
	}
}

// Search performs a search across one or more vector fields.
//
// Every answering path funnels through here, so this is the one place
// that stamps QueryTimeMs: the wall time of the whole call, including
// both rungs of the fallback ladder.
func (c *Collection) Search(ctx context.Context, req SearchRequest) (resp *SearchResponse, err error) {
	start := time.Now()
	defer func() {
		if resp != nil {
			resp.QueryTimeMs = float64(time.Since(start).Nanoseconds()) / float64(time.Millisecond)
		}
	}()

	c.mu.RLock()
	defer c.mu.RUnlock()

	if req.CollectionName != c.schema.Name {
		return nil, fmt.Errorf("collection mismatch: expected %s, got %s", c.schema.Name, req.CollectionName)
	}
	if len(req.Queries) == 0 {
		return nil, fmt.Errorf("%w: at least one query field is required", ErrInvalidArgument)
	}
	if len(req.Queries) > MaxSearchFields {
		return nil, fmt.Errorf("%w: at most %d query fields are supported", ErrInvalidArgument, MaxSearchFields)
	}
	if req.TopK <= 0 || req.TopK > MaxSearchTopK {
		return nil, fmt.Errorf("%w: top_k must be in [1, %d]", ErrInvalidArgument, MaxSearchTopK)
	}
	if req.EfSearch < 0 || req.EfSearch > MaxSearchEf {
		return nil, fmt.Errorf("%w: ef_search must be in [0, %d]", ErrInvalidArgument, MaxSearchEf)
	}
	if req.HybridParams != nil {
		if err := validateHybridSearchParams(req.Queries, req.HybridParams); err != nil {
			return nil, err
		}
	}
	if math.IsNaN(req.ScoreFloor) || req.ScoreFloor < 0 {
		return nil, fmt.Errorf("%w: score_floor must be a finite value >= 0", ErrInvalidSearchArgument)
	}
	if math.IsNaN(req.UsageBoost) || req.UsageBoost < 0 || req.UsageBoost >= 1 {
		return nil, fmt.Errorf("%w: usage_boost must be in [0, 1)", ErrInvalidSearchArgument)
	}
	if req.Fallback != nil {
		if req.HybridParams != nil {
			return nil, fmt.Errorf("%w: fallback and hybrid_params are mutually exclusive", ErrInvalidSearchArgument)
		}
		if err := validateFallbackParams(req.Queries, req.Fallback); err != nil {
			return nil, err
		}
	}

	// Parse metadata filters if provided
	var metadataFilter filter.Filter
	if req.Filters != nil && len(req.Filters) > 0 {
		var err error
		metadataFilter, err = filter.FromMap(req.Filters)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid filter: %v", ErrInvalidArgument, err)
		}
	}

	includeVectors := false
	if req.IncludeVectors != nil {
		includeVectors = *req.IncludeVectors
	}
	if err := validateCanonicalSearchResponseBudget(c.schema, req.TopK, includeVectors); err != nil {
		return nil, err
	}

	// Resolve ef_search: request override > server default
	efSearch := c.defaultEfSearch
	if req.EfSearch > 0 {
		efSearch = req.EfSearch
	}

	// Auto-fallback ladder: exactly two query fields, primary first, then
	// secondary if the primary answer is weak (zero hits — including hits
	// wiped out by ScoreFloor — or best raw score worse than Threshold when
	// one is set). The floor is applied to the primary answer before the
	// decision, so "no confident result" always routes to the secondary.
	if req.Fallback != nil {
		primaryField := req.Fallback.Primary
		primaryResp, err := c.searchSingleField(ctx, primaryField, req.Queries[primaryField], req.TopK, efSearch, includeVectors, req.UsageBoost, metadataFilter)
		if err != nil {
			return nil, err
		}
		primaryLower := c.scoreLowerIsBetter(primaryField)
		c.finalizeSearch(primaryResp, req, primaryLower)
		needsFallback := len(primaryResp.Documents) == 0
		if !needsFallback && req.Fallback.Threshold > 0 && primaryQualityWorse(float64(primaryResp.BestScore), req.Fallback.Threshold, primaryLower) {
			needsFallback = true
		}
		if !needsFallback {
			c.recordSearchUsage(primaryResp)
			return primaryResp, nil
		}

		secondaryField := req.Fallback.Secondary
		secondaryResp, err := c.searchSingleField(ctx, secondaryField, req.Queries[secondaryField], req.TopK, efSearch, includeVectors, req.UsageBoost, metadataFilter)
		if err != nil {
			return nil, err
		}
		c.finalizeSearch(secondaryResp, req, c.scoreLowerIsBetter(secondaryField))
		secondaryResp.FellBackTo = secondaryField
		secondaryResp.CandidatesExamined += primaryResp.CandidatesExamined
		c.recordSearchUsage(secondaryResp)
		return secondaryResp, nil
	}

	// Single-field search
	if len(req.Queries) == 1 {
		for fieldName, queryVec := range req.Queries {
			resp, err := c.searchSingleField(ctx, fieldName, queryVec, req.TopK, efSearch, includeVectors, req.UsageBoost, metadataFilter)
			if err != nil {
				return nil, err
			}
			c.finalizeSearch(resp, req, c.scoreLowerIsBetter(fieldName))
			c.recordSearchUsage(resp)
			return resp, nil
		}
	}

	// Multi-field hybrid search
	if req.HybridParams != nil {
		resp, err := c.searchHybrid(ctx, req, efSearch, includeVectors, metadataFilter)
		if err != nil {
			return nil, err
		}
		// Fused hybrid scores are contributions (higher is better).
		c.finalizeSearch(resp, req, false)
		c.recordSearchUsage(resp)
		return resp, nil
	}

	// Multiple fields without fusion (return error)
	return nil, fmt.Errorf("%w: multiple query fields require hybrid_params or fallback", ErrInvalidSearchArgument)
}

// scoreLowerIsBetter reports whether the field's raw scores are
// distances (smaller is better), the dense-field convention, as opposed
// to BM25 and fused hybrid scores where higher is better. Unknown fields
// default to the higher-is-better convention (they cannot score anyway).
func (c *Collection) scoreLowerIsBetter(fieldName string) bool {
	field := c.schema.GetField(fieldName)
	return field != nil && field.Type == VectorTypeDense
}

// primaryQualityWorse reports whether the primary field's best score is
// worse than the fallback threshold, in the field's score direction.
func primaryQualityWorse(best, threshold float64, lowerIsBetter bool) bool {
	if lowerIsBetter {
		return best > threshold
	}
	return best < threshold
}

// validateFallbackParams checks the auto-fallback contract: the two named
// query fields are distinct, non-empty, threshold finite, and both present
// in the request's query map.
func validateFallbackParams(queries map[string]interface{}, fb *FallbackParams) error {
	// Both fields must be present and differ, and the collection caps a
	// request at MaxSearchFields (2) query fields, so a valid
	// fallback request is exactly the two named fields.
	if fb.Primary == "" || fb.Secondary == "" {
		return fmt.Errorf("%w: fallback requires non-empty primary and secondary fields", ErrInvalidSearchArgument)
	}
	if fb.Primary == fb.Secondary {
		return fmt.Errorf("%w: fallback primary and secondary must differ", ErrInvalidSearchArgument)
	}
	if math.IsNaN(fb.Threshold) || fb.Threshold < 0 {
		return fmt.Errorf("%w: fallback threshold must be a finite value >= 0", ErrInvalidSearchArgument)
	}
	if _, ok := queries[fb.Primary]; !ok {
		return fmt.Errorf("%w: fallback primary field %q is not in queries", ErrInvalidSearchArgument, fb.Primary)
	}
	if _, ok := queries[fb.Secondary]; !ok {
		return fmt.Errorf("%w: fallback secondary field %q is not in queries", ErrInvalidSearchArgument, fb.Secondary)
	}
	return nil
}

// finalizeSearch applies the response-level retrieval contract: the
// confidence floor (drops documents whose returned score is worse than it,
// in the field's score direction), then computes BestScore (the best raw
// score among the surviving documents, 0 when none survive) and WeakMatch.
// It does not record usage, so the caller records only the response it
// actually returns.
func (c *Collection) finalizeSearch(resp *SearchResponse, req SearchRequest, lowerIsBetter bool) {
	if resp == nil {
		return
	}
	if req.ScoreFloor > 0 {
		keep := 0
		for i := range resp.Documents {
			if !primaryQualityWorse(float64(resp.Scores[i]), req.ScoreFloor, lowerIsBetter) {
				resp.Documents[keep] = resp.Documents[i]
				resp.Scores[keep] = resp.Scores[i]
				keep++
			}
		}
		resp.Documents = resp.Documents[:keep]
		resp.Scores = resp.Scores[:keep]
	}
	var best float32
	for i, s := range resp.Scores {
		// Keep s when it is not worse than the current best.
		if i == 0 || !primaryQualityWorse(float64(s), float64(best), lowerIsBetter) {
			best = s
		}
	}
	resp.BestScore = best
	resp.WeakMatch = req.ScoreFloor > 0 && len(resp.Documents) == 0
	resp.ScoreDirection = ScoreDirectionHigherIsBetter
	if lowerIsBetter {
		resp.ScoreDirection = ScoreDirectionLowerIsBetter
	}
}

// recordSearchUsage feeds the non-durable frecency signal with the
// documents actually returned to the caller.
func (c *Collection) recordSearchUsage(resp *SearchResponse) {
	if c == nil || c.usage == nil || resp == nil {
		return
	}
	for _, d := range resp.Documents {
		c.usage.Record(d.ID)
	}
}

// queryDenseVector extracts a dense query vector from a SearchRequest.Queries
// entry. Unlike Document.Vectors, SearchRequest.Queries stays interface{}-typed
// (it is unrelated to the Document.Vectors wire format): direct Go API callers
// pass a raw []float32, HTTP/gRPC decode now passes a Vector, and callers who
// json.Unmarshal straight into SearchRequest get the classic JSON-generic
// []float64/[]interface{} shapes, so all of those must still be accepted.
func queryDenseVector(queryVec interface{}) ([]float32, error) {
	switch v := queryVec.(type) {
	case []float32:
		return v, nil
	case Vector:
		if v.Dense != nil {
			return v.Dense, nil
		}
	case []float64:
		out := make([]float32, len(v))
		for i, value := range v {
			out[i] = float32(value)
		}
		return out, nil
	case []interface{}:
		out := make([]float32, len(v))
		for i, value := range v {
			switch n := value.(type) {
			case float64:
				out[i] = float32(n)
			case float32:
				out[i] = n
			case int:
				out[i] = float32(n)
			default:
				return nil, fmt.Errorf("invalid dense vector element type %T", value)
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("expected dense vector, got %T", queryVec)
}

// querySparseVector is the sparse counterpart of queryDenseVector.
func querySparseVector(queryVec interface{}) (*sparse.SparseVector, error) {
	switch v := queryVec.(type) {
	case *sparse.SparseVector:
		return v, nil
	case Vector:
		if v.Sparse != nil {
			return v.Sparse, nil
		}
	case map[string]interface{}:
		indices, err := coerceUint32Slice(v["indices"])
		if err != nil {
			return nil, fmt.Errorf("invalid sparse indices: %w", err)
		}
		values, err := coerceFloat32Slice(v["values"])
		if err != nil {
			return nil, fmt.Errorf("invalid sparse values: %w", err)
		}
		dim, err := coerceInt(v["dim"])
		if err != nil {
			return nil, fmt.Errorf("invalid sparse dimension: %w", err)
		}
		return sparse.NewSparseVector(indices, values, dim)
	}
	return nil, fmt.Errorf("expected *SparseVector, got %T", queryVec)
}

// coerceUint32Slice and coerceFloat32Slice/coerceInt below back
// querySparseVector's map[string]interface{} case: a JSON-decoded sparse
// query (e.g. json.Unmarshal straight into SearchRequest.Queries) arrives as
// a generic map with []interface{} index/value arrays and a float64 dim.

func coerceUint32Slice(value interface{}) ([]uint32, error) {
	switch v := value.(type) {
	case nil:
		return []uint32{}, nil
	case []uint32:
		out := make([]uint32, len(v))
		copy(out, v)
		return out, nil
	case []interface{}:
		out := make([]uint32, len(v))
		for i, item := range v {
			switch n := item.(type) {
			case float64:
				out[i] = uint32(n)
			case float32:
				out[i] = uint32(n)
			case int:
				out[i] = uint32(n)
			default:
				return nil, fmt.Errorf("invalid uint32 slice element type %T", item)
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected []uint32-compatible value, got %T", value)
	}
}

func coerceFloat32Slice(value interface{}) ([]float32, error) {
	switch v := value.(type) {
	case nil:
		return []float32{}, nil
	case []float32:
		out := make([]float32, len(v))
		copy(out, v)
		return out, nil
	case []float64:
		out := make([]float32, len(v))
		for i, item := range v {
			out[i] = float32(item)
		}
		return out, nil
	case []interface{}:
		out := make([]float32, len(v))
		for i, item := range v {
			switch n := item.(type) {
			case float64:
				out[i] = float32(n)
			case float32:
				out[i] = n
			case int:
				out[i] = float32(n)
			default:
				return nil, fmt.Errorf("invalid float32 slice element type %T", item)
			}
		}
		return out, nil
	default:
		return nil, fmt.Errorf("expected []float32-compatible value, got %T", value)
	}
}

func coerceInt(value interface{}) (int, error) {
	switch v := value.(type) {
	case int:
		return v, nil
	case float64:
		return int(v), nil
	case float32:
		return int(v), nil
	default:
		return 0, fmt.Errorf("expected int-compatible value, got %T", value)
	}
}

// searchSingleField performs a search on a single vector field.
// usageBlend is the opt-in frecency weight applied to the raw ranking
// (0 keeps the index order exactly). Response post-processing (floor,
// weak-match, usage recording) is applied by the caller via
// finalizeSearch so fallback can inspect the raw primary answer first.
func (c *Collection) searchSingleField(ctx context.Context, fieldName string, queryVec interface{}, k int, efSearch int, includeVectors bool, usageBlend float64, metadataFilter filter.Filter) (*SearchResponse, error) {
	field := c.schema.GetField(fieldName)
	if field == nil {
		return nil, fmt.Errorf("field not found: %s", fieldName)
	}

	var results []hybrid.SearchResult

	switch field.Type {
	case VectorTypeDense:
		denseQuery, err := queryDenseVector(queryVec)
		if err != nil {
			return nil, fmt.Errorf("invalid dense query for %s: %w", fieldName, err)
		}

		idx, ok := c.indexes[fieldName]
		if !ok {
			return nil, fmt.Errorf("index not found for field: %s", fieldName)
		}

		// Create search params with filter based on index type
		var params index.SearchParams
		switch field.Index.Type {
		case IndexTypeHNSW:
			params = index.HNSWSearchParams{
				EfSearch: efSearch,
				Filter:   metadataFilter,
			}
		default:
			// For other index types (FLAT), use HNSW params as fallback
			params = index.HNSWSearchParams{
				Filter: metadataFilter,
			}
		}

		idxResults, err := idx.Search(ctx, denseQuery, k, params)
		if err != nil {
			return nil, err
		}

		// Convert to SearchResult format
		results = make([]hybrid.SearchResult, len(idxResults))
		for i, r := range idxResults {
			results[i] = hybrid.SearchResult{
				DocID: r.ID,
				Score: r.Distance,
			}
		}

	case VectorTypeSparse:
		sparseQuery, err := querySparseVector(queryVec)
		if err != nil {
			return nil, fmt.Errorf("invalid sparse query for %s: %w", fieldName, err)
		}

		idx, ok := c.sparse[fieldName]
		if !ok {
			return nil, fmt.Errorf("sparse index not found for field: %s", fieldName)
		}

		sparseResults, err := idx.Search(ctx, sparseQuery, k)
		if err != nil {
			return nil, err
		}

		// Convert sparse.SearchResult to hybrid.SearchResult
		results = make([]hybrid.SearchResult, len(sparseResults))
		for i, r := range sparseResults {
			results[i] = hybrid.SearchResult{
				DocID: r.DocID,
				Score: r.Score,
			}
		}

	default:
		return nil, fmt.Errorf("unsupported vector type: %d", field.Type)
	}

	// Frecency re-order (opt-in). Ranking only: raw scores are preserved in
	// the response and the usage blend is bounded to < 1 by Search, so the
	// similarity signal always dominates.
	if usageBlend > 0 {
		rankByUsage(results, c.usage, usageBlend, field.Type == VectorTypeDense)
	}

	if err := c.validateSearchResultsBudget(results, includeVectors); err != nil {
		return nil, err
	}

	// Retrieve documents in a single tight loop for cache-friendly access.
	// Pre-allocate both slices at once to reduce allocator pressure.
	n := len(results)
	docs := make([]Document, n)
	scores := make([]float32, n)
	for i := 0; i < n; i++ {
		scores[i] = results[i].Score
		if doc, ok := c.documents[results[i].DocID]; ok {
			docs[i] = cloneDocumentForSearch(*doc, includeVectors)
		}
	}

	return &SearchResponse{
		Documents:          docs,
		Scores:             scores,
		CandidatesExamined: n,
	}, nil
}

// searchHybrid performs hybrid search across multiple vector fields.
func (c *Collection) searchHybrid(ctx context.Context, req SearchRequest, efSearch int, includeVectors bool, metadataFilter filter.Filter) (*SearchResponse, error) {
	if len(req.Queries) != 2 {
		return nil, fmt.Errorf("hybrid search currently supports exactly 2 fields")
	}

	// Identify dense and sparse fields
	var denseField, sparseField string
	var denseFieldConfig VectorField
	var denseQuery []float32
	var sparseQuery *sparse.SparseVector

	for fieldName, queryVec := range req.Queries {
		field := c.schema.GetField(fieldName)
		if field == nil {
			return nil, fmt.Errorf("field not found: %s", fieldName)
		}

		switch field.Type {
		case VectorTypeDense:
			denseField = fieldName
			denseFieldConfig = *field
			var err error
			denseQuery, err = queryDenseVector(queryVec)
			if err != nil {
				return nil, fmt.Errorf("invalid dense query for %s: %w", fieldName, err)
			}
		case VectorTypeSparse:
			sparseField = fieldName
			var err error
			sparseQuery, err = querySparseVector(queryVec)
			if err != nil {
				return nil, fmt.Errorf("invalid sparse query for %s: %w", fieldName, err)
			}
		}
	}

	// Search dense index
	var denseResults []hybrid.SearchResult
	if denseField != "" {
		idx := c.indexes[denseField]

		// Create search params with filter based on index type
		var params index.SearchParams
		switch denseFieldConfig.Index.Type {
		case IndexTypeHNSW:
			params = index.HNSWSearchParams{
				EfSearch: efSearch,
				Filter:   metadataFilter,
			}
		default:
			params = index.HNSWSearchParams{
				Filter: metadataFilter,
			}
		}

		idxResults, err := idx.Search(ctx, denseQuery, req.TopK*2, params) // Fetch more for fusion
		if err != nil {
			return nil, fmt.Errorf("dense search failed: %w", err)
		}

		denseResults = make([]hybrid.SearchResult, len(idxResults))
		for i, r := range idxResults {
			// Use similarity (higher is better) so weighted/linear fusion ranks
			// dense hits correctly alongside sparse similarity scores. RRF is
			// rank-only and unaffected.
			denseResults[i] = hybrid.SearchResult{
				DocID: r.ID,
				Score: r.Score,
			}
		}
	}

	// Search sparse index
	var sparseResults []hybrid.SearchResult
	if sparseField != "" {
		idx := c.sparse[sparseField]
		sparseRes, err := idx.Search(ctx, sparseQuery, req.TopK*2)
		if err != nil {
			return nil, fmt.Errorf("sparse search failed: %w", err)
		}

		// Convert sparse.SearchResult to hybrid.SearchResult
		sparseResults = make([]hybrid.SearchResult, len(sparseRes))
		for i, r := range sparseRes {
			sparseResults[i] = hybrid.SearchResult{
				DocID: r.DocID,
				Score: r.Score,
			}
		}
	}

	// Fuse results
	fusionParams := hybrid.FusionParams{
		Strategy:     hybrid.FusionRRF,
		K:            60.0,
		DenseWeight:  0.7,
		SparseWeight: 0.3,
	}

	// Override with request params if provided
	if req.HybridParams != nil {
		if req.HybridParams.Strategy == "weighted" {
			fusionParams.Strategy = hybrid.FusionWeighted
		} else if req.HybridParams.Strategy == "linear" {
			fusionParams.Strategy = hybrid.FusionLinear
		}

		if weights := req.HybridParams.Weights; weights != nil {
			if dw, ok := weights[denseField]; ok {
				fusionParams.DenseWeight = dw
			} else if dw, ok := weights["dense"]; ok {
				fusionParams.DenseWeight = dw
			}
			if sw, ok := weights[sparseField]; ok {
				fusionParams.SparseWeight = sw
			} else if sw, ok := weights["sparse"]; ok {
				fusionParams.SparseWeight = sw
			}
		}

		if req.HybridParams.RRFConstant > 0 {
			fusionParams.K = req.HybridParams.RRFConstant
		}
	}

	fusedResults, err := hybrid.HybridSearch(denseResults, sparseResults, fusionParams, req.TopK)
	if err != nil {
		return nil, fmt.Errorf("fusion failed: %w", err)
	}

	// Frecency re-order (opt-in), same bounded semantics as the single
	// field path: ranking only, raw fusion scores preserved.
	if req.UsageBoost > 0 {
		rankByUsage(fusedResults, c.usage, req.UsageBoost, false) // fused scores: higher is better
	}

	if err := c.validateSearchResultsBudget(fusedResults, includeVectors); err != nil {
		return nil, err
	}

	// Retrieve documents (lightweight copy: skip vectors unless requested)
	docs := make([]Document, len(fusedResults))
	scores := make([]float32, len(fusedResults))
	for i, r := range fusedResults {
		if doc, ok := c.documents[r.DocID]; ok {
			docs[i] = cloneDocumentForSearch(*doc, includeVectors)
		}
		scores[i] = r.Score
	}

	return &SearchResponse{
		Documents:          docs,
		Scores:             scores,
		CandidatesExamined: len(denseResults) + len(sparseResults),
	}, nil
}

func validateHybridSearchParams(queries map[string]interface{}, params *HybridSearchParams) error {
	switch params.Strategy {
	case "rrf", "weighted", "linear":
	default:
		return fmt.Errorf("%w: invalid hybrid strategy %q", ErrInvalidArgument, params.Strategy)
	}
	if math.IsNaN(float64(params.RRFConstant)) || math.IsInf(float64(params.RRFConstant), 0) || params.RRFConstant < 0 {
		return fmt.Errorf("%w: hybrid rrf_constant must be finite and non-negative", ErrInvalidArgument)
	}
	var weightSum float32
	for field, weight := range params.Weights {
		if _, ok := queries[field]; !ok && field != "dense" && field != "sparse" {
			return fmt.Errorf("%w: hybrid weight references unknown query field %q", ErrInvalidArgument, field)
		}
		if math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) || weight < 0 {
			return fmt.Errorf("%w: hybrid weight for %q must be finite and non-negative", ErrInvalidArgument, field)
		}
		weightSum += weight
	}
	if len(params.Weights) > 0 && weightSum <= 0 {
		return fmt.Errorf("%w: hybrid weights must contain a positive value", ErrInvalidArgument)
	}
	return nil
}

// BatchAdd adds multiple documents to the collection.
// When the underlying index implements index.BatchAdder, vectors are inserted
// in a single batch call (one lock cycle) instead of per-document.
func (c *Collection) BatchAdd(ctx context.Context, docs []Document) error {
	if c.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
	c.mu.Lock()
	normalized, nextID, err := c.prepareDocumentsLockedVariant(docs, false)
	if err != nil {
		c.mu.Unlock()
		return err
	}
	c.reserveDocumentsLocked(normalized, nextID)
	c.markPendingCommitLocked(normalized)
	c.mu.Unlock()

	err = c.addPreparedToIndexes(ctx, normalized)
	c.mu.Lock()
	c.clearPendingCommitLocked(normalized)
	if err != nil {
		c.rollbackDocumentsLocked(ctx, normalized)
		c.mu.Unlock()
		return err
	}
	c.mu.Unlock()
	for i := range docs {
		docs[i].ID = normalized[i].ID
	}
	return nil
}

// prepareCanonicalDocuments validates and deep-clones documents without
// changing collection or caller-owned state. IDs are resolved before WAL
// append, including explicit IDs, so replay cannot make a different choice.
//
// Vector fields are typed (Dense []float32 or Sparse *sparse.SparseVector),
// so index insertion uses the caller's dense slice directly with no
// unboxing. Deep-copy isolation is unchanged: cloneDocumentPreservingTypes
// copies every reachable slice, map, and pointer, so later caller mutations
// cannot reach stored state.
func (c *Collection) prepareCanonicalDocuments(docs []Document) ([]Document, uint64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.prepareDocumentsLockedVariant(docs, false)
}

func (c *Collection) prepareDocumentsLockedVariant(docs []Document, allowReplacement bool) ([]Document, uint64, error) {
	if len(docs) == 0 {
		return nil, c.nextID, fmt.Errorf("documents cannot be empty")
	}
	nextID := c.nextID
	if nextID == 0 {
		nextID = 1
	}
	reserved := make(map[uint64]struct{}, len(docs))
	normalized := make([]Document, len(docs))
	for i := range docs {
		clone := cloneDocumentPreservingTypes(docs[i])
		var err error
		if clone.ID == 0 {
			clone.ID, nextID, err = nextCanonicalID(nextID, reserved, c.documents)
			if err != nil {
				return nil, c.nextID, fmt.Errorf("document %d ID assignment failed: %w", i, err)
			}
		} else {
			if clone.ID == math.MaxUint64 {
				return nil, c.nextID, fmt.Errorf("document %d ID cannot be maximum uint64", i)
			}
			if _, exists := reserved[clone.ID]; exists {
				return nil, c.nextID, fmt.Errorf("document %d duplicates ID %d in batch", i, clone.ID)
			}
			if _, exists := c.documents[clone.ID]; exists && !allowReplacement {
				return nil, c.nextID, fmt.Errorf("%w: document %d ID %d", ErrDocumentExists, i, clone.ID)
			}
			if clone.ID >= nextID {
				nextID = clone.ID + 1
			}
		}
		reserved[clone.ID] = struct{}{}
		if err := clone.Validate(&c.schema); err != nil {
			return nil, c.nextID, fmt.Errorf("document %d validation failed: %w", i, err)
		}
		if err := validatePersistedDocument(&clone, &c.schema); err != nil {
			return nil, c.nextID, fmt.Errorf("document %d vector validation failed: %w", i, err)
		}
		normalized[i] = clone
	}
	return normalized, nextID, nil
}

// prepareCanonicalUpsert validates and deep-clones a single upsert document
// without changing collection or caller-owned state. In contrast with insert
// preparation, an existing live ID is accepted: the caller explicitly chose
// it for replacement. IDs are not auto-assigned here; upsert requires a
// caller-supplied nonzero ID.
func (c *Collection) prepareCanonicalUpsert(docs []Document) ([]Document, uint64, error) {
	if len(docs) != 1 {
		return nil, c.nextID, fmt.Errorf("upsert requires exactly one document")
	}
	if docs[0].ID == 0 {
		return nil, c.nextID, fmt.Errorf("upsert requires a caller-supplied document ID")
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.prepareDocumentsLockedVariant(docs, true)
}

func cloneDocumentPreservingTypes(doc Document) Document {
	clone := Document{ID: doc.ID}
	if doc.Vectors != nil {
		clone.Vectors = make(map[string]Vector, len(doc.Vectors))
		for key, value := range doc.Vectors {
			clone.Vectors[key] = value.Clone()
		}
	}
	if doc.Metadata != nil {
		clone.Metadata = make(map[string]interface{}, len(doc.Metadata))
		for key, value := range doc.Metadata {
			clone.Metadata[key] = cloneDocumentValue(value)
		}
	}
	return clone
}

func cloneDocumentForSearch(doc Document, includeVectors bool) Document {
	if includeVectors {
		return cloneDocumentPreservingTypes(doc)
	}
	clone := Document{ID: doc.ID}
	if doc.Metadata != nil {
		clone.Metadata = make(map[string]interface{}, len(doc.Metadata))
		for key, value := range doc.Metadata {
			clone.Metadata[key] = cloneDocumentValue(value)
		}
	}
	return clone
}

func (c *Collection) validateSearchResultsBudget(results []hybrid.SearchResult, includeVectors bool) error {
	perDocument, err := canonicalSearchResponseBytesPerDocument(c.schema, includeVectors)
	if err != nil {
		return err
	}
	var estimated int64
	for _, result := range results {
		resultBytes := perDocument
		if doc := c.documents[result.DocID]; doc != nil && doc.Metadata != nil {
			metadata, err := json.Marshal(doc.Metadata)
			if err != nil {
				return fmt.Errorf("encode document %d metadata for response budget: %w", result.DocID, err)
			}
			resultBytes += int64(len(metadata))
		}
		if resultBytes > int64(MaxSearchResponseBytes)-estimated {
			return fmt.Errorf(
				"%w: estimated response exceeds %d bytes",
				ErrSearchResponseBudgetExceeded,
				MaxSearchResponseBytes,
			)
		}
		estimated += resultBytes
	}
	return nil
}

func cloneDocumentValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case []float32:
		return append([]float32(nil), typed...)
	case []float64:
		return append([]float64(nil), typed...)
	case []uint32:
		return append([]uint32(nil), typed...)
	case []interface{}:
		clone := make([]interface{}, len(typed))
		for i := range typed {
			clone[i] = cloneDocumentValue(typed[i])
		}
		return clone
	case map[string]interface{}:
		clone := make(map[string]interface{}, len(typed))
		for key, item := range typed {
			clone[key] = cloneDocumentValue(item)
		}
		return clone
	case *sparse.SparseVector:
		if typed == nil {
			return (*sparse.SparseVector)(nil)
		}
		return &sparse.SparseVector{
			Indices: append([]uint32(nil), typed.Indices...),
			Values:  append([]float32(nil), typed.Values...),
			Dim:     typed.Dim,
		}
	default:
		// Anything outside the vector vocabulary (e.g. []string or
		// map[string]string metadata) must still be isolated: share only
		// immutable leaves and copy every reachable composite.
		return cloneCompositeDocumentValue(value)
	}
}

// cloneCompositeDocumentValue deep-copies slice, array, map, and pointer
// values that the typed cases above do not cover, so caller mutations after
// insertion can never alias stored state. Scalar leaves (numbers, strings,
// bools, nil) are immutable from the store's side and shared as-is; structs
// reached through pointers are copied as opaque values, which is safe because
// mutation through them would require exporting a reference interior that
// JSON-shaped callers cannot construct.
func cloneCompositeDocumentValue(value interface{}) interface{} {
	rv := reflect.ValueOf(value)
	if !rv.IsValid() {
		return value
	}
	cloned := cloneReflectValue(rv)
	if !cloned.IsValid() {
		return value
	}
	return cloned.Interface()
}

func cloneReflectValue(rv reflect.Value) reflect.Value {
	switch rv.Kind() {
	case reflect.Slice:
		if rv.IsNil() {
			return rv
		}
		out := reflect.MakeSlice(rv.Type(), rv.Len(), rv.Len())
		for i := 0; i < rv.Len(); i++ {
			out.Index(i).Set(cloneReflectValue(rv.Index(i)))
		}
		return out
	case reflect.Array:
		out := reflect.New(rv.Type()).Elem()
		for i := 0; i < rv.Len(); i++ {
			out.Index(i).Set(cloneReflectValue(rv.Index(i)))
		}
		return out
	case reflect.Map:
		if rv.IsNil() {
			return rv
		}
		out := reflect.MakeMapWithSize(rv.Type(), rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			// Keys stay as-is: they are comparable values, and cloning a key
			// could break identity-based lookups on pointer-keyed maps.
			out.SetMapIndex(iter.Key(), cloneReflectValue(iter.Value()))
		}
		return out
	case reflect.Ptr:
		if rv.IsNil() {
			return rv
		}
		out := reflect.New(rv.Type().Elem())
		out.Elem().Set(cloneReflectValue(rv.Elem()))
		return out
	default:
		return rv
	}
}

func (c *Collection) addPreparedDocuments(ctx context.Context, docs []Document, nextID uint64) error {
	return c.commitPrepared(ctx, docs, nextID)
}

// reserveDocumentsLocked stores docs under their prepared IDs and advances
// the ID cursor. The caller must hold c.mu.Lock().
func (c *Collection) reserveDocumentsLocked(docs []Document, nextID uint64) {
	for i := range docs {
		c.documents[docs[i].ID] = &docs[i]
	}
	c.nextID = nextID
}

// markPendingCommitLocked registers docs as reserved-but-not-yet-indexed so
// deleteDocumentDirect waits for their index insert instead of racing it.
// The caller must hold c.mu.Lock().
func (c *Collection) markPendingCommitLocked(docs []Document) {
	for i := range docs {
		c.pendingCommit[docs[i].ID] = struct{}{}
	}
}

// clearPendingCommitLocked un-registers docs (their index insert has
// finished, successfully or not) and wakes any deleteDocumentDirect callers
// waiting on one of these IDs. The caller must hold c.mu.Lock().
func (c *Collection) clearPendingCommitLocked(docs []Document) {
	for i := range docs {
		delete(c.pendingCommit, docs[i].ID)
	}
	c.commitCond.Broadcast()
}

// waitPendingCommitLocked blocks until docID has no in-flight index insert,
// so the caller's idx.Delete sees either a fully indexed ID or a genuinely
// absent one. The caller must hold c.mu.Lock(); Wait releases and reacquires
// it, so re-read any state derived from c.documents after this returns.
func (c *Collection) waitPendingCommitLocked(docID uint64) {
	for {
		if _, pending := c.pendingCommit[docID]; !pending {
			return
		}
		c.commitCond.Wait()
	}
}

// addPreparedToIndexes inserts every prepared document's vectors and
// metadata into the field indexes. It deliberately does NOT take c.mu: each
// index owns its own locking (internal/index/hnsw.go's writeMu/mu split lets
// its own Search proceed during this call), and c.indexes/c.sparse are
// populated once at construction and never mutated afterward (grep -n
// 'c.indexes\[' collection.go — the only write is in createDenseIndex,
// called from NewCollection), so reading them here without c.mu is safe.
func (c *Collection) addPreparedToIndexes(ctx context.Context, docs []Document) error {
	// For each field, collect vectors and batch-insert if possible.
	for _, field := range c.schema.Fields {
		if field.Type == VectorTypeDense {
			idx, ok := c.indexes[field.Name]
			if !ok {
				return fmt.Errorf("index not found for field: %s", field.Name)
			}

			// Try batch path
			if batcher, ok := idx.(index.BatchAdder); ok {
				batch := make(map[uint64][]float32, len(docs))
				for i := range docs {
					vec := docs[i].Vectors[field.Name]
					if vec.Dense == nil {
						return fmt.Errorf("doc %d field %s: expected dense vector, got %T", i, field.Name, vec)
					}
					batch[docs[i].ID] = vec.Dense
				}
				if err := batcher.BatchAdd(ctx, batch); err != nil {
					return fmt.Errorf("batch add to index %s: %w", field.Name, err)
				}
			} else {
				// Fallback: per-vector add.
				for i := range docs {
					vec := docs[i].Vectors[field.Name]
					if vec.Dense == nil {
						return fmt.Errorf("doc %d field %s: expected dense vector, got %T", i, field.Name, vec)
					}
					if err := idx.Add(ctx, docs[i].ID, vec.Dense); err != nil {
						return fmt.Errorf("doc %d add to index %s: %w", i, field.Name, err)
					}
				}
			}
		} else if field.Type == VectorTypeSparse {
			sparseIdx, ok := c.sparse[field.Name]
			if !ok {
				return fmt.Errorf("sparse index not found for field: %s", field.Name)
			}
			for i := range docs {
				vec := docs[i].Vectors[field.Name]
				if vec.Sparse == nil {
					return fmt.Errorf("doc %d field %s: expected sparse vector, got %T", i, field.Name, vec)
				}
				if err := sparseIdx.Add(ctx, docs[i].ID, vec.Sparse); err != nil {
					return fmt.Errorf("doc %d add to sparse index %s: %w", i, field.Name, err)
				}
			}
		}
	}

	// Phase 3: Set metadata.
	for i := range docs {
		if docs[i].Metadata != nil && len(docs[i].Metadata) > 0 {
			for _, field := range c.schema.Fields {
				if err := c.setIndexMetadata(field, docs[i].ID, docs[i].Metadata); err != nil {
					return fmt.Errorf("doc %d metadata for %s: %w", i, field.Name, err)
				}
			}
		}
	}
	return nil
}

// rollbackDocumentsLocked undoes a reservation whose index insert failed
// partway through: it deletes the reserved documents and best-effort deletes
// any postings addPreparedToIndexes may already have written for them
// (ignoring errors — a given doc may not have reached every field's index
// yet, and Delete on a not-found ID is harmless). The caller must hold
// c.mu.Lock().
//
// Before this rollback existed (addPreparedDocumentsLocked, pre-split), a
// partial index-insert failure left postings for the fields it did reach
// with no document stored to match them (a leak) and left nextID unchanged.
// This rollback is strictly better: no document is left half-indexed.
func (c *Collection) rollbackDocumentsLocked(ctx context.Context, docs []Document) {
	for i := range docs {
		id := docs[i].ID
		delete(c.documents, id)
		for _, idx := range c.indexes {
			_ = idx.Delete(ctx, id)
		}
		for _, sparseIdx := range c.sparse {
			_ = sparseIdx.Delete(ctx, id)
		}
	}
}

// commitPrepared reserves docs under c.mu, then inserts them into the field
// indexes WITHOUT c.mu, so a concurrent Search only waits for the brief
// reservation, not the whole index build. This changes observable semantics
// versus the old single-lock version, deliberately:
//   - Count/GetDocument can observe a document whose index insert is still
//     in flight (it is reserved in c.documents before addPreparedToIndexes
//     runs).
//   - Search cannot return it until the index actually has it; an index
//     insert error rolls the reservation back out.
//   - A Delete racing an in-flight batch on an ephemeral collection waits
//     on commitCond until the ID clears pendingCommit (below), then wins
//     either way (the doc and its postings end up gone). Without that wait,
//     Delete could reach an index before this call's insert for that ID even
//     started, get a spurious "not found", and abort with the document left
//     alive.
func (c *Collection) commitPrepared(ctx context.Context, docs []Document, nextID uint64) error {
	c.mu.Lock()
	c.reserveDocumentsLocked(docs, nextID)
	c.markPendingCommitLocked(docs)
	c.mu.Unlock()

	err := c.addPreparedToIndexes(ctx, docs)

	c.mu.Lock()
	c.clearPendingCommitLocked(docs)
	if err != nil {
		c.rollbackDocumentsLocked(ctx, docs)
	}
	c.mu.Unlock()
	return err
}

// Upsert inserts or replaces a single caller-addressed document. The new
// document is applied under the same write lock as an overwrite of any vector
// set the ID already owns: the old dense/sparse postings for the ID are
// removed first so the re-add cannot hit an index's live-ID rejection, then
// the new document is added through the normal prepared-add path. Unlike
// delete+insert, the ID counter placement is driven only by
// prepareCanonicalUpsert, so journal replay of an upsert is deterministic.
func (c *Collection) Upsert(ctx context.Context, doc *Document) error {
	if c.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
	if doc == nil {
		return fmt.Errorf("document cannot be nil")
	}
	normalized, nextID, err := c.prepareCanonicalUpsert([]Document{*doc})
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.upsertPreparedLocked(ctx, normalized, nextID)
}

// upsertPreparedLocked applies one or more prepared (canonicalized) documents
// as upserts. The caller must hold the write lock. Documents that already own
// live dense/sparse entries are first evicted from those indexes before the
// prepared add path reinserts them, which keeps index counts stable and lets
// HNSW resurrect instead of rejecting a duplicate live ID. nextID is the
// position prepareCanonicalUpsert finished its ID placement at and becomes the
// collection's new cursor.
func (c *Collection) upsertPreparedLocked(ctx context.Context, docs []Document, nextID uint64) error {
	// Evict existing live postings for any ID being replaced, then let
	// addPreparedToIndexes reinsert and re-register metadata in one pass.
	for i := range docs {
		docID := docs[i].ID
		// Same window as deleteDocumentDirect: a colliding explicit ID may be
		// reserved by an in-flight BatchAdd whose index insert has not landed.
		c.waitPendingCommitLocked(docID)
		if _, exists := c.documents[docID]; !exists {
			continue
		}
		for fieldName := range c.indexes {
			if err := c.indexes[fieldName].Delete(ctx, docID); err != nil {
				return fmt.Errorf("failed to delete from index %s: %w", fieldName, err)
			}
		}
		for fieldName := range c.sparse {
			if err := c.sparse[fieldName].Delete(ctx, docID); err != nil {
				return fmt.Errorf("failed to delete from sparse index %s: %w", fieldName, err)
			}
		}
	}
	preparedNextID := nextID
	if nextID == 0 {
		preparedNextID = c.nextID
	}
	// upsertPreparedLocked stays fully locked (the caller holds c.mu.Lock()
	// for its whole span), so index-insert-then-reserve here preserves the
	// pre-split addPreparedDocumentsLocked ordering and failure behavior
	// exactly: on an index error nothing is (re-)stored and nextID does not
	// advance. No rollback is needed for that reason.
	if err := c.addPreparedToIndexes(ctx, docs); err != nil {
		return err
	}
	c.reserveDocumentsLocked(docs, preparedNextID)
	return nil
}

// BulkAddDense inserts raw dense documents into a single field without full Document overhead.
// IDs and vectors must be the same length. Minimal Document records are created (ID only).
func (c *Collection) BulkAddDense(ctx context.Context, fieldName string, ids []uint64, vectors [][]float32) error {
	if c.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
	if len(ids) != len(vectors) {
		return fmt.Errorf("ids length %d != vectors length %d", len(ids), len(vectors))
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Validate field exists and is dense
	field := c.schema.GetField(fieldName)
	if field == nil {
		return fmt.Errorf("field not found: %s", fieldName)
	}
	if field.Type != VectorTypeDense {
		return fmt.Errorf("field %s is not dense", fieldName)
	}

	idx, ok := c.indexes[fieldName]
	if !ok {
		return fmt.Errorf("index not found for field: %s", fieldName)
	}

	// Validate dimensions
	for i, v := range vectors {
		if len(v) != field.Dim {
			return fmt.Errorf("vector %d dimension mismatch: expected %d, got %d", i, field.Dim, len(v))
		}
	}

	// Batch insert: prefer NoCopyBatchAdder (skips redundant vector copy)
	// since BulkAddDense callers (binary import) already provide fresh slices.
	if ncBatcher, ok := idx.(index.NoCopyBatchAdder); ok {
		batch := make(map[uint64][]float32, len(ids))
		for i, id := range ids {
			batch[id] = vectors[i]
		}
		if err := ncBatcher.BatchAddNoCopy(ctx, batch); err != nil {
			return fmt.Errorf("batch add to index %s: %w", fieldName, err)
		}
	} else if batcher, ok := idx.(index.BatchAdder); ok {
		batch := make(map[uint64][]float32, len(ids))
		for i, id := range ids {
			batch[id] = vectors[i]
		}
		if err := batcher.BatchAdd(ctx, batch); err != nil {
			return fmt.Errorf("batch add to index %s: %w", fieldName, err)
		}
	} else {
		for i, id := range ids {
			if err := idx.Add(ctx, id, vectors[i]); err != nil {
				return fmt.Errorf("vector %d add to index %s: %w", i, fieldName, err)
			}
		}
	}

	// Create or extend lightweight document records. Binary import is field-
	// oriented, so importing a second field for the same IDs must not discard the
	// vectors already attached to those documents.
	for i, id := range ids {
		doc := c.documents[id]
		if doc == nil {
			doc = &Document{ID: id, Vectors: make(map[string]Vector)}
			c.documents[id] = doc
		} else if doc.Vectors == nil {
			doc.Vectors = make(map[string]Vector)
		}
		doc.Vectors[fieldName] = Vector{Dense: vectors[i]}
		// Update nextID to stay ahead
		if id >= c.nextID {
			c.nextID = id + 1
		}
	}

	return nil
}

// GetDocument retrieves a document by ID.
func (c *Collection) GetDocument(docID uint64) (*Document, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	doc, ok := c.documents[docID]
	if !ok || doc == nil {
		return nil, false
	}
	clone := cloneDocumentPreservingTypes(*doc)
	// A direct fetch is the strongest "this tenant consumed this document"
	// signal, so it is recorded in the non-durable usage tracker too.
	if c.usage != nil {
		c.usage.Record(docID)
	}
	return &clone, true
}

// Delete removes a document from the collection.
func (c *Collection) Delete(ctx context.Context, docID uint64) error {
	if c.isDurableReadOnly() {
		return ErrCanonicalMutationRequired
	}
	return c.deleteDocumentDirect(ctx, docID)
}

func (c *Collection) deleteDocumentDirect(ctx context.Context, docID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if docID == 0 {
		return fmt.Errorf("document ID cannot be zero")
	}
	// A concurrent commitPrepared reserves the ID in c.documents before its
	// index inserts finish. Deleting here first would race an index that
	// hasn't received docID yet, get a spurious "not found", and abort
	// leaving the document alive (see commitPrepared). Wait for the ID to
	// clear pendingCommit so the indexes below are guaranteed to have it.
	c.waitPendingCommitLocked(docID)
	if _, exists := c.documents[docID]; !exists {
		return fmt.Errorf("%w: %d", ErrDocumentNotFound, docID)
	}

	// Remove from all indexes
	for fieldName := range c.indexes {
		idx := c.indexes[fieldName]
		if err := idx.Delete(ctx, docID); err != nil {
			return fmt.Errorf("failed to delete from index %s: %w", fieldName, err)
		}
	}

	for fieldName := range c.sparse {
		idx := c.sparse[fieldName]
		if err := idx.Delete(ctx, docID); err != nil {
			return fmt.Errorf("failed to delete from sparse index %s: %w", fieldName, err)
		}
	}

	// Remove from document storage
	delete(c.documents, docID)
	// The usage signal is keyed by document, so it dies with the document.
	// Nothing else drops an entry, so a retained one is unreclaimable: it is
	// persisted to the usage sidecar and restored on every open, growing the
	// tracker with cumulative deletes instead of the live set.
	c.usage.Forget(docID)

	return nil
}

func (c *Collection) setDurableReadOnly() {
	c.mu.Lock()
	c.durableReadOnly = true
	c.mu.Unlock()
}

func (c *Collection) isDurableReadOnly() bool {
	c.mu.RLock()
	readOnly := c.durableReadOnly
	c.mu.RUnlock()
	return readOnly
}

// Count returns the number of documents in the collection.
func (c *Collection) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.documents)
}

// documentBytesFor estimates one document's size without cloning it, for
// tenant usage bookkeeping paths that only need the number.
func (c *Collection) documentBytesFor(id uint64) (int64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	doc, ok := c.documents[id]
	if !ok || doc == nil {
		return 0, false
	}
	return documentBytes(doc), true
}

// documentBytesTotal sums documentBytes across every document in the
// collection, for tenant usage bookkeeping.
func (c *Collection) documentBytesTotal() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var total int64
	for _, doc := range c.documents {
		if doc == nil {
			continue
		}
		total += documentBytes(doc)
	}
	return total
}

// Schema returns the collection schema.
func (c *Collection) Schema() CollectionSchema {
	c.mu.RLock()
	defer c.mu.RUnlock()
	clone, err := cloneCanonicalSchema(c.schema)
	if err != nil {
		// The JSON round-trip failed (in practice: unserializable metadata),
		// so no live reference may escape under RLock. Return the structural
		// fields with a fresh Fields slice and drop the metadata map.
		schema := c.schema
		schema.Fields = append([]VectorField(nil), c.schema.Fields...)
		schema.Metadata = nil
		return schema
	}
	return clone
}

// isEphemeral reports the collection's durability class without cloning the
// whole schema, so the store can consult it on every document mutation.
func (c *Collection) isEphemeral() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.schema.Durability == DurabilityEphemeral
}

// UpdateMetadata updates the collection schema's metadata map.
func (c *Collection) UpdateMetadata(metadata map[string]interface{}) {
	if c.isDurableReadOnly() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.schema.Metadata == nil {
		c.schema.Metadata = make(map[string]interface{})
	}
	for k, v := range metadata {
		c.schema.Metadata[k] = v
	}
}

// Name returns the collection name.
func (c *Collection) Name() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.schema.Name
}

// ExportIndexes exports all dense index data as field -> serialized bytes.
func (c *Collection) ExportIndexes() (map[string][]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make(map[string][]byte, len(c.indexes))
	for name, idx := range c.indexes {
		data, err := idx.Export()
		if err != nil {
			return nil, fmt.Errorf("export index %s: %w", name, err)
		}
		result[name] = data
	}
	return result, nil
}

// ExportSparseIndexes exports all sparse index data as field -> serialized bytes.
func (c *Collection) ExportSparseIndexes() (map[string][]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.sparse) == 0 {
		return nil, nil
	}

	result := make(map[string][]byte, len(c.sparse))
	for name, idx := range c.sparse {
		data, err := idx.Export()
		if err != nil {
			return nil, fmt.Errorf("export sparse index %s: %w", name, err)
		}
		result[name] = data
	}
	return result, nil
}

// ImportSparseIndexes restores sparse index data from field -> serialized bytes.
func (c *Collection) ImportSparseIndexes(data map[string][]byte) error {
	if c.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	for name, raw := range data {
		idx, ok := c.sparse[name]
		if !ok {
			continue // Field no longer in schema, skip
		}
		if err := idx.Import(raw); err != nil {
			return fmt.Errorf("import sparse index %s: %w", name, err)
		}
	}
	return nil
}

// ImportIndexes restores dense index data from field -> serialized bytes.
func (c *Collection) ImportIndexes(data map[string][]byte) error {
	if c.isDurableReadOnly() {
		return ErrUnsupportedDurableMutation
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	for name, raw := range data {
		idx, ok := c.indexes[name]
		if !ok {
			return fmt.Errorf("index %s not found in schema", name)
		}
		if err := idx.Import(raw); err != nil {
			return fmt.Errorf("import index %s: %w", name, err)
		}
	}
	return nil
}

// ExportMetadata returns a copy of all document metadata.
func (c *Collection) ExportMetadata() map[uint64]map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make(map[uint64]map[string]interface{}, len(c.documents))
	for id, doc := range c.documents {
		if doc.Metadata != nil {
			meta := make(map[string]interface{}, len(doc.Metadata))
			for k, v := range doc.Metadata {
				meta[k] = cloneDocumentValue(v)
			}
			result[id] = meta
		}
	}
	return result
}

// ImportMetadata restores document metadata and creates minimal document records.
func (c *Collection) ImportMetadata(data map[uint64]map[string]interface{}) {
	if c.isDurableReadOnly() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	for id, meta := range data {
		if _, exists := c.documents[id]; !exists {
			c.documents[id] = &Document{ID: id}
		}
		c.documents[id].Metadata = meta
	}
}

// SetNextID sets the next document ID counter.
func (c *Collection) SetNextID(id uint64) {
	if c.isDurableReadOnly() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nextID = id
}

// GetNextID returns the current next document ID counter.
func (c *Collection) GetNextID() uint64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.nextID
}

// ExportDocuments returns a shallow copy of all document records for persistence.
func (c *Collection) ExportDocuments() map[uint64]*Document {
	c.mu.RLock()
	defer c.mu.RUnlock()
	docs := make(map[uint64]*Document, len(c.documents))
	for id, doc := range c.documents {
		if doc != nil {
			clone := cloneDocumentPreservingTypes(*doc)
			docs[id] = &clone
		}
	}
	return docs
}

// ImportDocuments restores document records.
func (c *Collection) ImportDocuments(docs map[uint64]*Document) {
	if c.isDurableReadOnly() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for id, doc := range docs {
		c.documents[id] = doc
	}
}

// getDocumentVector retrieves a dense vector for a document from the stored documents.
func (c *Collection) getDocumentVector(docID uint64, fieldName string) ([]float32, error) {
	doc, ok := c.documents[docID]
	if !ok {
		return nil, fmt.Errorf("document %d not found", docID)
	}
	vec := doc.Vectors[fieldName]
	if vec.Dense == nil {
		return nil, fmt.Errorf("expected dense vector, got %+v", vec)
	}
	return vec.Dense, nil
}

func l2NormalizeInPlace(vec []float32) {
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(norm))
	for i := range vec {
		vec[i] *= inv
	}
}

func cosineSimilarity(a, b []float32) float32 {
	// Use SIMD distance (returns 1 - cosine_similarity for normalized vectors)
	dist := simd.CosineDistanceF32(a, b)
	return 1.0 - dist
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Close releases all resources held by the collection.
// This should be called when deleting a collection.
func (c *Collection) Close() {
	if c.isDurableReadOnly() {
		return
	}
	c.closeDirect()
}

func (c *Collection) closeDirect() {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Clear indexes (they will be garbage collected)
	for name := range c.indexes {
		delete(c.indexes, name)
	}
	for name := range c.sparse {
		delete(c.sparse, name)
	}

	// Clear documents
	for id := range c.documents {
		delete(c.documents, id)
	}
}
