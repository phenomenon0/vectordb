package collection

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/phenomenon0/vectordb/internal/filter"
	"github.com/phenomenon0/vectordb/internal/hybrid"
	"github.com/phenomenon0/vectordb/internal/index"
	"github.com/phenomenon0/vectordb/internal/sparse"
)

// Collection manages multiple vector indexes for a single collection.
//
// A collection can have multiple vector fields, each with its own index:
//   - Dense fields use HNSW/IVF/FLAT indexes
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

	// Default ef_search for HNSW (from env HNSW_EFSEARCH or 64)
	defaultEfSearch int
}

// NewCollection creates a new multi-vector collection.
func NewCollection(schema CollectionSchema) (*Collection, error) {
	if err := schema.Validate(); err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}

	// Read default ef_search from environment (fallback: 200)
	defaultEf := 200
	if envEf := os.Getenv("HNSW_EFSEARCH"); envEf != "" {
		if parsed, err := strconv.Atoi(envEf); err == nil && parsed > 0 {
			defaultEf = parsed
		}
	}

	c := &Collection{
		schema:          schema,
		indexes:         make(map[string]index.Index),
		sparse:          make(map[string]*sparse.InvertedIndex),
		documents:       make(map[uint64]*Document),
		nextID:          1,
		defaultEfSearch: defaultEf,
	}

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

// createDenseIndex creates a dense vector index (HNSW/IVF/FLAT).
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

		idx, err = index.NewHNSWIndex(field.Dim, config)
		if err != nil {
			return fmt.Errorf("failed to create HNSW index: %w", err)
		}

	case IndexTypeIVF:
		// Set defaults if not provided
		if _, ok := config["nlist"]; !ok {
			config["nlist"] = 100
		}

		idx, err = index.NewIVFIndex(field.Dim, config)
		if err != nil {
			return fmt.Errorf("failed to create IVF index: %w", err)
		}

	case IndexTypeFLAT:
		idx, err = index.NewFLATIndex(field.Dim, config)
		if err != nil {
			return fmt.Errorf("failed to create FLAT index: %w", err)
		}

	default:
		return fmt.Errorf("index type %s not supported for dense vectors", field.Index.Type)
	}

	c.indexes[field.Name] = idx
	return nil
}

// createSparseIndex creates a sparse vector index (Inverted).
func (c *Collection) createSparseIndex(field VectorField) error {
	if field.Index.Type != IndexTypeInverted {
		return fmt.Errorf("sparse vectors require inverted index, got %s", field.Index.Type)
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
	c.mu.Lock()
	defer c.mu.Unlock()

	// Assign ID if not set
	if doc.ID == 0 {
		doc.ID = c.nextID
		c.nextID++
	}

	// Validate document against schema
	if err := doc.Validate(&c.schema); err != nil {
		return fmt.Errorf("document validation failed: %w", err)
	}

	// Add to each index
	for _, field := range c.schema.Fields {
		vector, ok := doc.Vectors[field.Name]
		if !ok {
			return fmt.Errorf("missing vector for field: %s", field.Name)
		}

		if err := c.addToIndex(ctx, field, doc.ID, vector); err != nil {
			return fmt.Errorf("failed to add to index %s: %w", field.Name, err)
		}

		// Store metadata in the index for filtered search
		if doc.Metadata != nil && len(doc.Metadata) > 0 {
			if err := c.setIndexMetadata(field, doc.ID, doc.Metadata); err != nil {
				return fmt.Errorf("failed to set metadata for index %s: %w", field.Name, err)
			}
		}
	}

	// Store document
	c.documents[doc.ID] = doc

	return nil
}

func coerceDenseVector(vector interface{}) ([]float32, error) {
	switch v := vector.(type) {
	case []float32:
		// Zero-alloc fast path: returns the input slice directly.
		// Safe because: (1) HTTP handlers create a fresh []float32 per request,
		// (2) HNSW Search does not mutate the query vector, and
		// (3) HNSW Add makes a defensive copy in idx.Add().
		return v, nil
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
	default:
		return nil, fmt.Errorf("expected dense vector, got %T", vector)
	}
}

func coerceUint32Slice(value interface{}) ([]uint32, error) {
	switch v := value.(type) {
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

func coerceSparseVector(vector interface{}) (*sparse.SparseVector, error) {
	switch v := vector.(type) {
	case *sparse.SparseVector:
		return v, nil
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
		sparseVec, err := sparse.NewSparseVector(indices, values, dim)
		if err != nil {
			return nil, fmt.Errorf("failed to create sparse vector: %w", err)
		}
		return sparseVec, nil
	default:
		return nil, fmt.Errorf("expected *SparseVector or map, got %T", vector)
	}
}

// addToIndex adds a vector to the appropriate index.
func (c *Collection) addToIndex(ctx context.Context, field VectorField, docID uint64, vector interface{}) error {
	switch field.Type {
	case VectorTypeDense:
		denseVec, err := coerceDenseVector(vector)
		if err != nil {
			return fmt.Errorf("invalid dense field %s: %w", field.Name, err)
		}

		idx, ok := c.indexes[field.Name]
		if !ok {
			return fmt.Errorf("index not found for field: %s", field.Name)
		}

		return idx.Add(ctx, docID, denseVec)

	case VectorTypeSparse:
		sparseVec, err := coerceSparseVector(vector)
		if err != nil {
			return fmt.Errorf("invalid sparse field %s: %w", field.Name, err)
		}

		idx, ok := c.sparse[field.Name]
		if !ok {
			return fmt.Errorf("sparse index not found for field: %s", field.Name)
		}

		return idx.Add(ctx, docID, sparseVec)

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

		// Index doesn't support metadata (e.g., FLAT index), silently skip
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
func (c *Collection) Search(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if req.CollectionName != c.schema.Name {
		return nil, fmt.Errorf("collection mismatch: expected %s, got %s", c.schema.Name, req.CollectionName)
	}

	// Parse metadata filters if provided
	var metadataFilter filter.Filter
	if req.Filters != nil && len(req.Filters) > 0 {
		var err error
		metadataFilter, err = filter.FromMap(req.Filters)
		if err != nil {
			return nil, fmt.Errorf("invalid filter: %w", err)
		}
	}

	includeVectors := true
	if req.IncludeVectors != nil {
		includeVectors = *req.IncludeVectors
	}

	// Resolve ef_search: request override > server default
	efSearch := c.defaultEfSearch
	if req.EfSearch > 0 {
		efSearch = req.EfSearch
	}

	// Single-field search
	if len(req.Queries) == 1 {
		for fieldName, queryVec := range req.Queries {
			return c.searchSingleField(ctx, fieldName, queryVec, req.TopK, efSearch, includeVectors, metadataFilter)
		}
	}

	// Multi-field hybrid search
	if req.HybridParams != nil {
		return c.searchHybrid(ctx, req, efSearch, includeVectors, metadataFilter)
	}

	// Multiple fields without fusion (return error)
	return nil, fmt.Errorf("multiple query fields require HybridParams")
}

// searchSingleField performs a search on a single vector field.
func (c *Collection) searchSingleField(ctx context.Context, fieldName string, queryVec interface{}, k int, efSearch int, includeVectors bool, metadataFilter filter.Filter) (*SearchResponse, error) {
	field := c.schema.GetField(fieldName)
	if field == nil {
		return nil, fmt.Errorf("field not found: %s", fieldName)
	}

	var results []hybrid.SearchResult

	switch field.Type {
	case VectorTypeDense:
		denseQuery, err := coerceDenseVector(queryVec)
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
		case IndexTypeIVF:
			params = index.IVFSearchParams{
				NProbe: 10, // Default value
				Filter: metadataFilter,
			}
		default:
			// For other index types (FLAT, DiskANN), use HNSW params as fallback
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
		sparseQuery, err := coerceSparseVector(queryVec)
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

	// Retrieve documents in a single tight loop for cache-friendly access.
	// Pre-allocate both slices at once to reduce allocator pressure.
	n := len(results)
	docs := make([]Document, n)
	scores := make([]float32, n)
	for i := 0; i < n; i++ {
		scores[i] = results[i].Score
		if doc, ok := c.documents[results[i].DocID]; ok {
			docs[i].ID = doc.ID
			docs[i].Metadata = doc.Metadata
			if includeVectors {
				docs[i].Vectors = doc.Vectors
			}
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
			denseQuery, err = coerceDenseVector(queryVec)
			if err != nil {
				return nil, fmt.Errorf("invalid dense query for %s: %w", fieldName, err)
			}
		case VectorTypeSparse:
			sparseField = fieldName
			var err error
			sparseQuery, err = coerceSparseVector(queryVec)
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
		case IndexTypeIVF:
			params = index.IVFSearchParams{
				NProbe: 10, // Default value
				Filter: metadataFilter,
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
			denseResults[i] = hybrid.SearchResult{
				DocID: r.ID,
				Score: r.Distance,
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
			if dw, ok := weights["dense"]; ok {
				fusionParams.DenseWeight = dw
			}
			if sw, ok := weights["sparse"]; ok {
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

	// Retrieve documents (lightweight copy: skip vectors unless requested)
	docs := make([]Document, len(fusedResults))
	scores := make([]float32, len(fusedResults))
	for i, r := range fusedResults {
		if doc, ok := c.documents[r.DocID]; ok {
			docs[i] = Document{
				ID:       doc.ID,
				Metadata: doc.Metadata,
			}
			if includeVectors {
				docs[i].Vectors = doc.Vectors
			}
		}
		scores[i] = r.Score
	}

	return &SearchResponse{
		Documents:          docs,
		Scores:             scores,
		CandidatesExamined: len(denseResults) + len(sparseResults),
	}, nil
}

// BatchAdd adds multiple documents to the collection.
// When the underlying index implements index.BatchAdder, vectors are inserted
// in a single batch call (one lock cycle) instead of per-document.
func (c *Collection) BatchAdd(ctx context.Context, docs []Document) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Phase 1: Assign IDs and validate all documents up front.
	for i := range docs {
		if docs[i].ID == 0 {
			docs[i].ID = c.nextID
			c.nextID++
		}
		if err := docs[i].Validate(&c.schema); err != nil {
			return fmt.Errorf("document %d validation failed: %w", i, err)
		}
	}

	// Phase 2: For each field, collect vectors and batch-insert if possible.
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
					vec, err := coerceDenseVector(docs[i].Vectors[field.Name])
					if err != nil {
						return fmt.Errorf("doc %d field %s: %w", i, field.Name, err)
					}
					batch[docs[i].ID] = vec
				}
				if err := batcher.BatchAdd(ctx, batch); err != nil {
					return fmt.Errorf("batch add to index %s: %w", field.Name, err)
				}
			} else {
				// Fallback: per-vector add (still under the single collection lock)
				for i := range docs {
					vec, err := coerceDenseVector(docs[i].Vectors[field.Name])
					if err != nil {
						return fmt.Errorf("doc %d field %s: %w", i, field.Name, err)
					}
					if err := idx.Add(ctx, docs[i].ID, vec); err != nil {
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
				sv, err := coerceSparseVector(docs[i].Vectors[field.Name])
				if err != nil {
					return fmt.Errorf("doc %d field %s: %w", i, field.Name, err)
				}
				if err := sparseIdx.Add(ctx, docs[i].ID, sv); err != nil {
					return fmt.Errorf("doc %d add to sparse index %s: %w", i, field.Name, err)
				}
			}
		}
	}

	// Phase 3: Set metadata and store documents.
	for i := range docs {
		if docs[i].Metadata != nil && len(docs[i].Metadata) > 0 {
			for _, field := range c.schema.Fields {
				if err := c.setIndexMetadata(field, docs[i].ID, docs[i].Metadata); err != nil {
					return fmt.Errorf("doc %d metadata for %s: %w", i, field.Name, err)
				}
			}
		}
		c.documents[docs[i].ID] = &docs[i]
	}

	return nil
}

// BulkAddDense inserts raw dense vectors into a single field without full Document overhead.
// IDs and vectors must be the same length. Minimal Document records are created (ID only).
func (c *Collection) BulkAddDense(ctx context.Context, fieldName string, ids []uint64, vectors [][]float32) error {
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

	// Create minimal document records
	for i, id := range ids {
		c.documents[id] = &Document{
			ID:      id,
			Vectors: map[string]interface{}{fieldName: vectors[i]},
		}
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
	return doc, ok
}

// Delete removes a document from the collection.
func (c *Collection) Delete(ctx context.Context, docID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

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

	return nil
}

// Count returns the number of documents in the collection.
func (c *Collection) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.documents)
}

// Schema returns the collection schema.
func (c *Collection) Schema() CollectionSchema {
	return c.schema
}

// Name returns the collection name.
func (c *Collection) Name() string {
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

// ImportIndexes restores dense index data from field -> serialized bytes.
func (c *Collection) ImportIndexes(data map[string][]byte) error {
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
				meta[k] = v
			}
			result[id] = meta
		}
	}
	return result
}

// ImportMetadata restores document metadata and creates minimal document records.
func (c *Collection) ImportMetadata(data map[uint64]map[string]interface{}) {
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

// ExportDocuments returns all document records for persistence.
func (c *Collection) ExportDocuments() map[uint64]*Document {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.documents
}

// ImportDocuments restores document records.
func (c *Collection) ImportDocuments(docs map[uint64]*Document) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for id, doc := range docs {
		c.documents[id] = doc
	}
}

// Close releases all resources held by the collection.
// This should be called when deleting a collection.
func (c *Collection) Close() {
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
