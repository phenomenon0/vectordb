package collection

import (
	"context"
	"fmt"
	"sync"

	"github.com/phenomenon0/vectordb/filter"
	"github.com/phenomenon0/vectordb/hybrid"
	"github.com/phenomenon0/vectordb/index"
	"github.com/phenomenon0/vectordb/sparse"
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
}

// NewCollection creates a new multi-vector collection.
func NewCollection(schema CollectionSchema) (*Collection, error) {
	if err := schema.Validate(); err != nil {
		return nil, fmt.Errorf("invalid schema: %w", err)
	}

	c := &Collection{
		schema:    schema,
		indexes:   make(map[string]index.Index),
		sparse:    make(map[string]*sparse.InvertedIndex),
		documents: make(map[uint64]*Document),
		nextID:    1,
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
func (c *Collection) Add(ctx context.Context, doc Document) error {
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
	c.documents[doc.ID] = &doc

	return nil
}

// addToIndex adds a vector to the appropriate index.
func (c *Collection) addToIndex(ctx context.Context, field VectorField, docID uint64, vector interface{}) error {
	switch field.Type {
	case VectorTypeDense:
		denseVec, ok := vector.([]float32)
		if !ok {
			return fmt.Errorf("expected []float32 for dense field %s, got %T", field.Name, vector)
		}

		idx, ok := c.indexes[field.Name]
		if !ok {
			return fmt.Errorf("index not found for field: %s", field.Name)
		}

		return idx.Add(ctx, docID, denseVec)

	case VectorTypeSparse:
		// Accept either *SparseVector or map format
		var sparseVec *sparse.SparseVector
		switch v := vector.(type) {
		case *sparse.SparseVector:
			sparseVec = v
		case map[string]interface{}:
			// Parse from map format: {indices: [...], values: [...], dim: X}
			indices, ok1 := v["indices"].([]uint32)
			values, ok2 := v["values"].([]float32)
			dim, ok3 := v["dim"].(int)
			if !ok1 || !ok2 || !ok3 {
				return fmt.Errorf("invalid sparse vector format for field %s", field.Name)
			}
			var err error
			sparseVec, err = sparse.NewSparseVector(indices, values, dim)
			if err != nil {
				return fmt.Errorf("failed to create sparse vector: %w", err)
			}
		default:
			return fmt.Errorf("expected *SparseVector or map for sparse field %s, got %T", field.Name, vector)
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

	// Single-field search
	if len(req.Queries) == 1 {
		for fieldName, queryVec := range req.Queries {
			return c.searchSingleField(ctx, fieldName, queryVec, req.TopK, metadataFilter)
		}
	}

	// Multi-field hybrid search
	if req.HybridParams != nil {
		return c.searchHybrid(ctx, req, metadataFilter)
	}

	// Multiple fields without fusion (return error)
	return nil, fmt.Errorf("multiple query fields require HybridParams")
}

// searchSingleField performs a search on a single vector field.
func (c *Collection) searchSingleField(ctx context.Context, fieldName string, queryVec interface{}, k int, metadataFilter filter.Filter) (*SearchResponse, error) {
	field := c.schema.GetField(fieldName)
	if field == nil {
		return nil, fmt.Errorf("field not found: %s", fieldName)
	}

	var results []hybrid.SearchResult

	switch field.Type {
	case VectorTypeDense:
		denseQuery, ok := queryVec.([]float32)
		if !ok {
			return nil, fmt.Errorf("expected []float32 for dense query, got %T", queryVec)
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
				EfSearch: 64, // Default value
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
		sparseQuery, ok := queryVec.(*sparse.SparseVector)
		if !ok {
			return nil, fmt.Errorf("expected *SparseVector for sparse query, got %T", queryVec)
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

	// Retrieve documents
	docs := make([]Document, len(results))
	scores := make([]float32, len(results))
	for i, r := range results {
		if doc, ok := c.documents[r.DocID]; ok {
			docs[i] = *doc
		}
		scores[i] = r.Score
	}

	return &SearchResponse{
		Documents:          docs,
		Scores:             scores,
		CandidatesExamined: len(results),
	}, nil
}

// searchHybrid performs hybrid search across multiple vector fields.
func (c *Collection) searchHybrid(ctx context.Context, req SearchRequest, metadataFilter filter.Filter) (*SearchResponse, error) {
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
			var ok bool
			denseQuery, ok = queryVec.([]float32)
			if !ok {
				return nil, fmt.Errorf("expected []float32 for dense query, got %T", queryVec)
			}
		case VectorTypeSparse:
			sparseField = fieldName
			var ok bool
			sparseQuery, ok = queryVec.(*sparse.SparseVector)
			if !ok {
				return nil, fmt.Errorf("expected *SparseVector for sparse query, got %T", queryVec)
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
				EfSearch: 64, // Default value
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

	// Retrieve documents
	docs := make([]Document, len(fusedResults))
	scores := make([]float32, len(fusedResults))
	for i, r := range fusedResults {
		if doc, ok := c.documents[r.DocID]; ok {
			docs[i] = *doc
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
func (c *Collection) BatchAdd(ctx context.Context, docs []Document) error {
	for i := range docs {
		if err := c.Add(ctx, docs[i]); err != nil {
			return fmt.Errorf("failed to add document %d: %w", i, err)
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
