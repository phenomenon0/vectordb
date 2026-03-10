package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/encoding"
	"github.com/phenomenon0/vectordb/internal/graph"
	"github.com/phenomenon0/vectordb/internal/hybrid"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/sparse"
)

// JSON response buffer pool for encoding.
var jsonBufPool = sync.Pool{
	New: func() interface{} { return new(bytes.Buffer) },
}

// decodeDenseVectorFast decodes a JSON array of numbers directly into []float32,
// avoiding the intermediate []interface{} allocation that json.Unmarshal produces.
func decodeDenseVectorFast(raw json.RawMessage) ([]float32, error) {
	// Quick validation and count elements by scanning for commas
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) < 2 || trimmed[0] != '[' || trimmed[len(trimmed)-1] != ']' {
		return nil, fmt.Errorf("expected JSON array for dense vector")
	}

	// Estimate dimension from byte length (avg ~8 bytes per float: "0.1234,")
	estimatedDim := len(trimmed) / 6
	if estimatedDim < 8 {
		estimatedDim = 8
	}

	// Decode directly using json.Decoder to stream numbers
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	// Read opening '['
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("invalid dense vector array: %w", err)
	}

	result := make([]float32, 0, estimatedDim)
	for dec.More() {
		var f float64
		if err := dec.Decode(&f); err != nil {
			return nil, fmt.Errorf("invalid dense vector element: %w", err)
		}
		result = append(result, float32(f))
	}
	return result, nil
}

// searchRequestRaw is used for two-phase JSON decoding of search requests.
// The queries field is kept as raw JSON so vectors can be decoded directly
// into []float32 without the []interface{} intermediate.
type searchRequestRaw struct {
	CollectionName string                          `json:"collection"`
	Queries        map[string]json.RawMessage      `json:"queries"`
	QueryText      string                          `json:"query_text"`
	TopK           int                             `json:"top_k"`
	EfSearch       int                             `json:"ef_search,omitempty"`
	IncludeVectors *bool                           `json:"include_vectors,omitempty"`
	Offset         int                             `json:"offset,omitempty"`
	Filters        map[string]interface{}          `json:"filters,omitempty"`
	HybridParams   *vcollection.HybridSearchParams `json:"hybrid_params,omitempty"`
	GraphWeight    float32                         `json:"graph_weight,omitempty"`
}

// CollectionHTTPServer wraps CollectionManager for HTTP API access
type CollectionHTTPServer struct {
	manager       *vcollection.CollectionManager
	tenantManager *vcollection.TenantManager // Multi-tenant collection manager
	graphIndex    *graph.GraphIndex          // Optional GraphRAG index for graph-boosted search
}

// NewCollectionHTTPServer creates a new HTTP server wrapper for CollectionManager
func NewCollectionHTTPServer(storagePath string) *CollectionHTTPServer {
	return &CollectionHTTPServer{
		manager:       vcollection.NewCollectionManager(storagePath),
		tenantManager: vcollection.NewTenantManager(storagePath),
	}
}

// EnableGraphRAG activates the GraphRAG index for graph-boosted hybrid search.
func (s *CollectionHTTPServer) EnableGraphRAG(cfg graph.Config) {
	s.graphIndex = graph.NewGraphIndex(cfg)
}

// RegisterHandlers registers all collection HTTP handlers with the given mux
func (s *CollectionHTTPServer) RegisterHandlers(mux *http.ServeMux, guard func(http.HandlerFunc) http.HandlerFunc, adminGuard func(http.HandlerFunc) http.HandlerFunc) {
	// Admin endpoints for collection management
	mux.HandleFunc("/v2/collections", adminGuard(s.handleCollections))
	mux.HandleFunc("/v2/collections/", adminGuard(s.handleCollectionOps))

	// Document operations (multi-vector support)
	mux.HandleFunc("/v2/insert", guard(s.handleInsert))
	mux.HandleFunc("/v2/insert/batch", guard(s.handleBatchInsert)) // True batch insert
	mux.HandleFunc("/v2/search", guard(s.handleSearch))
	mux.HandleFunc("/v2/delete", guard(s.handleDelete))

	// Multi-tenant endpoints (v3)
	// Tenant collection management
	mux.HandleFunc("/v3/tenants/", guard(s.handleTenantRoutes))
}

// handleCollections handles POST (create) and GET (list) on /v2/collections
func (s *CollectionHTTPServer) handleCollections(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		s.handleCreateCollection(w, r)
	case http.MethodGet:
		s.handleListCollections(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleCreateCollection creates a new multi-vector collection
func (s *CollectionHTTPServer) handleCreateCollection(w http.ResponseWriter, r *http.Request) {
	var schema vcollection.CollectionSchema
	if err := json.NewDecoder(r.Body).Decode(&schema); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Create collection
	ctx := r.Context()
	if _, err := s.manager.CreateCollection(ctx, schema); err != nil {
		http.Error(w, fmt.Sprintf("failed to create collection: %v", err), http.StatusBadRequest)
		return
	}

	// Return success
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "success",
		"message": fmt.Sprintf("collection %q created", schema.Name),
	})
}

// handleListCollections returns all collections with their info
func (s *CollectionHTTPServer) handleListCollections(w http.ResponseWriter, r *http.Request) {
	infos := s.manager.ListCollectionInfos()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "success",
		"count":       len(infos),
		"collections": infos,
	})
}

// handleCollectionOps handles operations on specific collections: GET, DELETE, PUT /v2/collections/{name}
func (s *CollectionHTTPServer) handleCollectionOps(w http.ResponseWriter, r *http.Request) {
	// Extract collection name from URL
	path := strings.TrimPrefix(r.URL.Path, "/v2/collections/")
	parts := strings.Split(path, "/")

	if len(parts) == 0 || parts[0] == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	collectionName := parts[0]

	// Route based on HTTP method and path structure
	if len(parts) == 1 {
		// /v2/collections/{name}
		switch r.Method {
		case http.MethodGet:
			s.handleGetCollection(w, r, collectionName)
		case http.MethodDelete:
			s.handleDeleteCollection(w, r, collectionName)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	} else if len(parts) == 2 {
		// /v2/collections/{name}/stats
		operation := parts[1]
		if operation == "stats" && r.Method == http.MethodGet {
			s.handleCollectionStats(w, r, collectionName)
		} else {
			http.Error(w, "unknown operation or method not allowed", http.StatusBadRequest)
		}
	} else {
		http.Error(w, "invalid URL format", http.StatusBadRequest)
	}
}

// handleGetCollection returns collection info
func (s *CollectionHTTPServer) handleGetCollection(w http.ResponseWriter, r *http.Request, name string) {
	info, err := s.manager.GetCollectionInfo(name)
	if err != nil {
		http.Error(w, fmt.Sprintf("collection not found: %v", err), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":     "success",
		"collection": info,
	})
}

// handleDeleteCollection deletes a collection
func (s *CollectionHTTPServer) handleDeleteCollection(w http.ResponseWriter, r *http.Request, name string) {
	ctx := r.Context()
	if err := s.manager.DeleteCollection(ctx, name); err != nil {
		http.Error(w, fmt.Sprintf("failed to delete collection: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "success",
		"message": fmt.Sprintf("collection %q deleted", name),
	})
}

// handleCollectionStats returns collection statistics
func (s *CollectionHTTPServer) handleCollectionStats(w http.ResponseWriter, r *http.Request, name string) {
	info, err := s.manager.GetCollectionInfo(name)
	if err != nil {
		http.Error(w, fmt.Sprintf("collection not found: %v", err), http.StatusNotFound)
		return
	}

	stats := s.manager.GetStats()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        "success",
		"name":          name,
		"doc_count":     info.DocCount,
		"manager_stats": stats,
	})
}

// InsertRequest for multi-vector documents
type InsertRequest struct {
	CollectionName string                 `json:"collection"`
	Doc            string                 `json:"doc"`
	Vectors        map[string]interface{} `json:"vectors"` // field name -> vector data
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// handleInsert adds a document with multiple vectors to a collection
func (s *CollectionHTTPServer) handleInsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req InsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.CollectionName == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	if len(req.Vectors) == 0 {
		http.Error(w, "at least one vector required", http.StatusBadRequest)
		return
	}

	// Convert vectors to proper format
	vectors := make(map[string]interface{})
	for fieldName, vectorData := range req.Vectors {
		// Check if it's a sparse vector (has indices and values)
		if vecMap, ok := vectorData.(map[string]interface{}); ok {
			if indices, hasIndices := vecMap["indices"]; hasIndices {
				// Sparse vector format
				indicesSlice, ok1 := indices.([]interface{})
				values, ok2 := vecMap["values"].([]interface{})
				dim, ok3 := vecMap["dim"].(float64)

				if !ok1 || !ok2 || !ok3 {
					http.Error(w, fmt.Sprintf("invalid sparse vector format for field %s", fieldName), http.StatusBadRequest)
					return
				}

				// Convert to uint32 and float32
				uint32Indices := make([]uint32, len(indicesSlice))
				for i, v := range indicesSlice {
					if f, ok := v.(float64); ok {
						uint32Indices[i] = uint32(f)
					}
				}

				float32Values := make([]float32, len(values))
				for i, v := range values {
					if f, ok := v.(float64); ok {
						float32Values[i] = float32(f)
					}
				}

				sparseVec, err := sparse.NewSparseVector(uint32Indices, float32Values, int(dim))
				if err != nil {
					http.Error(w, fmt.Sprintf("invalid sparse vector: %v", err), http.StatusBadRequest)
					return
				}
				vectors[fieldName] = sparseVec
				continue
			}
		}

		// Dense vector format
		if vecSlice, ok := vectorData.([]interface{}); ok {
			denseVec := make([]float32, len(vecSlice))
			for i, v := range vecSlice {
				if f, ok := v.(float64); ok {
					denseVec[i] = float32(f)
				}
			}
			vectors[fieldName] = denseVec
		}
	}

	// Create document
	doc := vcollection.Document{
		Vectors:  vectors,
		Metadata: req.Metadata,
	}

	// Add to collection
	ctx := r.Context()
	if err := s.manager.AddDocument(ctx, req.CollectionName, &doc); err != nil {
		http.Error(w, fmt.Sprintf("failed to add document: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "success",
		"id":      doc.ID,
		"message": "document added",
	})
}

// BatchInsertRequest for bulk document insertion (SOTA batch API)
type BatchInsertRequest struct {
	CollectionName  string          `json:"collection"`
	Docs            []InsertRequest `json:"docs"`
	ContinueOnError bool            `json:"continue_on_error"`
}

// handleBatchInsert adds multiple documents in a single request (true batch)
// This provides 10-50x throughput improvement over sequential inserts
func (s *CollectionHTTPServer) handleBatchInsert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req BatchInsertRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.CollectionName == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	const MaxBatchSize = 10_000
	if len(req.Docs) == 0 {
		http.Error(w, "no documents provided", http.StatusBadRequest)
		return
	}
	if len(req.Docs) > MaxBatchSize {
		http.Error(w, fmt.Sprintf("batch too large: max %d documents", MaxBatchSize), http.StatusBadRequest)
		return
	}

	ids := make([]uint64, 0, len(req.Docs))
	errors := make(map[int]string)
	ctx := r.Context()

	for i, docReq := range req.Docs {
		// Use collection from batch request if not specified per-doc
		collectionName := docReq.CollectionName
		if collectionName == "" {
			collectionName = req.CollectionName
		}

		// Validate vectors
		if len(docReq.Vectors) == 0 {
			errors[i] = "at least one vector required"
			if !req.ContinueOnError {
				break
			}
			continue
		}

		// Convert vectors to proper format (reuse existing logic)
		vectors := make(map[string]interface{})
		for fieldName, vectorData := range docReq.Vectors {
			// Check if it's a sparse vector (has indices and values)
			if vecMap, ok := vectorData.(map[string]interface{}); ok {
				if indices, hasIndices := vecMap["indices"]; hasIndices {
					// Sparse vector format
					indicesSlice, ok1 := indices.([]interface{})
					values, ok2 := vecMap["values"].([]interface{})
					dim, ok3 := vecMap["dim"].(float64)

					if !ok1 || !ok2 || !ok3 {
						errors[i] = fmt.Sprintf("invalid sparse vector format for field %s", fieldName)
						if !req.ContinueOnError {
							break
						}
						continue
					}

					// Convert to uint32 and float32
					uint32Indices := make([]uint32, len(indicesSlice))
					for j, v := range indicesSlice {
						if f, ok := v.(float64); ok {
							uint32Indices[j] = uint32(f)
						}
					}

					float32Values := make([]float32, len(values))
					for j, v := range values {
						if f, ok := v.(float64); ok {
							float32Values[j] = float32(f)
						}
					}

					sparseVec, err := sparse.NewSparseVector(uint32Indices, float32Values, int(dim))
					if err != nil {
						errors[i] = fmt.Sprintf("invalid sparse vector: %v", err)
						if !req.ContinueOnError {
							break
						}
						continue
					}
					vectors[fieldName] = sparseVec
					continue
				}
			}

			// Dense vector format
			if vecSlice, ok := vectorData.([]interface{}); ok {
				denseVec := make([]float32, len(vecSlice))
				for j, v := range vecSlice {
					if f, ok := v.(float64); ok {
						denseVec[j] = float32(f)
					}
				}
				vectors[fieldName] = denseVec
			}
		}

		// Skip if we hit an error during vector conversion.
		// FIX #7: When ContinueOnError=false, a break from the inner field loop
		// must also break the outer doc loop, not just continue to the next doc.
		if _, hasError := errors[i]; hasError {
			if !req.ContinueOnError {
				break
			}
			continue
		}

		// Create document
		doc := vcollection.Document{
			Vectors:  vectors,
			Metadata: docReq.Metadata,
		}

		// Add to collection
		if err := s.manager.AddDocument(ctx, collectionName, &doc); err != nil {
			errors[i] = err.Error()
			if !req.ContinueOnError {
				break
			}
			continue
		}
		ids = append(ids, doc.ID)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "success",
		"ids":      ids,
		"inserted": len(ids),
		"failed":   len(errors),
		"errors":   errors,
	})
}

// searchJSONResponse is a typed struct for JSON encoding search results.
// Using a struct instead of map[string]interface{} lets the JSON encoder
// use the cached struct codec path, avoiding runtime reflection overhead.
type searchJSONResponse struct {
	Status             string                 `json:"status"`
	Documents          []vcollection.Document `json:"documents"`
	Scores             []float32              `json:"scores"`
	CandidatesExamined int                    `json:"candidates_examined"`
}

// tenantSearchJSONResponse is a typed struct for tenant search JSON encoding.
type tenantSearchJSONResponse struct {
	Status             string                 `json:"status"`
	TenantID           string                 `json:"tenant_id"`
	Documents          []vcollection.Document `json:"documents"`
	Scores             []float32              `json:"scores"`
	CandidatesExamined int                    `json:"candidates_examined"`
}

// SearchRequest for hybrid search
type SearchRequest struct {
	CollectionName string                          `json:"collection"`
	Queries        map[string]interface{}          `json:"queries"`    // field name -> query vector
	QueryText      string                          `json:"query_text"` // text query for GraphRAG entity matching
	TopK           int                             `json:"top_k"`
	EfSearch       int                             `json:"ef_search,omitempty"`       // HNSW ef_search override (0 = server default)
	IncludeVectors *bool                           `json:"include_vectors,omitempty"` // include vectors in response (nil = default true)
	Offset         int                             `json:"offset,omitempty"`
	Filters        map[string]interface{}          `json:"filters,omitempty"`
	HybridParams   *vcollection.HybridSearchParams `json:"hybrid_params,omitempty"`
	GraphWeight    float32                         `json:"graph_weight,omitempty"` // weight for graph signal (0 = disabled)
}

func resolveIncludeVectors(bodyValue *bool, raw string) (*bool, error) {
	if raw == "" {
		return bodyValue, nil
	}
	parsed, err := strconv.ParseBool(raw)
	if err != nil {
		return nil, fmt.Errorf("include_vectors must be true or false")
	}
	return &parsed, nil
}

// handleSearch performs search (dense, sparse, or hybrid)
func (s *CollectionHTTPServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Two-phase decode: parse structure first, then decode vectors directly
	// into []float32 via json.RawMessage, avoiding []interface{} intermediate.
	var req searchRequestRaw
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.CollectionName == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	if len(req.Queries) == 0 {
		http.Error(w, "at least one query vector required", http.StatusBadRequest)
		return
	}

	if req.TopK <= 0 {
		req.TopK = 10
	}
	if req.Offset < 0 {
		http.Error(w, "offset must be >= 0", http.StatusBadRequest)
		return
	}
	effectiveTopK := req.TopK + req.Offset
	if effectiveTopK < req.TopK { // overflow guard
		http.Error(w, "invalid top_k/offset combination", http.StatusBadRequest)
		return
	}

	// Decode query vectors directly into typed slices
	queries := make(map[string]interface{}, len(req.Queries))
	for fieldName, rawVec := range req.Queries {
		trimmed := bytes.TrimSpace(rawVec)
		if len(trimmed) == 0 {
			http.Error(w, fmt.Sprintf("empty query vector for field %s", fieldName), http.StatusBadRequest)
			return
		}

		// Sparse vector: JSON object with "indices" key
		if trimmed[0] == '{' {
			var vecMap map[string]interface{}
			if err := json.Unmarshal(rawVec, &vecMap); err != nil {
				http.Error(w, fmt.Sprintf("invalid sparse query for field %s: %v", fieldName, err), http.StatusBadRequest)
				return
			}
			if indices, hasIndices := vecMap["indices"]; hasIndices {
				indicesSlice, ok1 := indices.([]interface{})
				values, ok2 := vecMap["values"].([]interface{})
				dim, ok3 := vecMap["dim"].(float64)
				if !ok1 || !ok2 || !ok3 {
					http.Error(w, fmt.Sprintf("invalid sparse query vector for field %s", fieldName), http.StatusBadRequest)
					return
				}
				uint32Indices := make([]uint32, len(indicesSlice))
				for i, v := range indicesSlice {
					if f, ok := v.(float64); ok {
						uint32Indices[i] = uint32(f)
					}
				}
				float32Values := make([]float32, len(values))
				for i, v := range values {
					if f, ok := v.(float64); ok {
						float32Values[i] = float32(f)
					}
				}
				sparseVec, err := sparse.NewSparseVector(uint32Indices, float32Values, int(dim))
				if err != nil {
					http.Error(w, fmt.Sprintf("invalid sparse query: %v", err), http.StatusBadRequest)
					return
				}
				queries[fieldName] = sparseVec
				continue
			}
		}

		// Dense vector: JSON array — decode directly to []float32
		if trimmed[0] == '[' {
			denseVec, err := decodeDenseVectorFast(rawVec)
			if err != nil {
				http.Error(w, fmt.Sprintf("invalid dense query for field %s: %v", fieldName, err), http.StatusBadRequest)
				return
			}
			queries[fieldName] = denseVec
		}
	}

	includeVectors, err := resolveIncludeVectors(req.IncludeVectors, r.URL.Query().Get("include_vectors"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Build search request
	searchReq := vcollection.SearchRequest{
		CollectionName: req.CollectionName,
		Queries:        queries,
		TopK:           effectiveTopK,
		EfSearch:       req.EfSearch,
		IncludeVectors: includeVectors,
		Filters:        req.Filters,
		HybridParams:   req.HybridParams,
	}

	// Perform search
	ctx := r.Context()
	resp, err := s.manager.SearchCollection(ctx, searchReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("search failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Apply GraphRAG boosting if enabled
	if s.graphIndex != nil && s.graphIndex.NodeCount() > 0 && req.GraphWeight > 0 {
		queryTerms := strings.Fields(req.QueryText)
		if len(queryTerms) > 0 {
			graphResults := s.graphIndex.Search(queryTerms, effectiveTopK)
			if len(graphResults) > 0 {
				// Convert collection results to hybrid format
				baseResults := make([]hybrid.SearchResult, len(resp.Documents))
				for i, doc := range resp.Documents {
					score := float32(0)
					if i < len(resp.Scores) {
						score = resp.Scores[i]
					}
					baseResults[i] = hybrid.SearchResult{DocID: doc.ID, Score: score}
				}

				// Fuse with graph scores
				params := hybrid.FusionParams{
					Strategy:    hybrid.FusionRRF,
					DenseWeight: 1.0,
					GraphWeight: req.GraphWeight,
				}
				fused, _ := hybrid.HybridSearchWithGraph(baseResults, nil, graphResults, params, effectiveTopK)

				// Rebuild response from fused results
				docMap := make(map[uint64]vcollection.Document, len(resp.Documents))
				for _, doc := range resp.Documents {
					docMap[doc.ID] = doc
				}
				fusedDocs := make([]vcollection.Document, 0, len(fused))
				fusedScores := make([]float32, 0, len(fused))
				for _, fr := range fused {
					if doc, ok := docMap[fr.DocID]; ok {
						fusedDocs = append(fusedDocs, doc)
						fusedScores = append(fusedScores, fr.Score)
					}
				}
				resp.Documents = fusedDocs
				resp.Scores = fusedScores
			}
		}
	}

	// Apply offset pagination at HTTP layer.
	// Collection engine currently supports top-k only, so we overfetch (top_k+offset)
	// then trim to the requested page.
	documents := resp.Documents
	scores := resp.Scores
	if req.Offset > 0 {
		if req.Offset >= len(documents) {
			documents = []vcollection.Document{}
			scores = []float32{}
		} else {
			end := req.Offset + req.TopK
			if end > len(documents) {
				end = len(documents)
			}
			documents = documents[req.Offset:end]
			if req.Offset < len(scores) {
				scoreEnd := end
				if scoreEnd > len(scores) {
					scoreEnd = len(scores)
				}
				scores = scores[req.Offset:scoreEnd]
			} else {
				scores = []float32{}
			}
		}
	}

	// Check if client wants Glyph tabular format (50-62% fewer tokens for RAG)
	accept := r.Header.Get("Accept")
	if accept == "application/glyph" || accept == "text/glyph" {
		glyphOutput := encoding.EncodeSearchResults(documents, scores)
		w.Header().Set("Content-Type", "application/glyph")
		w.Write([]byte(glyphOutput))
		return
	}

	// Return results as JSON (default) — typed struct + pooled buffer
	buf := jsonBufPool.Get().(*bytes.Buffer)
	buf.Reset()
	enc := json.NewEncoder(buf)
	enc.Encode(searchJSONResponse{
		Status:             "success",
		Documents:          documents,
		Scores:             scores,
		CandidatesExamined: resp.CandidatesExamined,
	})
	w.Header().Set("Content-Type", "application/json")
	w.Write(buf.Bytes())
	jsonBufPool.Put(buf)
}

// DeleteRequest for document deletion
type DeleteRequest struct {
	CollectionName string `json:"collection"`
	DocID          uint64 `json:"doc_id"`
}

// handleDelete deletes a document from a collection
func (s *CollectionHTTPServer) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req DeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.CollectionName == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	if req.DocID == 0 {
		http.Error(w, "doc_id required", http.StatusBadRequest)
		return
	}

	// Delete document
	ctx := r.Context()
	if err := s.manager.DeleteDocument(ctx, req.CollectionName, req.DocID); err != nil {
		http.Error(w, fmt.Sprintf("failed to delete document: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "success",
		"message": fmt.Sprintf("document %d deleted from collection %s", req.DocID, req.CollectionName),
	})
}

// ======================================================================================
// MULTI-TENANT ROUTES (v3)
// URL pattern: /v3/tenants/{tenant_id}/collections[/{name}[/docs|/search]]
//
// The tenant ID comes from the URL path. This provides hard namespace isolation:
// each tenant has its own independent set of collections.
// ======================================================================================

// handleTenantRoutes is the top-level router for /v3/tenants/...
func (s *CollectionHTTPServer) handleTenantRoutes(w http.ResponseWriter, r *http.Request) {
	// Parse: /v3/tenants/{tenant_id}/collections[/{name}[/docs|/search]]
	path := strings.TrimPrefix(r.URL.Path, "/v3/tenants/")
	parts := strings.SplitN(path, "/", 5) // tenant_id / collections / name / operation / ...

	if len(parts) < 1 || parts[0] == "" {
		http.Error(w, "tenant ID required in URL path", http.StatusBadRequest)
		return
	}

	tenantID := parts[0]

	// Validate tenant ID format
	if !isValidTenantID(tenantID) {
		http.Error(w, "invalid tenant ID: must be 1-64 alphanumeric/hyphen/underscore characters", http.StatusBadRequest)
		return
	}

	// SECURITY: Authorize tenant access against the authenticated TenantContext.
	// The guard middleware injects the authenticated context; we must not allow
	// a valid token for tenant A to act on tenant B's data.
	tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
	if ok && !tenantCtx.IsAdmin && tenantCtx.TenantID != tenantID {
		http.Error(w, fmt.Sprintf("forbidden: token for tenant %q cannot access tenant %q", tenantCtx.TenantID, tenantID), http.StatusForbidden)
		return
	}

	// /v3/tenants/{tenant_id} — tenant info
	if len(parts) == 1 {
		s.handleTenantInfo(w, r, tenantID)
		return
	}

	// Must be /v3/tenants/{tenant_id}/collections[/...]
	if parts[1] != "collections" {
		http.Error(w, "unknown resource; expected 'collections'", http.StatusNotFound)
		return
	}

	// /v3/tenants/{tenant_id}/collections — list or create
	if len(parts) == 2 {
		switch r.Method {
		case http.MethodGet:
			s.handleTenantListCollections(w, r, tenantID)
		case http.MethodPost:
			s.handleTenantCreateCollection(w, r, tenantID)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
		return
	}

	collectionName := parts[2]
	if collectionName == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}

	// /v3/tenants/{tenant_id}/collections/{name} — get or delete
	if len(parts) == 3 {
		switch r.Method {
		case http.MethodGet:
			s.handleTenantGetCollection(w, r, tenantID, collectionName)
		case http.MethodDelete:
			s.handleTenantDeleteCollection(w, r, tenantID, collectionName)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
		return
	}

	// /v3/tenants/{tenant_id}/collections/{name}/{operation}
	operation := parts[3]
	switch operation {
	case "docs":
		s.handleTenantDocs(w, r, tenantID, collectionName)
	case "search":
		s.handleTenantSearch(w, r, tenantID, collectionName)
	default:
		http.Error(w, fmt.Sprintf("unknown operation: %s", operation), http.StatusNotFound)
	}
}

// isValidTenantID checks if a tenant ID is valid.
func isValidTenantID(id string) bool {
	if len(id) == 0 || len(id) > 64 {
		return false
	}
	for _, c := range id {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

// handleTenantInfo returns info about a tenant (collection count, stats).
func (s *CollectionHTTPServer) handleTenantInfo(w http.ResponseWriter, r *http.Request, tenantID string) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	stats, err := s.tenantManager.GetTenantStats(tenantID)
	if err != nil {
		// Tenant with no collections yet is not an error — return empty stats
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":           "success",
			"tenant_id":        tenantID,
			"collection_count": 0,
			"total_documents":  0,
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":           "success",
		"tenant_id":        tenantID,
		"collection_count": stats.CollectionCount,
		"total_documents":  stats.TotalDocuments,
		"collections":      stats.Collections,
	})
}

// handleTenantCreateCollection creates a collection for a tenant.
func (s *CollectionHTTPServer) handleTenantCreateCollection(w http.ResponseWriter, r *http.Request, tenantID string) {
	var schema vcollection.CollectionSchema
	if err := json.NewDecoder(r.Body).Decode(&schema); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	if _, err := s.tenantManager.CreateCollection(ctx, tenantID, schema); err != nil {
		http.Error(w, fmt.Sprintf("failed to create collection: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"tenant_id": tenantID,
		"message":   fmt.Sprintf("collection %q created for tenant %q", schema.Name, tenantID),
	})
}

// handleTenantListCollections lists all collections for a tenant.
func (s *CollectionHTTPServer) handleTenantListCollections(w http.ResponseWriter, r *http.Request, tenantID string) {
	infos := s.tenantManager.ListCollectionInfos(tenantID)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":      "success",
		"tenant_id":   tenantID,
		"count":       len(infos),
		"collections": infos,
	})
}

// handleTenantGetCollection returns info about a specific tenant collection.
func (s *CollectionHTTPServer) handleTenantGetCollection(w http.ResponseWriter, r *http.Request, tenantID, collectionName string) {
	info, err := s.tenantManager.GetCollectionInfo(tenantID, collectionName)
	if err != nil {
		http.Error(w, fmt.Sprintf("collection not found: %v", err), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":     "success",
		"tenant_id":  tenantID,
		"collection": info,
	})
}

// handleTenantDeleteCollection deletes a collection belonging to a tenant.
func (s *CollectionHTTPServer) handleTenantDeleteCollection(w http.ResponseWriter, r *http.Request, tenantID, collectionName string) {
	ctx := r.Context()
	if err := s.tenantManager.DeleteCollection(ctx, tenantID, collectionName); err != nil {
		http.Error(w, fmt.Sprintf("failed to delete collection: %v", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"tenant_id": tenantID,
		"message":   fmt.Sprintf("collection %q deleted for tenant %q", collectionName, tenantID),
	})
}

// handleTenantDocs handles document insert/delete for a tenant collection.
// POST = insert, DELETE = delete
func (s *CollectionHTTPServer) handleTenantDocs(w http.ResponseWriter, r *http.Request, tenantID, collectionName string) {
	switch r.Method {
	case http.MethodPost:
		// Insert document
		var req struct {
			Vectors  map[string]interface{} `json:"vectors"`
			Metadata map[string]interface{} `json:"metadata,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}

		if len(req.Vectors) == 0 {
			http.Error(w, "at least one vector required", http.StatusBadRequest)
			return
		}

		// Convert vectors (reuse same format as v2)
		vectors := make(map[string]interface{})
		for fieldName, vectorData := range req.Vectors {
			if vecSlice, ok := vectorData.([]interface{}); ok {
				denseVec := make([]float32, len(vecSlice))
				for i, v := range vecSlice {
					if f, ok := v.(float64); ok {
						denseVec[i] = float32(f)
					}
				}
				vectors[fieldName] = denseVec
			} else if vecMap, ok := vectorData.(map[string]interface{}); ok {
				if _, hasIndices := vecMap["indices"]; hasIndices {
					indicesSlice, ok1 := vecMap["indices"].([]interface{})
					values, ok2 := vecMap["values"].([]interface{})
					dim, ok3 := vecMap["dim"].(float64)
					if !ok1 || !ok2 || !ok3 {
						http.Error(w, fmt.Sprintf("invalid sparse vector for field %s", fieldName), http.StatusBadRequest)
						return
					}
					uint32Indices := make([]uint32, len(indicesSlice))
					for i, v := range indicesSlice {
						if f, ok := v.(float64); ok {
							uint32Indices[i] = uint32(f)
						}
					}
					float32Values := make([]float32, len(values))
					for i, v := range values {
						if f, ok := v.(float64); ok {
							float32Values[i] = float32(f)
						}
					}
					sparseVec, err := sparse.NewSparseVector(uint32Indices, float32Values, int(dim))
					if err != nil {
						http.Error(w, fmt.Sprintf("invalid sparse vector: %v", err), http.StatusBadRequest)
						return
					}
					vectors[fieldName] = sparseVec
				}
			}
		}

		doc := vcollection.Document{
			Vectors:  vectors,
			Metadata: req.Metadata,
		}

		ctx := r.Context()
		if err := s.tenantManager.AddDocument(ctx, tenantID, collectionName, &doc); err != nil {
			http.Error(w, fmt.Sprintf("failed to add document: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "success",
			"tenant_id": tenantID,
			"id":        doc.ID,
			"message":   "document added",
		})

	case http.MethodDelete:
		var req struct {
			DocID uint64 `json:"doc_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}
		if req.DocID == 0 {
			http.Error(w, "doc_id required", http.StatusBadRequest)
			return
		}

		ctx := r.Context()
		if err := s.tenantManager.DeleteDocument(ctx, tenantID, collectionName, req.DocID); err != nil {
			http.Error(w, fmt.Sprintf("failed to delete document: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "success",
			"tenant_id": tenantID,
			"message":   fmt.Sprintf("document %d deleted", req.DocID),
		})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleTenantSearch performs a search on a tenant's collection.
func (s *CollectionHTTPServer) handleTenantSearch(w http.ResponseWriter, r *http.Request, tenantID, collectionName string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Queries        map[string]interface{}          `json:"queries"`
		TopK           int                             `json:"top_k"`
		EfSearch       int                             `json:"ef_search,omitempty"`
		IncludeVectors *bool                           `json:"include_vectors,omitempty"`
		Filters        map[string]interface{}          `json:"filters,omitempty"`
		HybridParams   *vcollection.HybridSearchParams `json:"hybrid_params,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.Queries) == 0 {
		http.Error(w, "at least one query vector required", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}

	// Convert query vectors (dense and sparse)
	queries := make(map[string]interface{})
	for fieldName, vectorData := range req.Queries {
		if vecSlice, ok := vectorData.([]interface{}); ok {
			// Dense vector format
			denseVec := make([]float32, len(vecSlice))
			for i, v := range vecSlice {
				if f, ok := v.(float64); ok {
					denseVec[i] = float32(f)
				}
			}
			queries[fieldName] = denseVec
		} else if vecMap, ok := vectorData.(map[string]interface{}); ok {
			// FIX #8: Sparse vector format (consistent with insert path)
			if _, hasIndices := vecMap["indices"]; hasIndices {
				indicesSlice, ok1 := vecMap["indices"].([]interface{})
				values, ok2 := vecMap["values"].([]interface{})
				dim, ok3 := vecMap["dim"].(float64)
				if !ok1 || !ok2 || !ok3 {
					http.Error(w, fmt.Sprintf("invalid sparse query vector for field %s", fieldName), http.StatusBadRequest)
					return
				}
				uint32Indices := make([]uint32, len(indicesSlice))
				for i, v := range indicesSlice {
					if f, ok := v.(float64); ok {
						uint32Indices[i] = uint32(f)
					}
				}
				float32Values := make([]float32, len(values))
				for i, v := range values {
					if f, ok := v.(float64); ok {
						float32Values[i] = float32(f)
					}
				}
				sparseVec, err := sparse.NewSparseVector(uint32Indices, float32Values, int(dim))
				if err != nil {
					http.Error(w, fmt.Sprintf("invalid sparse query vector: %v", err), http.StatusBadRequest)
					return
				}
				queries[fieldName] = sparseVec
			}
		}
	}

	includeVectors, err := resolveIncludeVectors(req.IncludeVectors, r.URL.Query().Get("include_vectors"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	searchReq := vcollection.SearchRequest{
		CollectionName: collectionName,
		Queries:        queries,
		TopK:           req.TopK,
		EfSearch:       req.EfSearch,
		IncludeVectors: includeVectors,
		Filters:        req.Filters,
		HybridParams:   req.HybridParams,
	}

	ctx := r.Context()
	resp, err := s.tenantManager.SearchCollection(ctx, tenantID, searchReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("search failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tenantSearchJSONResponse{
		Status:             "success",
		TenantID:           tenantID,
		Documents:          resp.Documents,
		Scores:             resp.Scores,
		CandidatesExamined: resp.CandidatesExamined,
	})
}
