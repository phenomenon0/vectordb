package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"unsafe"

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

func ensureJSONEOF(dec *json.Decoder) error {
	var trailing interface{}
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON value")
		}
		return fmt.Errorf("trailing data: %w", err)
	}
	return nil
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
	durableStore  *vcollection.DurableStore  // Canonical Linux persistence boundary

	persistenceMu       sync.Mutex
	snapshotMetadata    vcollection.CollectionSnapshotMetadata
	persistenceErr      error
	collectionStorePath string
}

// NewCollectionHTTPServer creates a new HTTP server wrapper for CollectionManager
func NewCollectionHTTPServer(storagePath string) *CollectionHTTPServer {
	return &CollectionHTTPServer{
		manager:             vcollection.NewCollectionManager(storagePath),
		tenantManager:       vcollection.NewTenantManager(storagePath),
		collectionStorePath: storagePath,
	}
}

// Save persists V2 and V3 state as one checksummed generation.
func (s *CollectionHTTPServer) Save(basePath string) error {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if s.durableStore != nil {
		if basePath != "" && basePath != s.collectionStorePath {
			return fmt.Errorf("durable collection store path is fixed at %q", s.collectionStorePath)
		}
		if err := s.durableStore.Checkpoint(); err != nil {
			s.persistenceErr = err
			return err
		}
		s.persistenceErr = nil
		return nil
	}
	if basePath == "" {
		return nil
	}
	metadata := s.snapshotMetadata
	var zero [16]byte
	if metadata.StoreID == zero {
		var err error
		metadata, err = vcollection.NewCollectionSnapshotMetadata()
		if err != nil {
			return err
		}
	}
	if err := vcollection.SaveUnifiedCollectionSnapshot(basePath, s.manager, s.tenantManager, metadata); err != nil {
		s.persistenceErr = err
		return err
	}
	s.snapshotMetadata = metadata
	s.persistenceErr = nil
	s.collectionStorePath = basePath
	return nil
}

// Load stages and validates the complete V2/V3 generation before swapping
// either live manager. A fresh store is durably initialized before success.
func (s *CollectionHTTPServer) Load(basePath string) error {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if s.durableStore != nil {
		return errors.New("cannot legacy-load over an open durable collection store")
	}
	if basePath == "" {
		return nil
	}
	manager, tenants, metadata, err := vcollection.OpenUnifiedCollectionSnapshot(basePath, s.collectionStorePath)
	if err != nil {
		s.persistenceErr = err
		return err
	}
	s.manager = manager
	s.tenantManager = tenants
	s.snapshotMetadata = metadata
	s.collectionStorePath = basePath
	s.persistenceErr = nil
	return nil
}

// LoadDurable opens the canonical Linux-only journaled store. All V3 and gRPC
// mutations subsequently flow through the returned TenantManager and its
// append-before-apply boundary.
func (s *CollectionHTTPServer) LoadDurable(basePath string) error {
	return s.loadDurable(basePath, nil)
}

// LoadDurableWithLimits opens the canonical store with shared HTTP/gRPC
// admission limits for new tenants and collections.
func (s *CollectionHTTPServer) LoadDurableWithLimits(basePath string, limits vcollection.StoreLimits) error {
	return s.loadDurable(basePath, &limits)
}

func (s *CollectionHTTPServer) loadDurable(basePath string, limits *vcollection.StoreLimits) error {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if basePath == "" {
		return errors.New("durable collection store path cannot be empty")
	}
	if s.durableStore != nil {
		return errors.New("durable collection store is already open")
	}
	legacyArtifacts, err := existingLegacyV2CollectionArtifacts(basePath)
	if err != nil {
		s.persistenceErr = err
		return err
	}
	if len(legacyArtifacts) > 0 {
		err := fmt.Errorf(
			"legacy V2 collection persistence requires an explicit offline migration before canonical startup: %v",
			legacyArtifacts,
		)
		s.persistenceErr = err
		return err
	}
	var store *vcollection.DurableStore
	if limits == nil {
		store, err = vcollection.OpenDurableStore(basePath, s.collectionStorePath)
	} else {
		store, err = vcollection.OpenDurableStoreWithLimits(basePath, s.collectionStorePath, *limits)
	}
	if err != nil {
		s.persistenceErr = err
		return err
	}
	s.durableStore = store
	s.tenantManager = store.Tenants()
	s.snapshotMetadata = store.Metadata()
	s.collectionStorePath = basePath
	s.persistenceErr = nil
	return nil
}

func existingLegacyV2CollectionArtifacts(basePath string) ([]string, error) {
	artifacts := make([]string, 0, 2)
	for _, path := range []string{basePath + ".manager", basePath + ".tenants"} {
		if _, err := os.Lstat(path); err == nil {
			artifacts = append(artifacts, path)
		} else if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("inspect legacy V2 collection artifact %q: %w", path, err)
		}
	}
	return artifacts, nil
}

func (s *CollectionHTTPServer) setPersistenceError(err error) {
	s.persistenceMu.Lock()
	s.persistenceErr = err
	s.persistenceMu.Unlock()
}

// PersistenceError returns the startup or checkpoint failure that makes the
// collection subsystem unsafe to serve.
func (s *CollectionHTTPServer) PersistenceError() error {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if s.persistenceErr == nil && s.durableStore != nil {
		return s.durableStore.Err()
	}
	return s.persistenceErr
}

// IsDurable reports whether the canonical journal and lifetime lock opened.
func (s *CollectionHTTPServer) IsDurable() bool {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	return s.durableStore != nil
}

// Close checkpoints and releases the lifetime lock for a durable store. It is
// called only after all HTTP and gRPC handlers have drained.
func (s *CollectionHTTPServer) Close() error {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if s.durableStore == nil {
		return nil
	}
	if err := s.durableStore.Close(); err != nil {
		s.persistenceErr = err
		return err
	}
	s.persistenceErr = nil
	return nil
}

// Abort releases an unopened production runtime without checkpointing. It is
// used only when startup is refused before listeners begin serving.
func (s *CollectionHTTPServer) Abort() error {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if s.durableStore == nil {
		return nil
	}
	if err := s.durableStore.Abort(); err != nil {
		s.persistenceErr = err
		return err
	}
	s.persistenceErr = nil
	return nil
}

// Manager returns the legacy V2 manager for explicit migration/compatibility
// code. Canonical durable state is never installed into this raw manager.
func (s *CollectionHTTPServer) Manager() *vcollection.CollectionManager {
	return s.manager
}

// LegacyCollectionCount is the checked canonical startup inspection for V2
// collections embedded in an already-unified snapshot. It does not expose the
// durable store's raw CollectionManager to request code.
func (s *CollectionHTTPServer) LegacyCollectionCount() (int, error) {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if s.durableStore != nil {
		return s.durableStore.LegacyCollectionCount()
	}
	return s.manager.CollectionCount(), nil
}

// TenantManager returns the canonical tenant-aware manager shared by V3 HTTP
// and gRPC in the production RC.
func (s *CollectionHTTPServer) TenantManager() *vcollection.TenantManager {
	return s.tenantManager
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

	// Binary bulk import (zero-copy dense vectors)
	mux.HandleFunc("/v2/import", guard(s.handleBinaryImport))

	// Recommend and Discover APIs
	mux.HandleFunc("/v2/recommend", guard(s.handleRecommend))
	mux.HandleFunc("/v2/discover", guard(s.handleDiscover))

	// Multi-tenant endpoints (v3)
	// Tenant collection management
	mux.HandleFunc("/v3/tenants/", guard(s.handleTenantRoutes))
}

// RegisterCanonicalHandlers exposes only the tenant-aware RC contract. Legacy
// V2, bulk import, recommend, and discover handlers are deliberately absent.
func (s *CollectionHTTPServer) RegisterCanonicalHandlers(mux *http.ServeMux, guard func(http.HandlerFunc) http.HandlerFunc) {
	mux.HandleFunc("/v3/tenants/", guard(func(w http.ResponseWriter, r *http.Request) {
		if !s.IsDurable() {
			http.Error(w, "durable collection persistence required", http.StatusServiceUnavailable)
			return
		}
		if err := s.PersistenceError(); err != nil {
			http.Error(w, "collection persistence unavailable", http.StatusServiceUnavailable)
			return
		}
		s.handleTenantRoutes(w, r)
	}))
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
		// /v2/collections/{name}/stats|get
		operation := parts[1]
		if operation == "stats" && r.Method == http.MethodGet {
			s.handleCollectionStats(w, r, collectionName)
		} else if operation == "get" && r.Method == http.MethodPost {
			s.handleCollectionGetDocuments(w, r, collectionName)
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

func (s *CollectionHTTPServer) handleCollectionGetDocuments(w http.ResponseWriter, r *http.Request, name string) {
	if _, err := s.manager.GetCollection(name); err != nil {
		http.Error(w, fmt.Sprintf("collection not found: %v", err), http.StatusNotFound)
		return
	}

	var req struct {
		IDs []uint64 `json:"ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if len(req.IDs) == 0 {
		http.Error(w, "at least one id required", http.StatusBadRequest)
		return
	}

	documents := make([]vcollection.Document, 0, len(req.IDs))
	for _, id := range req.IDs {
		doc, err := s.manager.GetDocument(name, id)
		if err != nil {
			continue
		}
		documents = append(documents, *doc)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"documents": documents,
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

	if len(req.Docs) == 0 {
		http.Error(w, "no documents provided", http.StatusBadRequest)
		return
	}
	if len(req.Docs) > limitMaxBatchSize {
		http.Error(w, fmt.Sprintf("batch too large: max %d documents (set LIMIT_MAX_BATCH_SIZE to increase)", limitMaxBatchSize), http.StatusBadRequest)
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
	IncludeVectors *bool                           `json:"include_vectors,omitempty"` // include vectors in response (nil = default false)
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
		http.Error(w, "top_k must be positive", http.StatusBadRequest)
		return
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

func (s *CollectionHTTPServer) handleRecommend(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var raw struct {
		Collection     string                 `json:"collection"`
		Field          string                 `json:"field"`
		PositiveIDs    []uint64               `json:"positive_ids"`
		NegativeIDs    []uint64               `json:"negative_ids"`
		NegativeWeight float32                `json:"negative_weight"`
		TopK           int                    `json:"top_k"`
		EfSearch       int                    `json:"ef_search"`
		Filters        map[string]interface{} `json:"filters,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if raw.Collection == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}
	if len(raw.PositiveIDs) == 0 {
		http.Error(w, "at least one positive_id required", http.StatusBadRequest)
		return
	}
	if raw.TopK <= 0 {
		raw.TopK = 10
	}

	req := vcollection.RecommendRequest{
		CollectionName: raw.Collection,
		FieldName:      raw.Field,
		PositiveIDs:    raw.PositiveIDs,
		NegativeIDs:    raw.NegativeIDs,
		NegativeWeight: raw.NegativeWeight,
		TopK:           raw.TopK,
		EfSearch:       raw.EfSearch,
		Filters:        raw.Filters,
	}

	resp, err := s.manager.Recommend(r.Context(), req)
	if err != nil {
		http.Error(w, fmt.Sprintf("recommend failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *CollectionHTTPServer) handleDiscover(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var raw struct {
		Collection   string                    `json:"collection"`
		Field        string                    `json:"field"`
		TargetID     uint64                    `json:"target_id"`
		TargetVector []float32                 `json:"target_vector"`
		Context      []vcollection.ContextPair `json:"context"`
		TopK         int                       `json:"top_k"`
		EfSearch     int                       `json:"ef_search"`
		Filters      map[string]interface{}    `json:"filters,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if raw.Collection == "" {
		http.Error(w, "collection name required", http.StatusBadRequest)
		return
	}
	if len(raw.Context) == 0 {
		http.Error(w, "at least one context pair required", http.StatusBadRequest)
		return
	}
	if raw.TopK <= 0 {
		raw.TopK = 10
	}

	req := vcollection.DiscoverRequest{
		CollectionName: raw.Collection,
		FieldName:      raw.Field,
		TargetID:       raw.TargetID,
		TargetVector:   raw.TargetVector,
		Context:        raw.Context,
		TopK:           raw.TopK,
		EfSearch:       raw.EfSearch,
		Filters:        raw.Filters,
	}

	resp, err := s.manager.Discover(r.Context(), req)
	if err != nil {
		http.Error(w, fmt.Sprintf("discover failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
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

	// /v3/tenants/{tenant_id} — tenant info
	if len(parts) == 1 {
		if !authorizeCanonicalHTTP(w, r, tenantID, "", "admin") {
			return
		}
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
			if !authorizeCanonicalHTTP(w, r, tenantID, "", "admin") {
				return
			}
			s.handleTenantListCollections(w, r, tenantID)
		case http.MethodPost:
			// Reject cross-tenant and under-privileged callers before decoding a
			// potentially large schema. Collection allowlist scope is evaluated
			// after the schema name is available in the handler.
			if !authorizeCanonicalHTTPPermission(w, r, tenantID, "admin") {
				return
			}
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
			if !authorizeCanonicalHTTP(w, r, tenantID, collectionName, "read") {
				return
			}
			s.handleTenantGetCollection(w, r, tenantID, collectionName)
		case http.MethodDelete:
			if !authorizeCanonicalHTTP(w, r, tenantID, collectionName, "admin") {
				return
			}
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
		if !authorizeCanonicalHTTP(w, r, tenantID, collectionName, "write") {
			return
		}
		if len(parts) == 5 {
			if parts[4] != "batch" {
				http.Error(w, fmt.Sprintf("unknown document operation: %s", parts[4]), http.StatusNotFound)
				return
			}
			s.handleTenantBatchDocs(w, r, tenantID, collectionName)
			return
		}
		s.handleTenantDocs(w, r, tenantID, collectionName)
	case "search":
		if !authorizeCanonicalHTTP(w, r, tenantID, collectionName, "read") {
			return
		}
		s.handleTenantSearch(w, r, tenantID, collectionName)
	default:
		http.Error(w, fmt.Sprintf("unknown operation: %s", operation), http.StatusNotFound)
	}
}

func authorizeCanonicalHTTP(w http.ResponseWriter, r *http.Request, tenantID, collection, permission string) bool {
	tenantCtx, _ := security.GetTenantContextFromContext(r.Context())
	return writeCanonicalHTTPAuthorizationResult(
		w,
		security.AuthorizeTenantAccess(tenantCtx, tenantID, collection, permission),
	)
}

func authorizeCanonicalHTTPPermission(w http.ResponseWriter, r *http.Request, tenantID, permission string) bool {
	tenantCtx, _ := security.GetTenantContextFromContext(r.Context())
	return writeCanonicalHTTPAuthorizationResult(
		w,
		security.AuthorizeTenantPermission(tenantCtx, tenantID, permission),
	)
}

func writeCanonicalHTTPAuthorizationResult(w http.ResponseWriter, err error) bool {
	if err == nil {
		return true
	}
	if security.IsAuthorizationFailure(err, security.AuthorizationUnauthenticated) {
		http.Error(w, "unauthorized: "+err.Error(), http.StatusUnauthorized)
		return false
	}
	http.Error(w, "forbidden: "+err.Error(), http.StatusForbidden)
	return false
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

func decodeCanonicalVector(fieldName string, value interface{}) (interface{}, error) {
	switch vector := value.(type) {
	case []interface{}:
		dense := make([]float32, len(vector))
		for i, item := range vector {
			number, ok := item.(float64)
			if !ok || math.IsNaN(number) || math.IsInf(number, 0) {
				return nil, fmt.Errorf("field %s dense element %d must be a finite number", fieldName, i)
			}
			dense[i] = float32(number)
			if math.IsInf(float64(dense[i]), 0) {
				return nil, fmt.Errorf("field %s dense element %d exceeds float32 range", fieldName, i)
			}
		}
		return dense, nil

	case map[string]interface{}:
		indicesRaw, indicesOK := vector["indices"].([]interface{})
		valuesRaw, valuesOK := vector["values"].([]interface{})
		dimRaw, dimOK := vector["dim"].(float64)
		if !indicesOK || !valuesOK || !dimOK || dimRaw < 1 || dimRaw != math.Trunc(dimRaw) || dimRaw > float64(limitMaxDimension) {
			return nil, fmt.Errorf("field %s sparse vector requires integer dim in [1,%d] plus indices and values arrays", fieldName, limitMaxDimension)
		}
		if len(indicesRaw) != len(valuesRaw) {
			return nil, fmt.Errorf("field %s sparse indices and values lengths differ", fieldName)
		}
		indices := make([]uint32, len(indicesRaw))
		values := make([]float32, len(valuesRaw))
		for i, item := range indicesRaw {
			number, ok := item.(float64)
			if !ok || number < 0 || number != math.Trunc(number) || number > math.MaxUint32 {
				return nil, fmt.Errorf("field %s sparse index %d must be a uint32", fieldName, i)
			}
			indices[i] = uint32(number)
		}
		for i, item := range valuesRaw {
			number, ok := item.(float64)
			if !ok || math.IsNaN(number) || math.IsInf(number, 0) {
				return nil, fmt.Errorf("field %s sparse value %d must be a finite number", fieldName, i)
			}
			values[i] = float32(number)
			if math.IsInf(float64(values[i]), 0) {
				return nil, fmt.Errorf("field %s sparse value %d exceeds float32 range", fieldName, i)
			}
		}
		result, err := sparse.NewSparseVector(indices, values, int(dimRaw))
		if err != nil {
			return nil, fmt.Errorf("field %s sparse vector: %w", fieldName, err)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("field %s must be a dense array or sparse vector object", fieldName)
	}
}

// handleTenantInfo returns info about a tenant (collection count, stats).
func canonicalPersistenceUnavailable(err error) bool {
	return errors.Is(err, vcollection.ErrDurableStoreClosed) || errors.Is(err, vcollection.ErrDurableStoreFaulted)
}

func writeCanonicalOperationError(w http.ResponseWriter, prefix string, err error, fallbackStatus int) {
	if canonicalPersistenceUnavailable(err) {
		http.Error(w, "collection persistence unavailable", http.StatusServiceUnavailable)
		return
	}
	if errors.Is(err, vcollection.ErrSearchResponseBudgetExceeded) {
		http.Error(w, fmt.Sprintf("%s: %v", prefix, err), http.StatusRequestEntityTooLarge)
		return
	}
	if errors.Is(err, vcollection.ErrTenantLimitExceeded) || errors.Is(err, vcollection.ErrCollectionLimitExceeded) {
		http.Error(w, fmt.Sprintf("%s: %v", prefix, err), http.StatusTooManyRequests)
		return
	}
	http.Error(w, fmt.Sprintf("%s: %v", prefix, err), fallbackStatus)
}

func (s *CollectionHTTPServer) handleTenantInfo(w http.ResponseWriter, r *http.Request, tenantID string) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	stats, err := s.tenantManager.GetTenantStats(tenantID)
	if err != nil {
		if canonicalPersistenceUnavailable(err) {
			writeCanonicalOperationError(w, "tenant info unavailable", err, http.StatusInternalServerError)
			return
		}
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
	r.Body = http.MaxBytesReader(w, r.Body, limitInsertBody)
	var schema vcollection.CollectionSchema
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&schema); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if !vcollection.IsValidCanonicalIdentifier(schema.Name) {
		http.Error(w, "invalid collection name: must be 1-64 alphanumeric/hyphen/underscore characters", http.StatusBadRequest)
		return
	}
	if !authorizeCanonicalHTTP(w, r, tenantID, schema.Name, "admin") {
		return
	}

	ctx := r.Context()
	if _, err := s.tenantManager.CreateCollection(ctx, tenantID, schema); err != nil {
		writeCanonicalOperationError(w, "failed to create collection", err, http.StatusBadRequest)
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
	infos, err := s.tenantManager.ListCollectionInfosChecked(tenantID)
	if err != nil {
		writeCanonicalOperationError(w, "failed to list collections", err, http.StatusInternalServerError)
		return
	}

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
		writeCanonicalOperationError(w, "collection not found", err, http.StatusNotFound)
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
		writeCanonicalOperationError(w, "failed to delete collection", err, http.StatusBadRequest)
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
		r.Body = http.MaxBytesReader(w, r.Body, limitInsertBody)
		var req struct {
			ID       uint64                 `json:"id,omitempty"`
			Vectors  map[string]interface{} `json:"vectors"`
			Metadata map[string]interface{} `json:"metadata,omitempty"`
		}
		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}
		if err := ensureJSONEOF(dec); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}

		if len(req.Vectors) == 0 {
			http.Error(w, "at least one vector required", http.StatusBadRequest)
			return
		}

		vectors := make(map[string]interface{}, len(req.Vectors))
		for fieldName, vectorData := range req.Vectors {
			vector, err := decodeCanonicalVector(fieldName, vectorData)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			vectors[fieldName] = vector
		}

		doc := vcollection.Document{
			ID:       req.ID,
			Vectors:  vectors,
			Metadata: req.Metadata,
		}

		ctx := r.Context()
		if err := s.tenantManager.AddDocument(ctx, tenantID, collectionName, &doc); err != nil {
			writeCanonicalOperationError(w, "failed to add document", err, http.StatusInternalServerError)
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
		r.Body = http.MaxBytesReader(w, r.Body, limitDeleteBody)
		var req struct {
			DocID uint64 `json:"doc_id"`
		}
		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}
		if err := ensureJSONEOF(dec); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}
		if req.DocID == 0 {
			http.Error(w, "doc_id required", http.StatusBadRequest)
			return
		}

		ctx := r.Context()
		if err := s.tenantManager.DeleteDocument(ctx, tenantID, collectionName, req.DocID); err != nil {
			writeCanonicalOperationError(w, "failed to delete document", err, http.StatusInternalServerError)
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

// handleTenantBatchDocs atomically validates a bounded canonical batch before
// handing it to the shared durable mutation boundary. Partial-success modes are
// intentionally not part of the RC contract.
func (s *CollectionHTTPServer) handleTenantBatchDocs(w http.ResponseWriter, r *http.Request, tenantID, collectionName string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Journal records are capped at 16 MiB. Keep the HTTP representation below
	// that bound so normalization/envelope overhead cannot create an unjournalable
	// acknowledged request.
	const maxCanonicalBatchBody = int64(12 << 20)
	r.Body = http.MaxBytesReader(w, r.Body, maxCanonicalBatchBody)
	var req struct {
		Documents []struct {
			ID       uint64                 `json:"id,omitempty"`
			Vectors  map[string]interface{} `json:"vectors"`
			Metadata map[string]interface{} `json:"metadata,omitempty"`
		} `json:"documents"`
	}
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if len(req.Documents) == 0 {
		http.Error(w, "at least one document required", http.StatusBadRequest)
		return
	}
	if len(req.Documents) > vcollection.CanonicalMaxBatchDocuments {
		http.Error(w, fmt.Sprintf("batch too large: maximum is %d documents", vcollection.CanonicalMaxBatchDocuments), http.StatusRequestEntityTooLarge)
		return
	}

	docs := make([]vcollection.Document, len(req.Documents))
	for i, input := range req.Documents {
		if len(input.Vectors) == 0 {
			http.Error(w, fmt.Sprintf("document %d requires at least one vector", i), http.StatusBadRequest)
			return
		}
		vectors := make(map[string]interface{}, len(input.Vectors))
		for fieldName, raw := range input.Vectors {
			vector, err := decodeCanonicalVector(fieldName, raw)
			if err != nil {
				http.Error(w, fmt.Sprintf("document %d: %v", i, err), http.StatusBadRequest)
				return
			}
			vectors[fieldName] = vector
		}
		docs[i] = vcollection.Document{ID: input.ID, Vectors: vectors, Metadata: input.Metadata}
	}

	if err := s.tenantManager.BatchAddDocuments(r.Context(), tenantID, collectionName, docs); err != nil {
		writeCanonicalOperationError(w, "failed to add document batch", err, http.StatusInternalServerError)
		return
	}
	ids := make([]uint64, len(docs))
	for i := range docs {
		ids[i] = docs[i].ID
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"tenant_id": tenantID,
		"ids":       ids,
		"inserted":  len(ids),
	})
}

// handleTenantSearch performs a search on a tenant's collection.
func (s *CollectionHTTPServer) handleTenantSearch(w http.ResponseWriter, r *http.Request, tenantID, collectionName string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, limitQueryBody)
	var req struct {
		Queries        map[string]interface{}          `json:"queries"`
		TopK           int                             `json:"top_k"`
		EfSearch       int                             `json:"ef_search,omitempty"`
		IncludeVectors *bool                           `json:"include_vectors,omitempty"`
		Filters        map[string]interface{}          `json:"filters,omitempty"`
		HybridParams   *vcollection.HybridSearchParams `json:"hybrid_params,omitempty"`
	}
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.Queries) == 0 {
		http.Error(w, "at least one query vector required", http.StatusBadRequest)
		return
	}
	if len(req.Queries) > vcollection.CanonicalMaxSearchFields {
		http.Error(w, fmt.Sprintf("at most %d query fields are supported", vcollection.CanonicalMaxSearchFields), http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		http.Error(w, "top_k must be positive", http.StatusBadRequest)
		return
	}
	if req.TopK > vcollection.CanonicalMaxSearchTopK {
		http.Error(w, fmt.Sprintf("top_k must not exceed %d", vcollection.CanonicalMaxSearchTopK), http.StatusBadRequest)
		return
	}

	queries := make(map[string]interface{}, len(req.Queries))
	for fieldName, vectorData := range req.Queries {
		vector, err := decodeCanonicalVector(fieldName, vectorData)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		queries[fieldName] = vector
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
		writeCanonicalOperationError(w, "search failed", err, http.StatusInternalServerError)
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

// handleBinaryImport handles POST /v2/import for high-throughput binary vector import.
//
// Wire format (little-endian, numpy-compatible):
//
//	[uint32: count] [uint32: dim]
//	[uint64: id₁] [float32 × dim: vector₁]
//	[uint64: id₂] [float32 × dim: vector₂]
//	...
//
// Query params: collection, field
// Content-Type: application/octet-stream
func (s *CollectionHTTPServer) handleBinaryImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	collectionName := r.URL.Query().Get("collection")
	fieldName := r.URL.Query().Get("field")
	if collectionName == "" || fieldName == "" {
		http.Error(w, "collection and field query params required", http.StatusBadRequest)
		return
	}

	// Cap request body to prevent OOM
	maxBodySize := limitBinaryImport
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)

	// Read 8-byte header: [uint32 count][uint32 dim]
	var header [8]byte
	if _, err := io.ReadFull(r.Body, header[:]); err != nil {
		http.Error(w, fmt.Sprintf("failed to read header: %v", err), http.StatusBadRequest)
		return
	}

	count := binary.LittleEndian.Uint32(header[0:4])
	dim := binary.LittleEndian.Uint32(header[4:8])

	if count == 0 {
		http.Error(w, "count must be > 0", http.StatusBadRequest)
		return
	}
	if dim == 0 || dim > uint32(limitMaxDimension) {
		http.Error(w, fmt.Sprintf("dim must be in [1, %d]", limitMaxDimension), http.StatusBadRequest)
		return
	}

	// Each record: 8 bytes (uint64 id) + dim*4 bytes (float32 vector)
	recordSize := 8 + uint64(dim)*4
	expectedSize := uint64(count) * recordSize
	if expectedSize > uint64(maxBodySize) {
		http.Error(w, fmt.Sprintf("payload too large: %d bytes", expectedSize), http.StatusRequestEntityTooLarge)
		return
	}

	// Read full payload
	body := make([]byte, expectedSize)
	if _, err := io.ReadFull(r.Body, body); err != nil {
		http.Error(w, fmt.Sprintf("failed to read body: %v, expected %d bytes", err, expectedSize), http.StatusBadRequest)
		return
	}

	// Decode vectors: zero-copy reinterpret on little-endian (amd64/arm64)
	ids := make([]uint64, count)
	vectors := make([][]float32, count)

	for i := uint32(0); i < count; i++ {
		offset := uint64(i) * recordSize

		// Read ID (little-endian uint64)
		ids[i] = binary.LittleEndian.Uint64(body[offset : offset+8])

		// Zero-copy decode float32 slice from body bytes.
		// Safe on little-endian architectures (amd64, arm64).
		// The body slice is kept alive for the duration of this handler.
		vecBytes := body[offset+8 : offset+8+uint64(dim)*4]

		// Alignment check: if the slice is 4-byte aligned, use unsafe reinterpret
		if uintptr(unsafe.Pointer(&vecBytes[0]))%4 == 0 {
			vectors[i] = unsafe.Slice((*float32)(unsafe.Pointer(&vecBytes[0])), dim)
		} else {
			// Fallback: copy for unaligned data
			vec := make([]float32, dim)
			for j := uint32(0); j < dim; j++ {
				vec[j] = math.Float32frombits(binary.LittleEndian.Uint32(vecBytes[j*4 : j*4+4]))
			}
			vectors[i] = vec
		}
	}

	// Insert via BulkAddDense
	if err := s.manager.BulkAddDense(r.Context(), collectionName, fieldName, ids, vectors); err != nil {
		http.Error(w, fmt.Sprintf("bulk import failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "success",
		"inserted": count,
	})
}
