package main

import (
	"bytes"
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

	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/graph"
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
		f32 := float32(f)
		// A finite-but-out-of-float32-range element would silently become
		// +Inf/-Inf and poison distance/similarity; reject it explicitly.
		if math.IsNaN(float64(f32)) || math.IsInf(float64(f32), 0) {
			return nil, fmt.Errorf("dense vector element %d out of float32 range", len(result))
		}
		result = append(result, f32)
	}

	// Consume the closing ']' and require the document to end there. Without
	// this, a trailing second array (or bracketed garbage) after the vector is
	// silently ignored even though it cannot be part of the vector.
	end, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("dense vector array missing closing ']': %w", err)
	}
	if d, ok := end.(json.Delim); !ok || d != ']' {
		return nil, fmt.Errorf("expected ']' at end of dense vector, got %v", end)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("unexpected trailing data after dense vector array: %v", err)
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
	durableStore  *vcollection.DurableStore  // canonical Linux persistence boundary
	embedder      *serverEmbedder            // process text embedder for `texts`; nil = none

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

// UsageLoaded reports whether the class B usage records this server opened
// with were the ones it was entitled to. A memory-only run has no sidecar to
// lose, so it reports true for the same reason a durable store with no
// sidecar does: the signal names a loss, not a file read (CTL-05).
func (s *CollectionHTTPServer) UsageLoaded() bool {
	s.persistenceMu.Lock()
	defer s.persistenceMu.Unlock()
	if s.durableStore == nil {
		return true
	}
	return s.durableStore.UsageLoaded()
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
// code. The canonical durable state is never installed into this raw manager.
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

// RegisterCanonicalHandlers exposes only the tenant-aware RC contract. Legacy
// V2, bulk import, recommend, and discover handlers are deliberately absent.
func (s *CollectionHTTPServer) RegisterCanonicalHandlers(mux *http.ServeMux, guard func(http.HandlerFunc) http.HandlerFunc) {
	mux.HandleFunc("/v3/tenants/", guard(func(w http.ResponseWriter, r *http.Request) {
		if !s.IsDurable() {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeUnavailable, "durable collection persistence required"))
			return
		}
		if err := s.PersistenceError(); err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeUnavailable, "collection persistence unavailable"))
			return
		}
		s.handleTenantRoutes(w, r)
	}))
}

// tenantSearchJSONResponse is a typed struct for tenant search JSON encoding.
type tenantSearchJSONResponse struct {
	Status             string                 `json:"status"`
	TenantID           string                 `json:"tenant_id"`
	Documents          []vcollection.Document `json:"documents"`
	Scores             []float32              `json:"scores"`
	CandidatesExamined int                    `json:"candidates_examined"`
	BestScore          float32                `json:"best_score,omitempty"`
	WeakMatch          bool                   `json:"weak_match"`
	FellBackTo         string                 `json:"fell_back_to,omitempty"`
	EmbeddedBy         map[string]string      `json:"embedded_by,omitempty"`
	ScoreDirection     string                 `json:"score_direction,omitempty"`
	QueryTimeMs        float64                `json:"query_time_ms"`
	RequestID          string                 `json:"request_id,omitempty"`
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
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "tenant ID required in URL path"))
		return
	}

	tenantID := parts[0]

	// Validate tenant ID format
	if !isValidTenantID(tenantID) {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "invalid tenant ID: must be 1-64 alphanumeric/hyphen/underscore characters"))
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
		apierror.WriteHTTP(w, apierror.New(apierror.CodeNotFound, "unknown resource; expected 'collections'"))
		return
	}

	// /v3/tenants/{tenant_id}/collections — list or create
	if len(parts) == 2 {
		switch r.Method {
		case http.MethodGet:
			// Discovery is read-gated: listing collection names and
			// schemas is the same class of read as searching them (CTL-04).
			if !authorizeCanonicalHTTP(w, r, tenantID, "", "read") {
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
			apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
		}
		return
	}

	collectionName := parts[2]
	if collectionName == "" {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "collection name required"))
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
			apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
		}
		return
	}

	// /v3/tenants/{tenant_id}/collections/{name}/{operation}
	operation := parts[3]
	switch operation {
	case "docs":
		// A fifth path segment addresses a single document: PUT upserts the
		// body under that numeric id, GET reads it back. The literal "batch"
		// keeps its collection-level route.
		if len(parts) == 5 && parts[4] != "" && parts[4] != "batch" {
			docID, err := parseDocIDPathPart(parts[4])
			if err != nil {
				apierror.WriteHTTP(w, apierror.New(apierror.CodeNotFound, err.Error()))
				return
			}
			switch r.Method {
			case http.MethodPut:
				if !authorizeCanonicalHTTP(w, r, tenantID, collectionName, "write") {
					return
				}
				s.handleTenantUpsertDoc(w, r, tenantID, collectionName, docID)
			case http.MethodGet:
				if !authorizeCanonicalHTTP(w, r, tenantID, collectionName, "read") {
					return
				}
				s.handleTenantGetDoc(w, r, tenantID, collectionName, docID)
			default:
				apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
			}
			return
		}
		if !authorizeCanonicalHTTP(w, r, tenantID, collectionName, "write") {
			return
		}
		if len(parts) == 5 && parts[4] == "batch" {
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
		apierror.WriteHTTP(w, apierror.New(apierror.CodeNotFound, fmt.Sprintf("unknown operation: %s", operation)))
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
		apierror.WriteHTTP(w, apierror.New(apierror.CodeUnauthenticated, "unauthorized: "+err.Error()))
		return false
	}
	apierror.WriteHTTP(w, apierror.New(apierror.CodePermissionDenied, "forbidden: "+err.Error()))
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

// decodeCanonicalVectorRaw decodes a vector field straight from its raw JSON
// form. Dense arrays — the ingest-dominant shape — take a boxing-free fast
// path; sparse objects and any other shape fall back to the generic decoder
// so their exact validation error contract is unchanged.
func decodeCanonicalVectorRaw(fieldName string, raw json.RawMessage) (interface{}, error) {
	trimmed := bytes.TrimLeft(raw, " \t\n\r")
	if len(trimmed) > 0 && trimmed[0] == '[' {
		vec, err := decodeDenseVectorFast(raw)
		if err != nil {
			return nil, fmt.Errorf("field %s: %w", fieldName, err)
		}
		return vec, nil
	}
	var generic interface{}
	if err := json.Unmarshal(raw, &generic); err != nil {
		return nil, fmt.Errorf("field %s: %v", fieldName, err)
	}
	return decodeCanonicalVector(fieldName, generic)
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

// writeCanonicalOperationError projects an engine error onto the wire:
// apierror.FromEngine classifies the sentinels, fallback names the code for
// anything the engine left unclassified.
func writeCanonicalOperationError(w http.ResponseWriter, err error, fallback string) {
	apierror.WriteHTTP(w, apierror.FromEngine(err, fallback))
}

func (s *CollectionHTTPServer) handleTenantInfo(w http.ResponseWriter, r *http.Request, tenantID string) {
	if r.Method != http.MethodGet {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
		return
	}

	stats, err := s.tenantManager.GetTenantStats(tenantID)
	if err != nil {
		if canonicalPersistenceUnavailable(err) {
			writeCanonicalOperationError(w, err, apierror.CodeInternal)
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
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}
	if !vcollection.IsValidCanonicalIdentifier(schema.Name) {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "invalid collection name: must be 1-64 alphanumeric/hyphen/underscore characters"))
		return
	}
	if !authorizeCanonicalHTTP(w, r, tenantID, schema.Name, "admin") {
		return
	}
	if aerr := resolveSchemaEmbedding(&schema, s.embedder); aerr != nil {
		apierror.WriteHTTP(w, aerr)
		return
	}

	ctx := r.Context()
	if _, err := s.tenantManager.CreateCollection(ctx, tenantID, schema); err != nil {
		writeCanonicalOperationError(w, err, apierror.CodeInvalidArgument)
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
		writeCanonicalOperationError(w, err, apierror.CodeInternal)
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
		writeCanonicalOperationError(w, err, apierror.CodeNotFound)
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
		writeCanonicalOperationError(w, err, apierror.CodeInvalidArgument)
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
			ID       uint64                     `json:"id,omitempty"`
			Vectors  map[string]json.RawMessage `json:"vectors"`
			Texts    map[string]string          `json:"texts,omitempty"`
			Metadata map[string]interface{}     `json:"metadata,omitempty"`
		}
		dec := json.NewDecoder(r.Body)
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
			return
		}
		if err := ensureJSONEOF(dec); err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
			return
		}

		if len(req.Vectors) == 0 && len(req.Texts) == 0 {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "at least one vector or text required"))
			return
		}

		vectors := make(map[string]interface{}, len(req.Vectors)+len(req.Texts))
		for fieldName, vectorData := range req.Vectors {
			vector, err := decodeCanonicalVectorRaw(fieldName, vectorData)
			if err != nil {
				apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, err.Error()))
				return
			}
			vectors[fieldName] = vector
		}
		if _, ok := s.applyTexts(w, tenantID, collectionName, req.Texts, vectors, false); !ok {
			return
		}

		doc := vcollection.Document{
			ID:       req.ID,
			Vectors:  vectors,
			Metadata: req.Metadata,
		}

		ctx := r.Context()
		if err := s.tenantManager.AddDocument(ctx, tenantID, collectionName, &doc); err != nil {
			writeCanonicalOperationError(w, err, apierror.CodeInternal)
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
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
			return
		}
		if err := ensureJSONEOF(dec); err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
			return
		}
		if req.DocID == 0 {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "doc_id required"))
			return
		}

		ctx := r.Context()
		if err := s.tenantManager.DeleteDocument(ctx, tenantID, collectionName, req.DocID); err != nil {
			writeCanonicalOperationError(w, err, apierror.CodeInternal)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "success",
			"tenant_id": tenantID,
			"message":   fmt.Sprintf("document %d deleted", req.DocID),
		})

	default:
		apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
	}
}

// parseDocIDPathPart resolves the final /docs/{doc_id} path segment. The
// literal "batch" is already routed before this helper runs, so any remaining
// value must be a numeric document id.
func parseDocIDPathPart(value string) (uint64, error) {
	docID, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("document id must be numeric, got %q", value)
	}
	if docID == 0 {
		return 0, fmt.Errorf("document id must be non-zero")
	}
	return docID, nil
}

// handleTenantUpsertDoc upserts a document under a caller-supplied path ID.
// PUT /v3/tenants/{t}/collections/{c}/docs/{id} — the ID is taken from the
// path, never from the body, so an acknowledgement always matches the address
// the caller used.
func (s *CollectionHTTPServer) handleTenantUpsertDoc(w http.ResponseWriter, r *http.Request, tenantID, collectionName string, docID uint64) {
	r.Body = http.MaxBytesReader(w, r.Body, limitInsertBody)
	var req struct {
		Vectors  map[string]json.RawMessage `json:"vectors"`
		Texts    map[string]string          `json:"texts,omitempty"`
		Metadata map[string]interface{}     `json:"metadata,omitempty"`
	}
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}
	if len(req.Vectors) == 0 && len(req.Texts) == 0 {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "at least one vector or text required"))
		return
	}

	vectors := make(map[string]interface{}, len(req.Vectors)+len(req.Texts))
	for fieldName, vectorData := range req.Vectors {
		vector, err := decodeCanonicalVectorRaw(fieldName, vectorData)
		if err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, err.Error()))
			return
		}
		vectors[fieldName] = vector
	}
	if _, ok := s.applyTexts(w, tenantID, collectionName, req.Texts, vectors, false); !ok {
		return
	}

	doc := vcollection.Document{
		ID:       docID,
		Vectors:  vectors,
		Metadata: req.Metadata,
	}
	if err := s.tenantManager.UpsertDocument(r.Context(), tenantID, collectionName, &doc); err != nil {
		writeCanonicalOperationError(w, err, apierror.CodeInternal)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"tenant_id": tenantID,
		"id":        docID,
		"message":   "document upserted",
	})
}

// handleTenantGetDoc serves a single document by caller-supplied ID.
func (s *CollectionHTTPServer) handleTenantGetDoc(w http.ResponseWriter, r *http.Request, tenantID, collectionName string, docID uint64) {
	doc, ok := s.tenantManager.GetDocument(tenantID, collectionName, docID)
	if !ok {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeNotFound, fmt.Sprintf("document %d not found", docID)))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"tenant_id": tenantID,
		"id":        doc.ID,
		"vectors":   doc.Vectors,
		"metadata":  doc.Metadata,
	})
}

// handleTenantBatchDocs atomically validates a bounded canonical batch before
// handing it to the shared durable mutation boundary. Partial-success modes are
// intentionally not part of the RC contract.
func (s *CollectionHTTPServer) handleTenantBatchDocs(w http.ResponseWriter, r *http.Request, tenantID, collectionName string) {
	if r.Method != http.MethodPost {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
		return
	}

	// Journal records are capped at 16 MiB. Keep the HTTP representation below
	// that bound so normalization/envelope overhead cannot create an unjournalable
	// acknowledged request.
	const maxCanonicalBatchBody = int64(12 << 20)
	r.Body = http.MaxBytesReader(w, r.Body, maxCanonicalBatchBody)
	var req struct {
		Documents []struct {
			ID       uint64                     `json:"id,omitempty"`
			Vectors  map[string]json.RawMessage `json:"vectors"`
			Texts    map[string]string          `json:"texts,omitempty"`
			Metadata map[string]interface{}     `json:"metadata,omitempty"`
		} `json:"documents"`
	}
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}
	if len(req.Documents) == 0 {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, "at least one document required"))
		return
	}
	if len(req.Documents) > vcollection.MaxBatchDocuments {
		apierror.WriteHTTP(w, apierror.New(apierror.CodePayloadTooLarge, fmt.Sprintf("batch too large: maximum is %d documents", vcollection.MaxBatchDocuments)))
		return
	}

	docs := make([]vcollection.Document, len(req.Documents))
	var fields []vcollection.FieldInfo // schema, loaded once if any document sends texts
	for i, input := range req.Documents {
		if len(input.Vectors) == 0 && len(input.Texts) == 0 {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("document %d requires at least one vector or text", i)))
			return
		}
		vectors := make(map[string]interface{}, len(input.Vectors)+len(input.Texts))
		for fieldName, raw := range input.Vectors {
			vector, err := decodeCanonicalVectorRaw(fieldName, raw)
			if err != nil {
				apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("document %d: %v", i, err)))
				return
			}
			vectors[fieldName] = vector
		}
		if len(input.Texts) > 0 {
			if fields == nil {
				info, err := s.tenantManager.GetCollectionInfo(tenantID, collectionName)
				if err != nil {
					writeCanonicalOperationError(w, err, apierror.CodeNotFound)
					return
				}
				fields = info.Fields
			}
			if _, aerr := resolveTexts(fields, s.embedder, input.Texts, vectors, false); aerr != nil {
				aerr.Message = fmt.Sprintf("document %d: %s", i, aerr.Message)
				apierror.WriteHTTP(w, aerr)
				return
			}
		}
		docs[i] = vcollection.Document{ID: input.ID, Vectors: vectors, Metadata: input.Metadata}
	}

	if err := s.tenantManager.BatchAddDocuments(r.Context(), tenantID, collectionName, docs); err != nil {
		writeCanonicalOperationError(w, err, apierror.CodeInternal)
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
		apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, limitQueryBody)
	var req struct {
		Queries        map[string]json.RawMessage      `json:"queries"`
		Texts          map[string]string               `json:"texts,omitempty"`
		TopK           int                             `json:"top_k"`
		EfSearch       int                             `json:"ef_search,omitempty"`
		IncludeVectors *bool                           `json:"include_vectors,omitempty"`
		Filters        map[string]interface{}          `json:"filters,omitempty"`
		HybridParams   *vcollection.HybridSearchParams `json:"hybrid_params,omitempty"`
		ScoreFloor     float64                         `json:"score_floor,omitempty"`
		Fallback       *vcollection.FallbackParams     `json:"fallback,omitempty"`
		UsageBoost     float64                         `json:"usage_boost,omitempty"`
	}
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}
	if err := ensureJSONEOF(dec); err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, fmt.Sprintf("invalid request: %v", err)))
		return
	}

	queries := make(map[string]interface{}, len(req.Queries)+len(req.Texts))
	for fieldName, vectorData := range req.Queries {
		vector, err := decodeCanonicalVectorRaw(fieldName, vectorData)
		if err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, err.Error()))
			return
		}
		queries[fieldName] = vector
	}
	embeddedBy, ok := s.applyTexts(w, tenantID, collectionName, req.Texts, queries, true)
	if !ok {
		return
	}

	includeVectors, err := resolveIncludeVectors(req.IncludeVectors, r.URL.Query().Get("include_vectors"))
	if err != nil {
		apierror.WriteHTTP(w, apierror.New(apierror.CodeInvalidArgument, err.Error()))
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
		ScoreFloor:     req.ScoreFloor,
		Fallback:       req.Fallback,
		UsageBoost:     req.UsageBoost,
	}

	ctx := r.Context()
	resp, err := s.tenantManager.SearchCollection(ctx, tenantID, searchReq)
	if err != nil {
		writeCanonicalOperationError(w, err, apierror.CodeInternal)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tenantSearchJSONResponse{
		Status:             "success",
		TenantID:           tenantID,
		Documents:          resp.Documents,
		Scores:             resp.Scores,
		CandidatesExamined: resp.CandidatesExamined,
		BestScore:          resp.BestScore,
		WeakMatch:          resp.WeakMatch,
		FellBackTo:         resp.FellBackTo,
		EmbeddedBy:         embeddedBy,
		ScoreDirection:     resp.ScoreDirection,
		QueryTimeMs:        resp.QueryTimeMs,
		RequestID:          requestIDFromContext(r.Context()),
	})
}
