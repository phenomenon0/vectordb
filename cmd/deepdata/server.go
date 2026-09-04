package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/index"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/releaseinfo"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/telemetry"
)

// ===========================================================================================
// REQUEST ID
// ===========================================================================================

// generateRequestID returns a 16-byte random hex string (32 chars).
func generateRequestID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// requestIDFromContext extracts the request ID from the context, or returns "".
func requestIDFromContext(ctx context.Context) string {
	if id, ok := ctx.Value(logging.RequestIDKey).(string); ok {
		return id
	}
	return ""
}

// truncateRequestID caps id to maxBytes without splitting a multi-byte UTF-8
// rune, so the echoed header and the logged request_id remain valid UTF-8 even
// after an arbitrary-length client-supplied ID is trimmed.
func truncateRequestID(id string, maxBytes int) string {
	if len(id) <= maxBytes {
		return id
	}
	cut := id[:maxBytes]
	for len(cut) > 0 && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut
}

// ===========================================================================================
// CONFIGURABLE REQUEST LIMITS
// Override via environment variables. Defaults are safe for most deployments.
// ===========================================================================================

var (
	// HTTP body size limits (bytes)
	limitInsertBody   = int64(envInt("LIMIT_INSERT_BODY_MB", 10)) * 1024 * 1024
	limitBatchBody    = int64(envInt("LIMIT_BATCH_BODY_MB", 50)) * 1024 * 1024
	limitQueryBody    = int64(envInt("LIMIT_QUERY_BODY_MB", 1)) * 1024 * 1024
	limitDeleteBody   = int64(envInt("LIMIT_DELETE_BODY_MB", 1)) * 1024 * 1024
	limitSnapshotBody = int64(envInt("LIMIT_SNAPSHOT_BODY_MB", 1024)) * 1024 * 1024
	limitBinaryImport = int64(envInt("LIMIT_BINARY_IMPORT_MB", 500)) * 1024 * 1024

	// Batch processing limits
	limitMaxBatchSize       = envInt("LIMIT_MAX_BATCH_SIZE", 10_000)
	limitMaxDocLength       = envInt("LIMIT_MAX_DOC_LENGTH", 1_000_000)
	limitMaxMetaKeys        = envInt("LIMIT_MAX_META_KEYS", 100)
	limitMaxMetaValueLength = envInt("LIMIT_MAX_META_VALUE_LEN", 10_000)
	limitMaxTotalBatchBytes = envInt("LIMIT_MAX_BATCH_BYTES", 100_000_000)

	// List/query result limits
	limitMaxListResults = envInt("LIMIT_MAX_LIST_RESULTS", 100_000)

	// Dimension limit
	limitMaxDimension = envInt("LIMIT_MAX_DIMENSION", 65_536)
)

// isValidCollectionName checks if a collection name contains only allowed characters.
// Valid names: 1-64 characters, alphanumeric, underscores, hyphens.
func isValidCollectionName(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	for _, c := range name {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

// validateVector checks that a vector contains no NaN or Inf values.
func validateVector(vec []float32) error {
	for i, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("invalid vector: NaN or Inf at index %d", i)
		}
	}
	return nil
}

// encodeResponse encodes v as JSON. The legacy Cowrie content negotiation was
// removed under SYS-03 together with the Cowrie dependency; JSON is the only
// wire format the release candidate speaks.
func encodeResponse(w http.ResponseWriter, _ *http.Request, v any) error {
	w.Header().Set("Content-Type", "application/json")
	return json.NewEncoder(w).Encode(v)
}

// sendResponse encodes and sends a response, logging any encoding errors.
// Use this instead of ignoring encodeResponse errors.
func sendResponse(w http.ResponseWriter, r *http.Request, v any) {
	if err := encodeResponse(w, r, v); err != nil {
		logging.Default().Error("failed to encode response", "error", err, "path", r.URL.Path, "request_id", requestIDFromContext(r.Context()))
	}
}

// decodeRequest decodes a JSON request body.
func decodeRequest(r *http.Request, v any) error {
	return json.NewDecoder(r.Body).Decode(v)
}

// newHTTPHandler retains the broad historical surface for focused compatibility
// tests and explicit migration tooling. Production serve uses
// newCanonicalHTTPHandler instead.
func newHTTPHandler(store *VectorStore, embedder Embedder, reranker Reranker, indexPath string) (http.Handler, *CollectionHTTPServer) {
	return newHTTPHandlerWithSurface(store, store.serverRuntime, embedder, reranker, indexPath, false)
}

// newCanonicalHTTPHandler serves the RC surface from the server runtime alone;
// the legacy engine is never constructed for it.
func newCanonicalHTTPHandler(rt *serverRuntime, embedder Embedder, reranker Reranker, indexPath string) (http.Handler, *CollectionHTTPServer) {
	return newHTTPHandlerWithSurface(nil, rt, embedder, reranker, indexPath, true)
}

// newHTTPHandlerWithSurface registers both surfaces. store is nil in canonical
// mode: every route that reaches it is a historical route, and canonicalRCSurface
// answers 404 for all of them before a handler runs.
func newHTTPHandlerWithSurface(store *VectorStore, rt *serverRuntime, embedder Embedder, reranker Reranker, indexPath string, canonicalOnly bool) (http.Handler, *CollectionHTTPServer) {
	mux := http.NewServeMux()
	var collectionHTTP *CollectionHTTPServer
	rt.ensureLimiters(canonicalOnly)

	// Authentication, global rate limiting and (canonical only) per-tenant
	// rate limiting all live on the server runtime.
	guard := rt.httpGuard(canonicalOnly)

	mux.HandleFunc("/insert", withMetrics("insert", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context from request context (set by guard middleware)
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		tenantID := tenantCtx.TenantID

		// Check write permission
		if !tenantCtx.Permissions["write"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: write permission required", http.StatusForbidden)
			return
		}

		// Per-tenant rate limiting
		if store.tenantRL != nil && !store.tenantRL.allow(tenantID) {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}

		// Add request size limit
		r.Body = http.MaxBytesReader(w, r.Body, limitInsertBody)

		var req struct {
			ID         string            `json:"id"`
			Doc        string            `json:"doc"`
			Meta       map[string]string `json:"meta"`
			Upsert     bool              `json:"upsert"`
			Collection string            `json:"collection"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		// Input validation
		if req.Doc == "" {
			http.Error(w, "doc required", http.StatusBadRequest)
			return
		}
		if len(req.Doc) > limitMaxDocLength {
			http.Error(w, fmt.Sprintf("doc too large: max %d bytes", limitMaxDocLength), http.StatusBadRequest)
			return
		}
		if len(req.Meta) > limitMaxMetaKeys {
			http.Error(w, fmt.Sprintf("too many metadata keys: max %d", limitMaxMetaKeys), http.StatusBadRequest)
			return
		}
		for k, v := range req.Meta {
			if len(k) > limitMaxMetaValueLength || len(v) > limitMaxMetaValueLength {
				http.Error(w, fmt.Sprintf("metadata key or value too large: max %d bytes", limitMaxMetaValueLength), http.StatusBadRequest)
				return
			}
		}

		// Check collection access via ACL
		if req.Collection == "" {
			req.Collection = "default"
		}
		// Validate collection name to prevent issues with special characters
		if !isValidCollectionName(req.Collection) {
			http.Error(w, "invalid collection name: must be 1-64 alphanumeric characters, underscores, or hyphens", http.StatusBadRequest)
			return
		}
		if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[req.Collection] {
			http.Error(w, fmt.Sprintf("forbidden: no access to collection '%s'", req.Collection), http.StatusForbidden)
			return
		}

		// Start telemetry span for insert operation
		_, span := telemetry.StartInsert(r.Context(), req.Collection, req.ID)
		defer span.End()

		vec, err := embedder.Embed(req.Doc)
		if err != nil {
			telemetry.RecordError(span, err)
			logging.Default().LogError(r.Context(), "embed", err, "path", r.URL.Path)
			http.Error(w, "embedding failed", http.StatusInternalServerError)
			return
		}

		if err := validateVector(vec); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		start := time.Now()
		var id string
		if req.Upsert {
			id, err = store.Upsert(vec, req.Doc, req.ID, req.Meta, req.Collection, tenantID)
		} else {
			id, err = store.Add(vec, req.Doc, req.ID, req.Meta, req.Collection, tenantID)
		}
		if err != nil {
			telemetry.RecordError(span, err)
			logging.Default().LogError(r.Context(), "insert", err, "collection", req.Collection, "tenant_id", tenantID)
			http.Error(w, "failed to insert document", http.StatusInternalServerError)
			return
		}

		insertDur := time.Since(start)
		logging.Default().Insert(r.Context(), id, req.Collection, len(vec), insertDur)
		telemetry.RecordOK(span)
		telemetry.InsertRequestsTotal.WithLabelValues("dense").Inc()
		telemetry.InsertDurationSeconds.WithLabelValues("dense").Observe(insertDur.Seconds())
		if err := encodeResponse(w, r, map[string]any{"id": id}); err != nil {
			logging.Default().LogError(r.Context(), "encode_response", err)
		}
	})))

	mux.HandleFunc("/batch_insert", withMetrics("batch_insert", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context from request context (set by guard middleware)
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		tenantID := tenantCtx.TenantID

		// Check write permission
		if !tenantCtx.Permissions["write"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: write permission required", http.StatusForbidden)
			return
		}

		// Per-tenant rate limiting
		if store.tenantRL != nil && !store.tenantRL.allow(tenantID) {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}

		// Add request size limit
		r.Body = http.MaxBytesReader(w, r.Body, limitBatchBody)

		var req struct {
			Docs []struct {
				ID         string            `json:"id"`
				Doc        string            `json:"doc"`
				Meta       map[string]string `json:"meta"`
				Collection string            `json:"collection"`
			} `json:"docs"`
			Upsert bool `json:"upsert"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if len(req.Docs) == 0 {
			http.Error(w, "no docs provided", http.StatusBadRequest)
			return
		}
		if len(req.Docs) > limitMaxBatchSize {
			http.Error(w, fmt.Sprintf("batch too large: max %d docs (set LIMIT_MAX_BATCH_SIZE to increase)", limitMaxBatchSize), http.StatusBadRequest)
			return
		}

		// Check total batch size to prevent memory exhaustion
		var totalBytes int64
		for _, d := range req.Docs {
			totalBytes += int64(len(d.Doc))
			if totalBytes > int64(limitMaxTotalBatchBytes) {
				http.Error(w, fmt.Sprintf("batch total size exceeds limit: max %d bytes", limitMaxTotalBatchBytes), http.StatusBadRequest)
				return
			}
		}

		ids := make([]string, 0, len(req.Docs))
		var errors []string

		start := time.Now()

		for i, d := range req.Docs {
			if d.Doc == "" {
				errors = append(errors, fmt.Sprintf("doc %d: empty document", i))
				continue
			}
			if len(d.Doc) > limitMaxDocLength {
				errors = append(errors, fmt.Sprintf("doc %d: too large", i))
				continue
			}
			if len(d.Meta) > limitMaxMetaKeys {
				errors = append(errors, fmt.Sprintf("doc %d: too many metadata keys", i))
				continue
			}

			// Check collection access
			collection := d.Collection
			if collection == "" {
				collection = "default"
			}
			if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[collection] {
				errors = append(errors, fmt.Sprintf("doc %d: no access to collection '%s'", i, collection))
				continue
			}

			vec, err := embedder.Embed(d.Doc)
			if err != nil {
				logging.Default().Error("batch embed failed", "doc_index", i, "error", err, "request_id", requestIDFromContext(r.Context()))
				errors = append(errors, fmt.Sprintf("doc %d: embed failed", i))
				continue
			}

			var id string
			if req.Upsert {
				id, err = store.Upsert(vec, d.Doc, d.ID, d.Meta, collection, tenantID)
			} else {
				id, err = store.Add(vec, d.Doc, d.ID, d.Meta, collection, tenantID)
			}
			if err != nil {
				logging.Default().Error("batch insert failed", "doc_index", i, "error", err, "request_id", requestIDFromContext(r.Context()))
				errors = append(errors, fmt.Sprintf("doc %d: insert failed", i))
				continue
			}
			ids = append(ids, id)
		}

		// Removed synchronous save - rely on WAL + background snapshots
		response := map[string]any{"ids": ids}
		if len(errors) > 0 {
			response["errors"] = errors
			logging.Default().FromContext(r.Context()).Warn("batch insert partial failure", "errors", len(errors), "success", len(ids))
		} else {
			logging.Default().BatchInsert(r.Context(), req.Docs[0].Collection, len(ids), time.Since(start))
		}

		if err := encodeResponse(w, r, response); err != nil {
			logging.Default().LogError(r.Context(), "encode_response", err)
		}
	})))

	// Sparse vector insert endpoint - accepts pre-computed sparse vectors
	mux.HandleFunc("/insert/sparse", withMetrics("insert_sparse", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		tenantID := tenantCtx.TenantID

		if !tenantCtx.Permissions["write"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: write permission required", http.StatusForbidden)
			return
		}

		if store.tenantRL != nil && !store.tenantRL.allow(tenantID) {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, limitInsertBody)

		var req struct {
			ID         string            `json:"id"`
			Doc        string            `json:"doc"`
			Indices    []uint32          `json:"indices"`   // Sparse vector indices
			Values     []float32         `json:"values"`    // Sparse vector values
			Dimension  int               `json:"dimension"` // Total dimension
			Meta       map[string]string `json:"meta"`
			Upsert     bool              `json:"upsert"`
			Collection string            `json:"collection"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		// Validate sparse vector
		if len(req.Indices) == 0 || len(req.Values) == 0 {
			http.Error(w, "sparse vector indices and values required", http.StatusBadRequest)
			return
		}
		if len(req.Indices) != len(req.Values) {
			http.Error(w, "indices and values length mismatch", http.StatusBadRequest)
			return
		}
		if req.Dimension <= 0 {
			http.Error(w, "dimension must be positive", http.StatusBadRequest)
			return
		}
		if req.Dimension != store.Dim {
			http.Error(w, fmt.Sprintf("sparse vector dimension %d does not match store dimension %d", req.Dimension, store.Dim), http.StatusBadRequest)
			return
		}

		// Check collection access
		if req.Collection == "" {
			req.Collection = "default"
		}
		if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[req.Collection] {
			http.Error(w, fmt.Sprintf("forbidden: no access to collection '%s'", req.Collection), http.StatusForbidden)
			return
		}

		// Create sparse vector
		sparseVec := &SparseCoO{
			Indices: req.Indices,
			Values:  req.Values,
			Dim:     req.Dimension,
		}

		// Validate and normalize if needed
		if err := sparseVec.Validate(); err != nil {
			http.Error(w, fmt.Sprintf("invalid sparse vector: %v", err), http.StatusBadRequest)
			return
		}

		// Convert to dense for storage (temporary until VectorData integration)
		// TODO: Store as VectorData with sparse type
		vec := sparseVec.ToDense()

		var id string
		var err error
		if req.Upsert {
			id, err = store.Upsert(vec, req.Doc, req.ID, req.Meta, req.Collection, tenantID)
		} else {
			id, err = store.Add(vec, req.Doc, req.ID, req.Meta, req.Collection, tenantID)
		}
		if err != nil {
			logging.Default().LogError(r.Context(), "insert_sparse", err, "collection", req.Collection)
			http.Error(w, "failed to insert document", http.StatusInternalServerError)
			return
		}

		if err := encodeResponse(w, r, map[string]any{"id": id}); err != nil {
			logging.Default().LogError(r.Context(), "encode_response", err)
		}
	})))

	mux.HandleFunc("/query", withMetrics("query", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context from request context (set by guard middleware)
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		tenantID := tenantCtx.TenantID

		// Check read permission
		if !tenantCtx.Permissions["read"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: read permission required", http.StatusForbidden)
			return
		}

		// Per-tenant rate limiting
		if store.tenantRL != nil && !store.tenantRL.allow(tenantID) {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}

		// Add request size limit
		r.Body = http.MaxBytesReader(w, r.Body, limitQueryBody)

		var req struct {
			Query       string              `json:"query"`
			TopK        int                 `json:"top_k"`
			Mode        string              `json:"mode"` // ann or scan
			Meta        map[string]string   `json:"meta"`
			MetaAny     []map[string]string `json:"meta_any"`
			MetaNot     map[string]string   `json:"meta_not"`
			IncludeMeta bool                `json:"include_meta"`
			Collection  string              `json:"collection"`
			Offset      int                 `json:"offset"`
			Limit       int                 `json:"limit"`
			MetaRanges  []RangeFilter       `json:"meta_ranges"`
			HybridAlpha float64             `json:"hybrid_alpha"`
			ScoreMode   string              `json:"score_mode"` // "vector" (default), "hybrid", "lexical"
			EfSearch    int                 `json:"ef_search"`
			PageToken   string              `json:"page_token"`
			PageSize    int                 `json:"page_size"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		// Input validation
		const (
			MaxQueryLength = 10_000
			MaxTopK        = 1000
		)

		if len(req.Query) > MaxQueryLength {
			http.Error(w, fmt.Sprintf("query too long: max %d bytes", MaxQueryLength), http.StatusBadRequest)
			return
		}
		if req.TopK < 0 {
			http.Error(w, "top_k must be >= 0", http.StatusBadRequest)
			return
		}
		if req.Offset < 0 {
			http.Error(w, "offset must be >= 0", http.StatusBadRequest)
			return
		}
		if req.Limit < 0 {
			http.Error(w, "limit must be >= 0", http.StatusBadRequest)
			return
		}
		if req.PageSize < 0 {
			http.Error(w, "page_size must be >= 0", http.StatusBadRequest)
			return
		}

		// Start telemetry span for search operation
		_, span := telemetry.StartSearch(r.Context(), req.Collection, req.TopK, req.Mode)
		defer span.End()
		searchStart := time.Now()

		if req.TopK == 0 {
			req.TopK = 3
		}
		if req.TopK > MaxTopK {
			http.Error(w, fmt.Sprintf("top_k too large: max %d", MaxTopK), http.StatusBadRequest)
			return
		}
		if req.Limit == 0 || req.Limit > req.TopK {
			req.Limit = req.TopK
		}
		// Normalize and validate mode to prevent unbounded Prometheus label cardinality
		if req.Mode == "" {
			req.Mode = "ann"
		}
		switch req.Mode {
		case "ann", "scan", "lex":
			// valid
		default:
			// Unknown mode: treat as ANN but don't propagate arbitrary strings to metrics
			req.Mode = "ann"
		}

		// Auto scan-mode for small collections: ANN index may not return
		// recently inserted vectors until compaction. Use scan for small sets.
		scanThreshold := envInt("SCAN_THRESHOLD", 500)
		if req.Mode == "ann" && scanThreshold > 0 {
			store.RLock()
			collCount := 0
			for _, coll := range store.Coll {
				if coll == req.Collection || (req.Collection == "" && coll == "default") {
					collCount++
				}
			}
			store.RUnlock()
			if collCount > 0 && collCount < scanThreshold {
				req.Mode = "scan"
			}
		}
		if req.HybridAlpha == 0 {
			req.HybridAlpha = 0.5
		}
		if req.ScoreMode == "" {
			req.ScoreMode = "vector"
		}
		pageSize := req.Limit
		if req.PageSize > 0 {
			pageSize = req.PageSize
		}

		queryHash := hashQueryCursor(
			req.Query,
			req.TopK,
			pageSize,
			req.Limit,
			req.Meta,
			req.MetaAny,
			req.MetaNot,
			req.MetaRanges,
			req.Collection,
			req.Mode,
			req.ScoreMode,
			req.EfSearch,
			req.HybridAlpha,
			tenantID,
		)

		offset := req.Offset
		var cursor pageCursor
		hasCursor := false
		if req.PageToken != "" {
			v, err := decodePageToken(req.PageToken)
			if err != nil {
				http.Error(w, "invalid page token", http.StatusBadRequest)
				return
			}
			offset = v.Offset
			hasCursor = true
			cursor = v
			if v.FilterHash != queryHash {
				http.Error(w, "page token invalid for current query", http.StatusBadRequest)
				return
			}
		}

		qTokens := tokenize(req.Query)
		qVec := []float32{}
		var err error
		if req.Mode != "lex" {
			qVec, err = embedder.EmbedQuery(req.Query)
			if err != nil {
				logging.Default().LogError(r.Context(), "embed_query", err, "mode", req.Mode)
				http.Error(w, "embedding failed", http.StatusInternalServerError)
				return
			}
		}
		var ids []int
		if req.Mode == "scan" {
			ids = store.SearchScan(qVec, req.TopK, req.Collection)
		} else if req.Mode == "lex" {
			ids = store.SearchLex(qTokens, req.TopK)
		} else {
			ids = store.SearchANNWithParams(qVec, req.TopK, req.Collection, req.EfSearch)
		}
		resultItems := make([]queryResultItem, 0, len(ids))
		rangeCandidates := store.candidateIDsForRange(req.MetaRanges)

		for _, idx := range ids {
			hid := hashID(store.GetID(idx))

			// Tenant filtering: Only return vectors owned by requesting tenant
			// Admin can see all tenants, or if tenant is "default"
			vectorTenant := store.TenantID[hid]
			if vectorTenant == "" {
				vectorTenant = "default" // Backward compatibility
			}
			if !tenantCtx.IsAdmin && vectorTenant != tenantID {
				continue
			}

			// Collection ACL check
			collection := store.Coll[hid]
			if collection == "" {
				collection = "default"
			}
			if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[collection] {
				continue
			}

			if rangeCandidates != nil {
				if _, ok := rangeCandidates[hid]; !ok {
					continue
				}
			}
			meta := store.Meta[hid]
			if !matchesMeta(meta, req.Meta) {
				continue
			}
			if !matchesRanges(meta, store.NumMeta[hid], store.TimeMeta[hid], req.MetaRanges) {
				continue
			}
			if len(req.MetaAny) > 0 && !matchesAny(meta, req.MetaAny) {
				continue
			}
			if len(req.MetaNot) > 0 && matchesMeta(meta, req.MetaNot) {
				continue
			}
			if req.Collection != "" && store.Coll[hid] != req.Collection {
				continue
			}
			item := queryResultItem{
				doc: store.GetDoc(idx),
				id:  store.GetID(idx),
			}
			if req.IncludeMeta {
				cp := make(map[string]string, len(meta))
				for k, v := range meta {
					cp[k] = v
				}
				item.meta = cp
			}
			switch req.ScoreMode {
			case "hybrid":
				item.score = float32(store.hybridScore(hid, qVec, qTokens, req.HybridAlpha))
			case "lexical":
				item.score = float32(store.bm25(hid, qTokens))
			default: // vector
				if idx*store.Dim < len(store.Data) && (idx+1)*store.Dim <= len(store.Data) {
					item.score = DotProduct(qVec, store.Data[idx*store.Dim:(idx+1)*store.Dim])
				} else {
					item.score = 0 // Fallback for missing data
				}
			}
			// Safely get sequence number with bounds check
			var seq uint64
			if idx < len(store.Seqs) {
				seq = store.Seqs[idx]
			} else {
				seq = uint64(idx) // Fallback to index as sequence
			}
			item.seq = seq
			resultItems = append(resultItems, item)
		}
		if len(resultItems) > 0 {
			sort.Slice(resultItems, func(i, j int) bool {
				if resultItems[i].score == resultItems[j].score {
					return resultItems[i].seq < resultItems[j].seq
				}
				return resultItems[i].score > resultItems[j].score
			})
		}
		if hasCursor && cursor.Offset > 0 {
			if cursor.Offset > len(resultItems) || resultItems[cursor.Offset-1].seq != cursor.LastSeq {
				http.Error(w, "page token is stale; rerun the query", http.StatusBadRequest)
				return
			}
		}

		start := offset
		if start > len(resultItems) {
			start = len(resultItems)
		}
		end := start + pageSize
		if end > len(resultItems) {
			end = len(resultItems)
		}
		pageItems := make([]queryResultItem, 0, end-start)
		if start < end {
			pageItems = append(pageItems, resultItems[start:end]...)
		}

		nextPage := ""
		if end < len(resultItems) && len(pageItems) > 0 {
			lastSeq := pageItems[len(pageItems)-1].seq
			nextPage = encodePageToken(end, queryHash, lastSeq)
		}

		pageDocs := make([]string, 0, len(pageItems))
		for _, item := range pageItems {
			pageDocs = append(pageDocs, item.doc)
		}

		rDocs, rerankScores, stats, err := reranker.Rerank(req.Query, pageDocs, req.Limit)
		if err != nil {
			telemetry.RecordError(span, err)
			logging.Default().LogError(r.Context(), "rerank", err)
			http.Error(w, "reranking failed", http.StatusInternalServerError)
			return
		}

		pageItems = reorderQueryResultItems(pageItems, rDocs)

		respIDs := make([]string, 0, len(pageItems))
		respMeta := make([]map[string]string, 0, len(pageItems))
		respScores := make([]float32, 0, len(pageItems))
		for _, item := range pageItems {
			respIDs = append(respIDs, item.id)
			respScores = append(respScores, item.score)
			if req.IncludeMeta {
				respMeta = append(respMeta, item.meta)
			}
		}
		if len(respScores) == 0 {
			respScores = rerankScores
		}
		if !req.IncludeMeta {
			respMeta = nil
		}

		// Record search results in span
		searchDur := time.Since(searchStart)
		telemetry.RecordSearchResults(span, len(respIDs), 0) // latency tracked by HTTP middleware
		telemetry.RecordOK(span)
		telemetry.SearchRequestsTotal.WithLabelValues(req.Mode).Inc()
		telemetry.SearchDurationSeconds.WithLabelValues(req.Mode).Observe(searchDur.Seconds())
		logging.Default().Search(r.Context(), req.Collection, req.TopK, len(respIDs), searchDur)

		w.Header().Set("Content-Type", "application/json")

		// Build structured results array for web UI compatibility
		// UI reads: r.id, r.score, r.text||r.document, r.metadata
		results := make([]map[string]any, 0, len(respIDs))
		for i, id := range respIDs {
			item := map[string]any{"id": id}
			if i < len(rDocs) {
				item["text"] = rDocs[i]
				item["document"] = rDocs[i]
			}
			if i < len(respScores) {
				item["score"] = respScores[i]
			}
			if req.IncludeMeta && i < len(respMeta) {
				item["metadata"] = respMeta[i]
			}
			results = append(results, item)
		}

		response := map[string]any{
			"ids":     respIDs,
			"docs":    rDocs,
			"scores":  respScores,
			"stats":   stats,
			"meta":    respMeta,
			"next":    nextPage,
			"results": results,
		}

		if err := json.NewEncoder(w).Encode(response); err != nil {
			// Log error but don't change response (already started writing)
			logging.Default().LogError(r.Context(), "encode_response", err)
		}
		return
	})))

	// Sparse vector query endpoint - accepts sparse query vectors
	mux.HandleFunc("/query/sparse", withMetrics("query_sparse", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		tenantID := tenantCtx.TenantID

		if !tenantCtx.Permissions["read"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: read permission required", http.StatusForbidden)
			return
		}

		if store.tenantRL != nil && !store.tenantRL.allow(tenantID) {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, limitInsertBody)

		var req struct {
			Indices     []uint32            `json:"indices"`   // Sparse query indices
			Values      []float32           `json:"values"`    // Sparse query values
			Dimension   int                 `json:"dimension"` // Total dimension
			TopK        int                 `json:"top_k"`
			Collection  string              `json:"collection"`
			Meta        map[string]string   `json:"meta,omitempty"`
			MetaAny     []map[string]string `json:"meta_any,omitempty"`
			MetaNot     map[string]string   `json:"meta_not,omitempty"`
			IncludeMeta bool                `json:"include_meta"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		// Validate sparse query vector
		if len(req.Indices) == 0 || len(req.Values) == 0 {
			http.Error(w, "sparse query indices and values required", http.StatusBadRequest)
			return
		}
		if len(req.Indices) != len(req.Values) {
			http.Error(w, "indices and values length mismatch", http.StatusBadRequest)
			return
		}
		if req.Dimension <= 0 {
			http.Error(w, "dimension must be positive", http.StatusBadRequest)
			return
		}
		if req.Dimension != store.Dim {
			http.Error(w, fmt.Sprintf("sparse vector dimension %d does not match store dimension %d", req.Dimension, store.Dim), http.StatusBadRequest)
			return
		}
		if req.TopK <= 0 {
			req.TopK = 10
		}

		// Check collection access
		if req.Collection == "" {
			req.Collection = "default"
		}
		if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[req.Collection] {
			http.Error(w, fmt.Sprintf("forbidden: no access to collection '%s'", req.Collection), http.StatusForbidden)
			return
		}

		// Create sparse query vector
		sparseQuery := &SparseCoO{
			Indices: req.Indices,
			Values:  req.Values,
			Dim:     req.Dimension,
		}

		if err := sparseQuery.Validate(); err != nil {
			http.Error(w, fmt.Sprintf("invalid sparse query: %v", err), http.StatusBadRequest)
			return
		}

		// Convert to dense for search (temporary until sparse index integration)
		// TODO: Use sparse index directly
		qVec := sparseQuery.ToDense()

		// Search using ANN
		ids := store.SearchANN(qVec, req.TopK)

		// Collect results with filtering
		docs := make([]string, 0, len(ids))
		respIDs := make([]string, 0, len(ids))
		respMeta := make([]map[string]string, 0, len(ids))
		respScores := make([]float32, 0, len(ids))

		for _, idx := range ids {
			hid := hashID(store.GetID(idx))

			// Tenant filtering
			vectorTenant := store.TenantID[hid]
			if vectorTenant == "" {
				vectorTenant = "default"
			}
			if !tenantCtx.IsAdmin && vectorTenant != tenantID {
				continue
			}

			// Collection ACL check
			collection := store.Coll[hid]
			if collection == "" {
				collection = "default"
			}
			if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[collection] {
				continue
			}

			// Metadata filtering
			meta := store.Meta[hid]
			if !matchesMeta(meta, req.Meta) {
				continue
			}
			if len(req.MetaAny) > 0 && !matchesAny(meta, req.MetaAny) {
				continue
			}
			if len(req.MetaNot) > 0 && matchesMeta(meta, req.MetaNot) {
				continue
			}
			if req.Collection != "" && store.Coll[hid] != req.Collection {
				continue
			}

			docs = append(docs, store.GetDoc(idx))
			respIDs = append(respIDs, store.GetID(idx))
			if req.IncludeMeta {
				cp := make(map[string]string, len(meta))
				for k, v := range meta {
					cp[k] = v
				}
				respMeta = append(respMeta, cp)
			}

			// Compute cosine similarity between query and stored vector
			score := float32(0)
			if idx >= 0 && (idx+1)*store.Dim <= len(store.Data) {
				storedVec := store.Data[idx*store.Dim : (idx+1)*store.Dim]
				var dot, normQ, normS float32
				for d := 0; d < store.Dim && d < len(qVec); d++ {
					dot += qVec[d] * storedVec[d]
					normQ += qVec[d] * qVec[d]
					normS += storedVec[d] * storedVec[d]
				}
				if normQ > 0 && normS > 0 {
					score = dot / (float32(math.Sqrt(float64(normQ))) * float32(math.Sqrt(float64(normS))))
				}
			}
			respScores = append(respScores, score)
		}

		response := map[string]any{
			"ids":    respIDs,
			"docs":   docs,
			"scores": respScores,
			"meta":   respMeta,
		}

		if err := encodeResponse(w, r, response); err != nil {
			logging.Default().LogError(r.Context(), "encode_response", err)
		}
	})))

	mux.HandleFunc("/delete", withMetrics("delete", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context from request context (set by guard middleware)
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		tenantID := tenantCtx.TenantID

		// Check write permission
		if !tenantCtx.Permissions["write"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: write permission required", http.StatusForbidden)
			return
		}

		// Per-tenant rate limiting
		if store.tenantRL != nil && !store.tenantRL.allow(tenantID) {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, limitDeleteBody)

		var req struct {
			ID string `json:"id"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}
		if req.ID == "" {
			http.Error(w, "id required", http.StatusBadRequest)
			return
		}

		// Start telemetry span for delete operation
		_, span := telemetry.StartDelete(r.Context(), "default", req.ID)
		defer span.End()

		// SECURITY: Verify tenant ownership before deletion
		// This prevents cross-tenant data deletion attacks
		hid := hashID(req.ID)
		store.RLock()
		existingTenant := store.TenantID[hid]
		_, exists := store.idToIx[hid]
		isDeleted := store.Deleted[hid]
		store.RUnlock()

		// Check if document exists
		if !exists || isDeleted {
			http.Error(w, "document not found", http.StatusNotFound)
			return
		}

		// Verify ownership (admins can delete any document)
		if !tenantCtx.IsAdmin && existingTenant != "" && existingTenant != tenantID {
			http.Error(w, "forbidden: document belongs to different tenant", http.StatusForbidden)
			return
		}

		if err := store.Delete(req.ID); err != nil {
			telemetry.RecordError(span, err)
			logging.Default().LogError(r.Context(), "delete", err, "id", req.ID)
			http.Error(w, "failed to delete document", http.StatusInternalServerError)
			return
		}

		// Removed synchronous save - rely on WAL + background snapshots
		telemetry.RecordOK(span)

		if err := encodeResponse(w, r, map[string]any{"deleted": req.ID}); err != nil {
			logging.Default().LogError(r.Context(), "encode_response", err)
		}
	})))

	// Scroll API: paginated iteration over all documents (no search, just enumerate)
	mux.HandleFunc("/scroll", withMetrics("scroll", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		tenantID := tenantCtx.TenantID

		// Parse parameters from query string or JSON body
		collection := r.URL.Query().Get("collection")
		limit := 100
		offset := 0
		includeMeta := r.URL.Query().Get("include_meta") == "true"

		if v := r.URL.Query().Get("limit"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= limitMaxListResults {
				limit = n
			}
		}
		if v := r.URL.Query().Get("offset"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n >= 0 {
				offset = n
			}
		}

		store.RLock()
		defer store.RUnlock()

		ids := make([]string, 0, limit)
		docs := make([]string, 0, limit)
		metas := make([]map[string]string, 0, limit)
		total := 0

		for i := 0; i < store.Count; i++ {
			hid := hashID(store.GetID(i))

			// Skip deleted
			if store.Deleted[hid] {
				continue
			}

			// Tenant filtering
			vectorTenant := store.TenantID[hid]
			if vectorTenant == "" {
				vectorTenant = "default"
			}
			if !tenantCtx.IsAdmin && vectorTenant != tenantID {
				continue
			}

			// Collection filtering
			coll := store.Coll[hid]
			if coll == "" {
				coll = "default"
			}
			if collection != "" && coll != collection {
				continue
			}

			// Collection ACL
			if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[coll] {
				continue
			}

			total++
			if total <= offset {
				continue
			}
			if len(ids) >= limit {
				continue // keep counting total
			}

			ids = append(ids, store.GetID(i))
			docs = append(docs, store.GetDoc(i))
			if includeMeta {
				meta := store.Meta[hid]
				cp := make(map[string]string, len(meta))
				for k, v := range meta {
					cp[k] = v
				}
				metas = append(metas, cp)
			}
		}

		nextOffset := offset + len(ids)
		if nextOffset >= total {
			nextOffset = -1 // no more pages
		}

		response := map[string]any{
			"ids":         ids,
			"docs":        docs,
			"total":       total,
			"next_offset": nextOffset,
		}
		if includeMeta {
			response["meta"] = metas
		}

		sendResponse(w, r, response)
	})))

	mux.HandleFunc("/health", withMetrics("health", guard(func(w http.ResponseWriter, r *http.Request) {
		store.RLock()
		total := store.Count
		deleted := len(store.Deleted)
		active := total - deleted
		lastSaved := store.lastSaved
		walFault := store.walFault
		_, embedderIsONNX := embedder.(*OnnxEmbedder)
		_, embedderIsOpenAI := embedder.(*OpenAIEmbedder)
		_, embedderIsOllama := embedder.(*OllamaEmbedder)
		_, rerankerIsONNX := reranker.(*OnnxCrossEncoderReranker)
		store.RUnlock()
		snapAge := ageMillis(indexPath, lastSaved)
		walAge := ageMillis(store.walPath, time.Time{})

		// Determine embedder type
		embedderType := "hash"
		if embedderIsONNX {
			embedderType = "onnx"
		} else if embedderIsOpenAI {
			embedderType = "openai"
		} else if embedderIsOllama {
			embedderType = "ollama"
		}

		// Get collection counts (for dashboard)
		store.RLock()
		collectionCounts := make(map[string]int)
		for _, coll := range store.Coll {
			collectionCounts[coll]++
		}
		store.RUnlock()

		// Build collections list for response
		collections := make([]map[string]any, 0, len(collectionCounts))
		for name, count := range collectionCounts {
			collections = append(collections, map[string]any{
				"name":         name,
				"vector_count": count,
			})
		}

		// Build response with mode info
		healthy := walFault == nil
		response := map[string]any{
			"ok":              healthy,
			"total":           total,
			"active":          active,
			"deleted":         deleted,
			"hnsw_ids":        len(store.idToIx),
			"checksum":        store.checksum,
			"wal_bytes":       fileSize(store.walPath),
			"index_bytes":     fileSize(indexPath),
			"snapshot_age_ms": snapAge,
			"wal_age_ms":      walAge,
			"embedder": map[string]any{
				"type": embedderType,
			},
			"reranker": map[string]any{
				"type": map[bool]string{true: "onnx", false: "simple"}[rerankerIsONNX],
			},
			"collections": collections,
		}

		if walFault != nil {
			response["wal_error"] = walFault.Error()
		}

		// Add mode information if available
		if CurrentMode != nil {
			response["mode"] = GetModeInfo(CurrentMode)
		}

		sendResponse(w, r, response)
	})))

	// Prometheus metrics are an operational surface like the health probes,
	// but unlike probes they can leak request-volume/operation detail, so they
	// are gated behind the same guard used for API routes when REQUIRE_AUTH is
	// on. In credentialless dev mode the guard authorizes anonymous access.
	mux.Handle("/metrics", guard(globalMetrics.Handler().ServeHTTP))

	// GET /v3/status — the server describing itself: version, the operation
	// list, the embedder it will use, its limits and its capabilities. It
	// names no tenant, so it is gated on the caller's own read permission
	// (CTL-04).
	mux.Handle("/v3/status", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeMethodNotAllowed, "method not allowed"))
			return
		}
		tenantCtx, _ := security.GetTenantContextFromContext(r.Context())
		var ownTenant string
		if tenantCtx != nil {
			ownTenant = tenantCtx.TenantID
		}
		if !writeCanonicalHTTPAuthorizationResult(w, security.AuthorizeTenantPermission(tenantCtx, ownTenant, "read")) {
			return
		}
		var embedder *serverEmbedder
		usageLoaded := true
		readOnly := false
		if collectionHTTP != nil {
			embedder = collectionHTTP.embedder
			usageLoaded = collectionHTTP.UsageLoaded()
			if store := collectionHTTP.DurableStore(); store != nil {
				readOnly = store.IsReplica()
			}
		}
		payload, err := statusPayload(embedder, usageLoaded, readOnly, requestIDFromContext(r.Context()))
		if err != nil {
			apierror.WriteHTTP(w, apierror.New(apierror.CodeInternal, "status unavailable"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(payload)
	}))

	// Kubernetes-style health probes
	// /healthz - Liveness probe: Is the process alive and not deadlocked?
	livenessHandler := func(w http.ResponseWriter, r *http.Request) {
		// Liveness check: verify we can acquire locks (not deadlocked).
		// Use a context-aware pattern to avoid leaking goroutines when the
		// timeout fires while the lock is still held (e.g., during a snapshot).
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()

		done := make(chan struct{})
		go func() {
			// There is no legacy engine to deadlock on, so the
			// probe answers as it always did with an idle store: ok.
			if store != nil {
				store.RLock()
				store.RUnlock()
			}
			close(done)
		}()

		select {
		case <-done:
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("ok"))
		case <-ctx.Done():
			// Lock could not be acquired within timeout — possible deadlock
			// or long-running write operation (snapshot). The goroutine will
			// eventually complete when the lock is released and be collected.
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte("deadlock detected"))
		}
	}
	mux.HandleFunc("/healthz", livenessHandler)
	mux.HandleFunc("/livez", livenessHandler)

	// /readyz - Readiness probe: Is the service ready to accept traffic?
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if canonicalOnly {
			issues := []string{}
			type canonicalHealth struct {
				durable  bool
				readOnly bool
				err      error
			}
			health := make(chan canonicalHealth, 1)
			go func() {
				if collectionHTTP == nil {
					health <- canonicalHealth{}
					return
				}
				durable := collectionHTTP.IsDurable()
				var err error
				var readOnly bool
				if durable {
					err = collectionHTTP.PersistenceError()
					readOnly = collectionHTTP.DurableStore().IsReplica()
				}
				health <- canonicalHealth{durable: durable, readOnly: readOnly, err: err}
			}()
			var state canonicalHealth
			select {
			case state = <-health:
				if !state.durable {
					issues = append(issues, "durable collection store not initialized")
				} else if state.err != nil {
					logging.Default().Error("readyz: durable collection fault", "error", state.err)
					issues = append(issues, "durable collection store faulted")
				}
			case <-time.After(5 * time.Second):
				issues = append(issues, "durable collection health check timed out")
			}
			w.Header().Set("Content-Type", "application/json")
			if len(issues) == 0 {
				w.WriteHeader(http.StatusOK)
				// embedder names the process text embedder ("none" when
				// callers must send vectors); no live embedding call here.
				_ = json.NewEncoder(w).Encode(map[string]any{
					"ready":  true,
					"checks": []string{"collection_snapshot", "mutation_journal", "lifetime_lock"},
					// A read replica is ready -- it answers reads correctly --
					// so it stays 200 and stays in the pool. read_only is how
					// it declines writes to a load balancer that would
					// otherwise treat every ready backend as interchangeable.
					"read_only": state.readOnly,
					"embedder":  collectionHTTP.embedder.Label(),
					"version":   releaseinfo.Version(),
				})
			} else {
				w.WriteHeader(http.StatusServiceUnavailable)
				_ = json.NewEncoder(w).Encode(map[string]any{"ready": false, "issues": issues})
			}
			return
		}

		// Use a timeout to prevent readiness probes from hanging indefinitely
		// when the write lock is held during long snapshot operations.
		type storeState struct {
			ready      bool
			indexCount int
			walErr     error
		}

		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()

		stateCh := make(chan storeState, 1)
		go func() {
			store.RLock()
			s := storeState{
				ready:      store.Count >= 0 && store.Dim > 0,
				indexCount: len(store.indexes),
				walErr:     store.walFault,
			}
			store.RUnlock()
			stateCh <- s
		}()

		var state storeState
		var lockAcquired bool
		select {
		case state = <-stateCh:
			lockAcquired = true
		case <-ctx.Done():
			lockAcquired = false
		}

		issues := []string{}

		if !lockAcquired {
			issues = append(issues, "lock acquisition timeout — snapshot or heavy write in progress")
		} else {
			if !state.ready {
				issues = append(issues, "store not initialized")
			}
			if state.indexCount == 0 {
				issues = append(issues, "no index available")
			}
			if state.walErr != nil {
				logging.Default().Error("readyz: WAL fault", "error", state.walErr)
				issues = append(issues, "wal replay failed")
			}
		}

		// Check 2: Embedder is initialized.
		// Avoid live embedding calls in readiness probes to prevent external dependency/cost spikes.
		if embedder == nil {
			issues = append(issues, "embedder not initialized")
		}

		// Return result
		if len(issues) == 0 {
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ready":   true,
				"checks":  []string{"store", "index", "embedder_initialized"},
				"version": releaseinfo.Version(),
			})
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ready":  false,
				"issues": issues,
			})
		}
	})

	mux.HandleFunc("/integrity", withMetrics("integrity", guard(func(w http.ResponseWriter, r *http.Request) {
		store.RLock()
		ck := store.validateChecksum()
		indexOK := len(store.indexes) > 0
		store.RUnlock()
		sendResponse(w, r, map[string]any{
			"ok":          ck && indexOK,
			"checksum_ok": ck,
			"index_ok":    indexOK,
		})
	})))

	mux.HandleFunc("/compact", withMetrics("compact", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		if !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: admin permission required", http.StatusForbidden)
			return
		}
		if err := store.Compact(indexPath); err != nil {
			logging.Default().LogError(r.Context(), "compact", err)
			http.Error(w, "compact failed", http.StatusInternalServerError)
			return
		}
		sendResponse(w, r, map[string]any{"ok": true})
	})))

	// Export snapshot (read-only)
	mux.HandleFunc("/export", withMetrics("export", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		if !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: admin permission required", http.StatusForbidden)
			return
		}
		store.RLock()
		path := indexPath
		store.RUnlock()
		http.ServeFile(w, r, path)
	})))

	// Online snapshot import is deliberately unavailable in the single-node RC.
	mux.HandleFunc("/import", withMetrics("import", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		if !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: admin permission required", http.StatusForbidden)
			return
		}
		// Online import cannot safely cross the live WAL/checkpoint generation
		// boundary yet. The single-node RC therefore supports restore only while
		// the server is stopped; leaving this endpoint active could mix an old WAL
		// with imported state after a crash.
		http.Error(w, "online snapshot import is disabled; use the offline restore procedure", http.StatusNotImplemented)
		return
	})))

	// ==================================================================================
	// ADMIN API ENDPOINTS - ACL & Quota Management
	// ==================================================================================

	// Global administration middleware. A tenant-admin JWT remains constrained
	// to its tenant and optional collection scope; only the configured static
	// server credential may reach legacy global administration tooling.
	adminGuard := func(next http.HandlerFunc) http.HandlerFunc {
		return guard(func(w http.ResponseWriter, r *http.Request) {
			// Read tenant context from request context (set by guard middleware)
			tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
			if !ok || !tenantCtx.IsServerAdmin {
				http.Error(w, "forbidden: server admin permission required", http.StatusForbidden)
				return
			}
			next(w, r)
		})
	}

	// Grant collection access to a tenant
	mux.HandleFunc("/admin/acl/grant", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			TenantID   string `json:"tenant_id"`
			Collection string `json:"collection"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if req.TenantID == "" || req.Collection == "" {
			http.Error(w, "tenant_id and collection required", http.StatusBadRequest)
			return
		}

		store.acl.GrantCollectionAccess(req.TenantID, req.Collection)

		sendResponse(w, r, map[string]any{
			"ok":         true,
			"tenant_id":  req.TenantID,
			"collection": req.Collection,
			"action":     "granted",
		})
	}))

	// Revoke collection access from a tenant
	mux.HandleFunc("/admin/acl/revoke", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			TenantID   string `json:"tenant_id"`
			Collection string `json:"collection"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if req.TenantID == "" || req.Collection == "" {
			http.Error(w, "tenant_id and collection required", http.StatusBadRequest)
			return
		}

		store.acl.RevokeCollectionAccess(req.TenantID, req.Collection)

		sendResponse(w, r, map[string]any{
			"ok":         true,
			"tenant_id":  req.TenantID,
			"collection": req.Collection,
			"action":     "revoked",
		})
	}))

	// Grant permission to a tenant
	mux.HandleFunc("/admin/permission/grant", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			TenantID   string `json:"tenant_id"`
			Permission string `json:"permission"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if req.TenantID == "" || req.Permission == "" {
			http.Error(w, "tenant_id and permission required", http.StatusBadRequest)
			return
		}

		// Validate permission
		validPerms := map[string]bool{"read": true, "write": true, "admin": true}
		if !validPerms[req.Permission] {
			http.Error(w, "invalid permission: must be read, write, or admin", http.StatusBadRequest)
			return
		}

		store.acl.GrantPermission(req.TenantID, req.Permission)

		sendResponse(w, r, map[string]any{
			"ok":         true,
			"tenant_id":  req.TenantID,
			"permission": req.Permission,
			"action":     "granted",
		})
	}))

	// Revoke permission from a tenant
	mux.HandleFunc("/admin/permission/revoke", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			TenantID   string `json:"tenant_id"`
			Permission string `json:"permission"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if req.TenantID == "" || req.Permission == "" {
			http.Error(w, "tenant_id and permission required", http.StatusBadRequest)
			return
		}

		store.acl.RevokePermission(req.TenantID, req.Permission)

		sendResponse(w, r, map[string]any{
			"ok":         true,
			"tenant_id":  req.TenantID,
			"permission": req.Permission,
			"action":     "revoked",
		})
	}))

	// Set storage quota for a tenant
	mux.HandleFunc("/admin/quota/set", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			TenantID string `json:"tenant_id"`
			MaxBytes int64  `json:"max_bytes"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if req.TenantID == "" {
			http.Error(w, "tenant_id required", http.StatusBadRequest)
			return
		}

		if req.MaxBytes <= 0 {
			http.Error(w, "max_bytes must be positive", http.StatusBadRequest)
			return
		}

		store.quotas.SetQuota(req.TenantID, req.MaxBytes)

		sendResponse(w, r, map[string]any{
			"ok":        true,
			"tenant_id": req.TenantID,
			"max_bytes": req.MaxBytes,
		})
	}))

	// Get tenant quota and usage
	mux.HandleFunc("/admin/quota/", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenantID from URL path
		path := strings.TrimPrefix(r.URL.Path, "/admin/quota/")
		tenantID := strings.TrimSpace(path)

		if tenantID == "" {
			http.Error(w, "tenant_id required in URL path", http.StatusBadRequest)
			return
		}

		usedBytes, vectorCount := store.quotas.GetUsage(tenantID)
		maxBytes := store.quotas.GetQuota(tenantID)

		var utilizationPct float64
		if maxBytes > 0 {
			utilizationPct = float64(usedBytes) / float64(maxBytes) * 100
		}

		sendResponse(w, r, map[string]any{
			"tenant_id":       tenantID,
			"used_bytes":      usedBytes,
			"max_bytes":       maxBytes,
			"vector_count":    vectorCount,
			"utilization_pct": utilizationPct,
			"has_quota_limit": maxBytes > 0,
		})
	}))

	// Set per-tenant rate limit
	mux.HandleFunc("/admin/ratelimit/set", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			TenantID string `json:"tenant_id"`
			RPS      int    `json:"rps"`
			Burst    int    `json:"burst"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if req.TenantID == "" {
			http.Error(w, "tenant_id required", http.StatusBadRequest)
			return
		}

		if req.RPS <= 0 || req.Burst <= 0 {
			http.Error(w, "rps and burst must be positive", http.StatusBadRequest)
			return
		}

		if store.tenantRL != nil {
			store.tenantRL.setLimit(req.TenantID, req.RPS, req.Burst)
		}

		sendResponse(w, r, map[string]any{
			"ok":        true,
			"tenant_id": req.TenantID,
			"rps":       req.RPS,
			"burst":     req.Burst,
		})
	}))

	// Legacy Collection Management API endpoints (v1 - single-index collections)
	// NOTE: These are kept for backward compatibility. Use /v2/collections for multi-vector support.
	mux.HandleFunc("/admin/collection/create", withMetrics("collection_create", adminGuard(handleCollectionCreate(store))))
	mux.HandleFunc("/admin/collection/list", withMetrics("collection_list", adminGuard(handleCollectionList(store))))
	mux.HandleFunc("/admin/collection/stats", withMetrics("collection_stats_all", adminGuard(handleAllCollectionStats(store))))

	// Pattern-based routes for specific collection operations
	// Note: These handlers parse the collection name from the URL path
	mux.HandleFunc("/admin/collection/", withMetrics("collection_ops", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/admin/collection/")
		parts := strings.Split(path, "/")

		if len(parts) == 0 || parts[0] == "" {
			http.Error(w, "collection name required", http.StatusBadRequest)
			return
		}

		_ = parts[0] // collectionName used in handlers via path parsing

		// Route based on path structure and HTTP method
		if len(parts) == 1 {
			// /admin/collection/{name}
			switch r.Method {
			case http.MethodGet:
				handleCollectionGet(store)(w, r)
			case http.MethodDelete:
				handleCollectionDelete(store)(w, r)
			default:
				http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			}
		} else if len(parts) == 2 {
			// /admin/collection/{name}/stats or /admin/collection/{name}/config
			operation := parts[1]
			switch operation {
			case "stats":
				if r.Method == http.MethodGet {
					handleCollectionStats(store)(w, r)
				} else {
					http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				}
			case "config":
				if r.Method == http.MethodPut {
					handleCollectionUpdate(store)(w, r)
				} else {
					http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
				}
			default:
				http.Error(w, "unknown operation", http.StatusNotFound)
			}
		} else {
			http.Error(w, "invalid URL format", http.StatusBadRequest)
		}
	})))

	// NEW Multi-Vector Collection API (v2) - supports hybrid search with dense + sparse vectors
	// Initialize collection HTTP server for multi-vector support
	collectionBasePath := ""
	if indexPath != "" {
		collectionBasePath = indexPath + ".collections"
	}
	collectionHTTP = NewCollectionHTTPServer(collectionBasePath)
	collectionHTTP.embedder, _ = embedder.(*serverEmbedder)
	if collectionBasePath != "" {
		var err error
		if canonicalOnly {
			err = collectionHTTP.LoadDurableWithLimits(collectionBasePath, vcollection.StoreLimits{
				MaxTenants:     envInt("MAX_TENANTS", 100_000),
				MaxCollections: envInt("MAX_COLLECTIONS", 10_000),
			})
		} else {
			err = collectionHTTP.Load(collectionBasePath)
		}
		if err != nil {
			collectionHTTP.setPersistenceError(fmt.Errorf("load collection state: %w", err))
		} else if canonicalOnly {
			// Before a single route is registered: a replica directory that
			// came up unbound would take one local write and fork its history
			// from the leader's at the same LSN.
			if err := bindReplicaReadOnly(collectionHTTP, collectionBasePath); err != nil {
				collectionHTTP.setPersistenceError(err)
			}
		}
	}
	// Only the canonical tenant-aware V3 surface is served. Legacy V2/root
	// collection routes were removed; unsupported paths cannot be registered.
	collectionHTTP.RegisterCanonicalHandlers(mux, guard)

	// ==========================================================================
	// KNOWLEDGE GRAPH EXTRACTION API ENDPOINTS (v2)
	// ==========================================================================
	// LLM-based entity/relationship extraction from text
	// Endpoints: /v2/extract, /v2/extract/batch, /v2/extract/temporal, /v2/extract/status
	if !canonicalOnly {
		RegisterExtractionHandlers(mux)
	}

	// ==========================================================================
	// MODE & COST TRACKING API ENDPOINTS
	// ==========================================================================

	// GET /api/mode - Returns current mode information
	mux.HandleFunc("/api/mode", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		if CurrentMode == nil {
			sendResponse(w, r, map[string]any{
				"error": "mode not initialized",
			})
			return
		}

		sendResponse(w, r, GetModeInfo(CurrentMode))
	})

	// POST /api/config/embedder - Hot-swap the embedder at runtime
	mux.HandleFunc("/api/config/embedder", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			// Return current embedder info
			if CurrentMode != nil {
				sendResponse(w, r, map[string]any{
					"type":      CurrentMode.EmbedderType,
					"model":     CurrentMode.EmbedderModel,
					"dimension": CurrentMode.Dimension,
				})
			} else {
				sendResponse(w, r, map[string]any{"error": "mode not initialized"})
			}
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		if !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: admin permission required", http.StatusForbidden)
			return
		}

		var req struct {
			Type  string `json:"type"`  // "ollama", "openai", "hash"
			Model string `json:"model"` // model name
			URL   string `json:"url"`   // for ollama: base URL
			Key   string `json:"key"`   // API key for the provider
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}

		se, ok := embedder.(*SwappableEmbedder)
		if !ok {
			http.Error(w, "embedder is not swappable", http.StatusInternalServerError)
			return
		}

		var newEmb Embedder
		var newDim int
		var errMsg string
		var newEmbType, newEmbModel string

		switch req.Type {
		case "ollama":
			url := req.URL
			if url == "" {
				url = "http://localhost:11434"
			}
			model := req.Model
			if model == "" {
				model = "nomic-embed-text"
			}
			// Test connectivity
			client := &http.Client{Timeout: 5 * time.Second}
			resp, err := client.Get(url + "/api/tags")
			if err != nil {
				errMsg = "cannot reach Ollama at " + url + ": " + err.Error()
				break
			}
			resp.Body.Close()
			emb := NewOllamaEmbedder(url, model)
			// Test embed to get dimension
			vec, err := emb.Embed("dimension test")
			if err != nil {
				errMsg = "ollama embed failed: " + err.Error()
				break
			}
			newDim = len(vec)
			newEmb = emb
			newEmbType = "ollama"
			newEmbModel = model

		case "openai":
			key := req.Key
			if key == "" {
				key = os.Getenv("OPENAI_API_KEY")
			}
			if key == "" {
				errMsg = "OpenAI API key required (pass 'key' field or set OPENAI_API_KEY)"
				break
			}
			emb := NewOpenAIEmbedder(key)
			vec, err := emb.Embed("dimension test")
			if err != nil {
				errMsg = "openai embed failed: " + err.Error()
				break
			}
			newDim = len(vec)
			newEmb = emb
			newEmbType = "openai"
			newEmbModel = "text-embedding-3-small"

		case "hash":
			dim := 384
			if req.Model != "" {
				if d, err := strconv.Atoi(req.Model); err == nil && d > 0 {
					dim = d
				}
			}
			newEmb = NewHashEmbedder(dim)
			newDim = dim
			newEmbType = "hash"
			newEmbModel = fmt.Sprintf("hash-%d", dim)

		default:
			errMsg = "unknown embedder type: " + req.Type + " (valid: ollama, openai, hash)"
		}

		if errMsg != "" {
			sendResponse(w, r, map[string]any{"ok": false, "error": errMsg})
			return
		}

		// Block swaps that would change dimension — existing vectors become incompatible
		if newDim != store.Dim {
			sendResponse(w, r, map[string]any{
				"ok":    false,
				"error": fmt.Sprintf("cannot swap to embedder with dimension %d: store requires dimension %d (create a new collection for different dimensions)", newDim, store.Dim),
			})
			return
		}

		oldDim := se.Dim()
		se.Swap(newEmb)
		if CurrentMode != nil {
			CurrentMode.EmbedderType = newEmbType
			CurrentMode.EmbedderModel = newEmbModel
			CurrentMode.Dimension = newDim
		}

		sendResponse(w, r, map[string]any{
			"ok":                true,
			"type":              req.Type,
			"model":             CurrentMode.EmbedderModel,
			"dimension":         newDim,
			"dimension_changed": oldDim != newDim,
			"warning": func() string {
				if oldDim != newDim {
					return fmt.Sprintf("dimension changed from %d to %d — existing vectors will not be compatible", oldDim, newDim)
				}
				return ""
			}(),
		})
	}))

	// POST /api/config/keys - Store LLM API keys (kept in memory only, not persisted)
	mux.HandleFunc("/api/config/keys", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			// Return which keys are set (not the actual values)
			sendResponse(w, r, map[string]any{
				"openai":     os.Getenv("OPENAI_API_KEY") != "",
				"deepseek":   os.Getenv("DEEPSEEK_API_KEY") != "",
				"anthropic":  os.Getenv("ANTHROPIC_API_KEY") != "",
				"openrouter": os.Getenv("OPENROUTER_API_KEY") != "",
				"cerebras":   os.Getenv("CEREBRAS_API_KEY") != "",
			})
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		if !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: admin permission required", http.StatusForbidden)
			return
		}

		var req map[string]string
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}

		validKeys := map[string]string{
			"openai":     "OPENAI_API_KEY",
			"deepseek":   "DEEPSEEK_API_KEY",
			"anthropic":  "ANTHROPIC_API_KEY",
			"openrouter": "OPENROUTER_API_KEY",
			"cerebras":   "CEREBRAS_API_KEY",
		}

		set := []string{}
		for name, value := range req {
			envName, ok := validKeys[name]
			if !ok {
				continue
			}
			if value != "" {
				os.Setenv(envName, value)
				set = append(set, name)
			}
		}

		sendResponse(w, r, map[string]any{
			"ok":  true,
			"set": set,
		})
	}))

	// POST /api/embed - Embed a single text using server-side embedder
	mux.HandleFunc("/api/embed", withMetrics("embed", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		if !tenantCtx.Permissions["read"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: read permission required", http.StatusForbidden)
			return
		}

		var req struct {
			Text    string `json:"text"`
			Purpose string `json:"purpose"` // "query" or "document" (default: "document")
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}

		if req.Text == "" {
			http.Error(w, "text is required", http.StatusBadRequest)
			return
		}

		var vec []float32
		var err error
		if req.Purpose == "query" {
			vec, err = embedder.EmbedQuery(req.Text)
		} else {
			vec, err = embedder.Embed(req.Text)
		}
		if err != nil {
			logging.Default().LogError(r.Context(), "embed", err, "purpose", req.Purpose)
			http.Error(w, "embedding failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"embedding": vec,
			"dimension": len(vec),
		})
	})))

	// POST /api/embed/batch - Embed multiple texts using server-side embedder
	mux.HandleFunc("/api/embed/batch", withMetrics("embed_batch", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
		if !ok {
			http.Error(w, "internal error: missing tenant context", http.StatusInternalServerError)
			return
		}
		if !tenantCtx.Permissions["read"] && !tenantCtx.IsAdmin {
			http.Error(w, "forbidden: read permission required", http.StatusForbidden)
			return
		}

		var req struct {
			Texts   []string `json:"texts"`
			Purpose string   `json:"purpose"` // "query" or "document" (default: "document")
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}

		if len(req.Texts) == 0 {
			http.Error(w, "texts array is required", http.StatusBadRequest)
			return
		}

		embeddings := make([][]float32, len(req.Texts))
		for i, text := range req.Texts {
			var vec []float32
			var err error
			if req.Purpose == "query" {
				vec, err = embedder.EmbedQuery(text)
			} else {
				vec, err = embedder.Embed(text)
			}
			if err != nil {
				logging.Default().LogError(r.Context(), "embed_batch", err, "text_index", i, "purpose", req.Purpose)
				http.Error(w, fmt.Sprintf("embedding failed for text %d", i), http.StatusInternalServerError)
				return
			}
			embeddings[i] = vec
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"embeddings": embeddings,
			"count":      len(embeddings),
			"dimension":  len(embeddings[0]),
		})
	})))

	// ==========================================================================
	// INDEX MANAGEMENT API ENDPOINTS
	// ==========================================================================

	// GET /api/index/types - List available index types
	mux.HandleFunc("/api/index/types", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		types := index.SupportedTypes()
		sendResponse(w, r, map[string]any{
			"types": types,
			"descriptions": map[string]string{
				"hnsw":       "Hierarchical Navigable Small World - balanced speed/recall",
				"ivf":        "Inverted File Index - fast with cluster pruning",
				"ivf_binary": "IVF + Binary Quantization - 30x compression, 1000+ QPS",
				"binary":     "Binary Quantization - 32x compression, fast Hamming search",
				"flat":       "Flat/Brute Force - exact search, no index",
				"diskann":    "DiskANN - billion-scale disk-based index",
			},
		})
	})

	// POST /api/index/create - Create a new index for a collection
	mux.HandleFunc("/api/index/create", withMetrics("index_create", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Collection string                 `json:"collection"`
			IndexType  string                 `json:"index_type"` // hnsw, ivf, ivf_binary, binary, flat
			Config     map[string]interface{} `json:"config"`     // Index-specific config
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
			return
		}

		if req.Collection == "" {
			req.Collection = "default"
		}
		if req.IndexType == "" {
			req.IndexType = "hnsw"
		}

		// Create the index using the factory
		idx, err := index.Create(req.IndexType, store.Dim, req.Config)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to create index: %v", err), http.StatusBadRequest)
			return
		}

		// Store the index
		store.Lock()
		store.indexes[req.Collection] = idx
		store.Unlock()

		sendResponse(w, r, map[string]any{
			"ok":         true,
			"collection": req.Collection,
			"index_type": req.IndexType,
			"config":     req.Config,
			"stats":      idx.Stats(),
		})
	})))

	// GET /api/index/stats - Get index statistics for a collection
	mux.HandleFunc("/api/index/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		collection := r.URL.Query().Get("collection")
		if collection == "" {
			collection = "default"
		}

		store.RLock()
		idx, exists := store.indexes[collection]
		store.RUnlock()

		if !exists {
			http.Error(w, fmt.Sprintf("collection '%s' not found", collection), http.StatusNotFound)
			return
		}

		stats := idx.Stats()
		sendResponse(w, r, map[string]any{
			"collection": collection,
			"stats":      stats,
		})
	})

	// GET /api/index/list - List all indexes
	mux.HandleFunc("/api/index/list", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		store.RLock()
		indexes := make(map[string]interface{})
		for name, idx := range store.indexes {
			indexes[name] = map[string]interface{}{
				"type":  idx.Name(),
				"stats": idx.Stats(),
			}
		}
		store.RUnlock()

		sendResponse(w, r, map[string]any{
			"indexes": indexes,
			"count":   len(indexes),
		})
	})

	// Wrap with OTel HTTP middleware for request tracing
	// This adds automatic span creation for all HTTP requests
	otelMiddleware := telemetry.HTTPMiddleware()

	// Wrap with CORS middleware to allow browser requests from different origins.
	// Default: wildcard without credentials.
	// To allow credentialed requests, set CORS_ALLOWED_ORIGINS to a comma-separated allowlist.
	corsAllowedOriginsRaw := strings.TrimSpace(os.Getenv("CORS_ALLOWED_ORIGINS"))
	corsAllowAllOrigins := corsAllowedOriginsRaw == ""
	corsAllowedOrigins := make(map[string]struct{})
	if !corsAllowAllOrigins {
		for _, item := range strings.Split(corsAllowedOriginsRaw, ",") {
			origin := strings.TrimSpace(item)
			if origin == "" {
				continue
			}
			if origin == "*" {
				corsAllowAllOrigins = true
				corsAllowedOrigins = map[string]struct{}{}
				break
			}
			corsAllowedOrigins[origin] = struct{}{}
		}
	}
	corsMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := strings.TrimSpace(r.Header.Get("Origin"))
			if corsAllowAllOrigins {
				w.Header().Set("Access-Control-Allow-Origin", "*")
			} else if origin != "" {
				if _, ok := corsAllowedOrigins[origin]; ok {
					w.Header().Set("Access-Control-Allow-Origin", origin)
					w.Header().Set("Access-Control-Allow-Credentials", "true")
					w.Header().Add("Vary", "Origin")
				} else if r.Method == http.MethodOptions {
					http.Error(w, "forbidden origin", http.StatusForbidden)
					return
				}
			}

			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, X-Tenant-ID")
			w.Header().Set("Access-Control-Max-Age", "86400") // 24 hours

			// Handle preflight OPTIONS requests
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	}

	// Panic recovery middleware - prevents server crash from panics in handlers
	recoveryMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					stack := make([]byte, 4096)
					n := runtime.Stack(stack, false)
					logging.Default().Error("panic recovered in HTTP handler",
						"error", err,
						"path", r.URL.Path,
						"method", r.Method,
						"request_id", requestIDFromContext(r.Context()),
						"stack", string(stack[:n]),
					)
					// Return 500 error to client instead of closing connection — never expose panic details
					http.Error(w, "internal server error", http.StatusInternalServerError)
				}
			}()
			next.ServeHTTP(w, r)
		})
	}

	// Request context timeout middleware — cancels handler context after the deadline.
	// This is separate from HTTP server WriteTimeout (which is a hard TCP-level cutoff).
	// The context timeout lets handlers cooperatively abort long operations.
	// Streaming endpoints (snapshot, export) should check ctx.Done() and handle gracefully.
	requestTimeoutSec := envInt("HTTP_REQUEST_TIMEOUT_SEC", 120)
	requestTimeoutMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip timeout for streaming/long-running endpoints
			p := r.URL.Path
			if strings.HasPrefix(p, "/snapshot") || strings.HasPrefix(p, "/export") ||
				strings.HasPrefix(p, "/import") || p == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}
			ctx, cancel := context.WithTimeout(r.Context(), time.Duration(requestTimeoutSec)*time.Second)
			defer cancel()
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}

	// Request ID middleware — generates a unique ID for every request, stores it
	// in the context for structured logging, and echoes it in the X-Request-ID
	// response header so clients can correlate responses with server-side logs.
	// If the client or reverse proxy already provides X-Request-ID, it is reused.
	requestIDMiddleware := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id := r.Header.Get("X-Request-ID")
			if id == "" {
				id = generateRequestID()
			}
			// Cap client-supplied IDs to prevent log inflation and strip
			// non-printable characters to avoid log injection.
			if len(id) > 128 {
				id = truncateRequestID(id, 128)
			}
			w.Header().Set("X-Request-ID", id)
			ctx := context.WithValue(r.Context(), logging.RequestIDKey, id)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}

	var routed http.Handler = corsMiddleware(otelMiddleware(mux))
	if canonicalOnly {
		// Keep the allowlist outside CORS so OPTIONS cannot make an unsupported
		// legacy/provider route appear reachable.
		routed = canonicalRCSurface(routed)
	}
	return requestIDMiddleware(recoveryMiddleware(requestTimeoutMiddleware(routed))), collectionHTTP
}

func canonicalTenantIDFromPath(path string) string {
	const prefix = "/v3/tenants/"
	if !strings.HasPrefix(path, prefix) {
		return ""
	}
	tenantID := strings.TrimPrefix(path, prefix)
	if slash := strings.IndexByte(tenantID, '/'); slash >= 0 {
		tenantID = tenantID[:slash]
	}
	return tenantID
}

func canonicalRateLimitTenant(tenantCtx *security.TenantContext, targetTenant string) string {
	if tenantCtx == nil {
		return "default"
	}
	tenantKey := tenantCtx.TenantID
	if tenantCtx.IsServerAdmin && isValidTenantID(targetTenant) {
		tenantKey = targetTenant
	}
	if tenantKey == "" {
		return "default"
	}
	return tenantKey
}

func canonicalRCSurface(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if strings.HasPrefix(path, "/v3/tenants/") || path == "/v3/status" || path == "/healthz" || path == "/readyz" || path == "/livez" || path == "/metrics" {
			next.ServeHTTP(w, r)
			return
		}
		e := apierror.New(apierror.CodeNotFound, "no such route on the RC surface: "+path)
		e.Hint = "the RC serves /v3/tenants/{tenant}/collections..., /v3/status, /healthz, /readyz, /livez and /metrics; the route table is in the contract"
		apierror.WriteHTTP(w, e)
	})
}

func ageMillis(path string, fallback time.Time) int64 {
	if info, err := os.Stat(path); err == nil {
		return int64(time.Since(info.ModTime()).Milliseconds())
	}
	if !fallback.IsZero() {
		return int64(time.Since(fallback).Milliseconds())
	}
	return 0
}

type pageCursor struct {
	Offset     int    `json:"offset"`
	FilterHash string `json:"filter_hash"`
	LastSeq    uint64 `json:"last_seq"`
}

func encodePageToken(offset int, filterHash string, lastSeq uint64) string {
	cur := pageCursor{Offset: offset, FilterHash: filterHash, LastSeq: lastSeq}
	b, _ := json.Marshal(cur)
	return base64.StdEncoding.EncodeToString(b)
}

type queryResultItem struct {
	doc   string
	id    string
	meta  map[string]string
	score float32
	seq   uint64
}

func reorderQueryResultItems(items []queryResultItem, rerankedDocs []string) []queryResultItem {
	if len(items) == 0 || len(rerankedDocs) == 0 {
		return items[:0]
	}

	positionsByDoc := make(map[string][]int, len(items))
	for i, item := range items {
		positionsByDoc[item.doc] = append(positionsByDoc[item.doc], i)
	}

	reordered := make([]queryResultItem, 0, len(rerankedDocs))
	for _, doc := range rerankedDocs {
		positions := positionsByDoc[doc]
		if len(positions) == 0 {
			continue
		}
		pos := positions[0]
		positionsByDoc[doc] = positions[1:]
		reordered = append(reordered, items[pos])
	}

	return reordered
}

func decodePageToken(tok string) (pageCursor, error) {
	var cur pageCursor
	data, err := base64.StdEncoding.DecodeString(tok)
	if err != nil {
		return cur, err
	}
	if err := json.Unmarshal(data, &cur); err != nil {
		return cur, err
	}
	return cur, nil
}

func hashQueryCursor(query string, topK int, pageSize int, limit int, meta map[string]string, any []map[string]string, not map[string]string, ranges []RangeFilter, coll string, mode string, scoreMode string, efSearch int, hybridAlpha float64, tenantID string) string {
	type filterHash struct {
		Query     string              `json:"query"`
		TopK      int                 `json:"top_k"`
		PageSize  int                 `json:"page_size"`
		Limit     int                 `json:"limit"`
		Meta      map[string]string   `json:"meta"`
		Any       []map[string]string `json:"any"`
		Not       map[string]string   `json:"not"`
		Ranges    []RangeFilter       `json:"ranges"`
		Coll      string              `json:"coll"`
		Mode      string              `json:"mode"`
		ScoreMode string              `json:"score_mode"`
		EfSearch  int                 `json:"ef_search"`
		TenantID  string              `json:"tenant_id"`
	}
	type hybridHash struct {
		Alpha float64 `json:"alpha"`
	}
	payload := filterHash{
		Query:     query,
		TopK:      topK,
		PageSize:  pageSize,
		Limit:     limit,
		Meta:      meta,
		Any:       any,
		Not:       not,
		Ranges:    ranges,
		Coll:      coll,
		Mode:      mode,
		ScoreMode: scoreMode,
		EfSearch:  efSearch,
		TenantID:  tenantID,
	}
	b, _ := json.Marshal(struct {
		Filters filterHash `json:"filters"`
		Hybrid  hybridHash `json:"hybrid"`
	}{
		Filters: payload,
		Hybrid: hybridHash{
			Alpha: hybridAlpha,
		},
	})
	sum := fnv.New64a()
	_, _ = sum.Write(b)
	return fmt.Sprintf("%x", sum.Sum64())
}

// Compact purges tombstoned row storage while preserving the per-collection
// indexes (Delete already removes their entries), then saves a snapshot.
func (vs *VectorStore) Compact(path string) error {
	if err := func() error {
		vs.Lock()
		defer vs.Unlock()

		newIDToIx := make(map[uint64]int)
		newData := make([]float32, 0, len(vs.Data))
		newDocs := make([]string, 0, len(vs.Docs))
		newIDs := make([]string, 0, len(vs.IDs))
		newSeqs := make([]uint64, 0, len(vs.Seqs))
		newTenantID := make(map[uint64]string, len(vs.TenantID))
		for i, id := range vs.IDs {
			hid := hashID(id)
			if vs.Deleted[hid] {
				continue
			}
			if i >= len(vs.Seqs) {
				return fmt.Errorf("missing sequence for vector %q", id)
			}
			vec := vs.Data[i*vs.Dim : (i+1)*vs.Dim]
			base := len(newDocs)
			newData = append(newData, vec...)
			newDocs = append(newDocs, vs.Docs[i])
			newIDs = append(newIDs, id)
			newSeqs = append(newSeqs, vs.Seqs[i])
			newIDToIx[hid] = base
			newTenantID[hid] = vs.TenantID[hid]
		}
		vs.Data = newData
		vs.Docs = newDocs
		vs.IDs = newIDs
		vs.Seqs = newSeqs
		vs.Count = len(newDocs)
		vs.idToIx = newIDToIx
		vs.TenantID = newTenantID
		vs.Deleted = make(map[uint64]bool) // Clear tombstones
		return nil
	}(); err != nil {
		return err
	}

	// Save acquires its own read lock; calling it after the compaction critical
	// section avoids recursive RWMutex acquisition and permits normal writes to
	// proceed while the durable snapshot is encoded.
	return vs.Save(path)
}
