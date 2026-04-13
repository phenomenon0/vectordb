package main

import (
	"bytes"
	"context"
	"embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"io/fs"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Neumenon/cowrie/go/codec"
	"github.com/phenomenon0/vectordb/internal/index"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/obsidian"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/telemetry"

)

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

//go:embed web-ui/dist
var webUIFS embed.FS

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

// encodeResponse encodes v using the codec preferred by the client (Accept header).
// Cowrie is used when Accept: application/cowrie is present, JSON otherwise.
// For small responses, both formats are similar; for responses with []float32 arrays
// (like query scores), Cowrie provides ~48% size reduction.
func encodeResponse(w http.ResponseWriter, r *http.Request, v any) error {
	responseCodec := codec.FromRequest(r)
	w.Header().Set("Content-Type", responseCodec.ContentType())
	return responseCodec.Encode(w, v)
}

// sendResponse encodes and sends a response, logging any encoding errors.
// Use this instead of ignoring encodeResponse errors.
func sendResponse(w http.ResponseWriter, r *http.Request, v any) {
	if err := encodeResponse(w, r, v); err != nil {
		logging.Default().Error("failed to encode response", "error", err, "path", r.URL.Path)
	}
}

// decodeRequest decodes the request body using the appropriate codec based on Content-Type.
// Supports both JSON (default) and Cowrie (Content-Type: application/cowrie).
func decodeRequest(r *http.Request, v any) error {
	requestCodec := codec.FromContentType(r.Header.Get("Content-Type"))
	return requestCodec.Decode(r.Body, v)
}

// newHTTPHandler builds the HTTP mux for insert/query/delete/health/metrics.
func newHTTPHandler(store *VectorStore, embedder Embedder, reranker Reranker, indexPath string) (http.Handler, *CollectionHTTPServer) {
	mux := http.NewServeMux()
	configDir := "."
	if indexPath != "" {
		configDir = filepath.Dir(indexPath)
	}
	if store.rl == nil {
		rps := envInt("API_RPS", 100)
		store.rl = newRateLimiter(rps, rps, envInt("MAX_RATE_LIMIT_KEYS", 100_000), time.Minute)
	}
	trustProxy := os.Getenv("TRUST_PROXY") == "1"

	// SECURITY FIX: Proper JWT validation guard
	// When JWT_SECRET is configured, all requests MUST have valid JWT tokens
	guard := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			token := r.Header.Get("Authorization")
			if token == "" {
				token = r.URL.Query().Get("token")
			}

			authenticated := false
			var tenantCtx *security.TenantContext

			// JWT authentication (when enabled) - SECURE VERSION
			if store.jwtMgr != nil {
				if token == "" {
					// No token provided, but JWT is configured
					if store.requireAuth {
						http.Error(w, "unauthorized: missing authentication token", http.StatusUnauthorized)
						return
					}
					// If not required, use default context (backward compatibility)
					tenantCtx = &security.TenantContext{
						TenantID:    "default",
						Permissions: map[string]bool{"read": true, "write": true},
						Collections: make(map[string]bool),
						IsAdmin:     false,
					}
				} else {
					// Token provided - MUST be valid
					jwtToken := strings.TrimPrefix(token, "Bearer ")
					var err error
					tenantCtx, err = store.jwtMgr.ValidateTenantToken(jwtToken)
					if err != nil {
						http.Error(w, "unauthorized: invalid token: "+err.Error(), http.StatusUnauthorized)
						return
					}
					authenticated = true
				}
			} else {
				// No JWT manager configured - fallback to legacy API token auth
				// Simple API token authentication (legacy)
				if store.apiToken != "" {
					if token == "Bearer "+store.apiToken || token == store.apiToken {
						authenticated = true
					} else if token != "" {
						http.Error(w, "unauthorized", http.StatusUnauthorized)
						return
					}
				}

				if store.requireAuth && !authenticated {
					http.Error(w, "unauthorized", http.StatusUnauthorized)
					return
				}

				requestedTenantID := strings.TrimSpace(r.Header.Get("X-Tenant-ID"))
				if requestedTenantID != "" && !isValidTenantID(requestedTenantID) {
					http.Error(w, "invalid X-Tenant-ID header", http.StatusBadRequest)
					return
				}
				tenantID := "default"
				if requestedTenantID != "" {
					tenantID = requestedTenantID
				}

				// Use default tenant context for non-JWT mode
				if tenantCtx == nil {
					tenantCtx = &security.TenantContext{
						TenantID:    tenantID,
						Permissions: map[string]bool{"read": true, "write": true},
						Collections: make(map[string]bool),
						IsAdmin:     store.jwtMgr == nil && store.apiToken == "" && requestedTenantID == "",
					}
				}
			}

			// Global rate limiting (per-IP or per-token)
			if store.rl != nil {
				// Use IP address for anonymous users instead of shared "anon" key
				key := token
				if key == "" {
					// Extract client IP (handle X-Forwarded-For and X-Real-IP headers)
					clientIP := ""
					if trustProxy {
						clientIP = r.Header.Get("X-Forwarded-For")
						if clientIP == "" {
							clientIP = r.Header.Get("X-Real-IP")
						}
					}
					if clientIP == "" {
						clientIP = r.RemoteAddr
					}
					// Use first IP in X-Forwarded-For chain
					if idx := strings.Index(clientIP, ","); idx > 0 {
						clientIP = clientIP[:idx]
					}
					// Strip port from RemoteAddr
					if idx := strings.LastIndex(clientIP, ":"); idx > 0 {
						clientIP = clientIP[:idx]
					}
					key = "ip:" + strings.TrimSpace(clientIP)
				}
				if !store.rl.allow(key) {
					http.Error(w, "rate limited", http.StatusTooManyRequests)
					return
				}
			}

			// Add tenant context to request context for handlers to use
			if tenantCtx != nil {
				ctx := r.Context()
				ctx = context.WithValue(ctx, security.TenantContextKey, tenantCtx)
				r = r.WithContext(ctx)
			}

			next(w, r)
		}
	}

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
			http.Error(w, "embed error: "+err.Error(), http.StatusInternalServerError)
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
			http.Error(w, fmt.Sprintf("failed to insert document: %v", err), http.StatusInternalServerError)
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
				errors = append(errors, fmt.Sprintf("doc %d: embed error: %v", i, err))
				continue
			}

			var id string
			if req.Upsert {
				id, err = store.Upsert(vec, d.Doc, d.ID, d.Meta, collection, tenantID)
			} else {
				id, err = store.Add(vec, d.Doc, d.ID, d.Meta, collection, tenantID)
			}
			if err != nil {
				errors = append(errors, fmt.Sprintf("doc %d: insert error: %v", i, err))
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
			http.Error(w, fmt.Sprintf("failed to insert sparse vector: %v", err), http.StatusInternalServerError)
			return
		}

		if err := encodeResponse(w, r, map[string]any{"id": id}); err != nil {
			fmt.Printf("error encoding response: %v\n", err)
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
				http.Error(w, "embed error: "+err.Error(), http.StatusInternalServerError)
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
			http.Error(w, "rerank error: "+err.Error(), http.StatusInternalServerError)
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

		// Select codec based on Accept header (JSON default, Cowrie opt-in)
		responseCodec := codec.FromRequest(r)
		w.Header().Set("Content-Type", responseCodec.ContentType())

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

		if err := responseCodec.Encode(w, response); err != nil {
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
			fmt.Printf("error encoding response: %v\n", err)
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
			http.Error(w, fmt.Sprintf("failed to delete document: %v", err), http.StatusInternalServerError)
			return
		}

		// Removed synchronous save - rely on WAL + background snapshots
		telemetry.RecordOK(span)

		if err := encodeResponse(w, r, map[string]any{"deleted": req.ID}); err != nil {
			fmt.Printf("error encoding response: %v\n", err)
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
		walReplayErr := store.walReplayError
		_, embedderIsONNX := embedder.(*OnnxEmbedder)
		_, embedderIsOpenAI := embedder.(*OpenAIEmbedder)
		_, embedderIsTracked := embedder.(*TrackedEmbedder)
		_, embedderIsOllama := embedder.(*OllamaEmbedder)
		_, rerankerIsONNX := reranker.(*OnnxCrossEncoderReranker)
		store.RUnlock()
		snapAge := ageMillis(indexPath, lastSaved)
		walAge := ageMillis(store.walPath, time.Time{})

		// Determine embedder type
		embedderType := "hash"
		if embedderIsONNX {
			embedderType = "onnx"
		} else if embedderIsOpenAI || embedderIsTracked {
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
		healthy := walReplayErr == nil
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

		if walReplayErr != nil {
			response["wal_replay_error"] = walReplayErr.Error()
		}

		// Add mode information if available
		if CurrentMode != nil {
			response["mode"] = GetModeInfo(CurrentMode)
		}

		sendResponse(w, r, response)
	})))

	mux.Handle("/metrics", globalMetrics.Handler())

	// Kubernetes-style health probes
	// /healthz - Liveness probe: Is the process alive and not deadlocked?
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		// Liveness check: verify we can acquire locks (not deadlocked)
		done := make(chan bool, 1)
		go func() {
			store.RLock()
			store.RUnlock()
			done <- true
		}()

		select {
		case <-done:
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("ok"))
		case <-time.After(5 * time.Second):
			// Deadlock detected - process should be restarted
			w.WriteHeader(http.StatusServiceUnavailable)
			w.Write([]byte("deadlock detected"))
		}
	})

	// /readyz - Readiness probe: Is the service ready to accept traffic?
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		issues := []string{}

		// Check 1: Store is initialized
		store.RLock()
		storeReady := store.Count >= 0 && store.Dim > 0
		indexCount := len(store.indexes)
		walErr := store.walReplayError
		store.RUnlock()

		if !storeReady {
			issues = append(issues, "store not initialized")
		}
		if indexCount == 0 {
			issues = append(issues, "no index available")
		}
		if walErr != nil {
			issues = append(issues, "wal replay failed: "+walErr.Error())
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
				"version": "1.0.0",
			})
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ready":  false,
				"issues": issues,
			})
		}
	})

	// Serve Web UI Dashboard
	webUISubFS, err := fs.Sub(webUIFS, "web-ui/dist")
	if err != nil {
		fmt.Printf("Warning: failed to create web UI sub-filesystem: %v\n", err)
	} else {
		// Serve dashboard at /dashboard/
		mux.Handle("/dashboard/", http.StripPrefix("/dashboard/", http.FileServer(http.FS(webUISubFS))))

		// Serve /assets/ from dist (Vite builds with absolute /assets/ paths)
		mux.Handle("/assets/", http.FileServer(http.FS(webUISubFS)))

		// Redirect root to dashboard
		mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/" {
				http.Redirect(w, r, "/dashboard/", http.StatusFound)
				return
			}
			// Let other handlers take precedence
			http.NotFound(w, r)
		})
	}

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
			http.Error(w, fmt.Sprintf("compact failed: %v", err), http.StatusInternalServerError)
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

	// Import snapshot (overwrites current index) - Two-phase commit with validation
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

		// Add request size limit (max 1GB for snapshot)
		r.Body = http.MaxBytesReader(w, r.Body, limitSnapshotBody)

		// Phase 1: Validate imported snapshot
		tmp, err := os.CreateTemp("", "vectordb-import-*.gob")
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to create temp file: %v", err), http.StatusInternalServerError)
			return
		}
		defer os.Remove(tmp.Name())

		if _, err := io.Copy(tmp, r.Body); err != nil {
			http.Error(w, fmt.Sprintf("failed to read snapshot: %v", err), http.StatusInternalServerError)
			return
		}
		if err := tmp.Close(); err != nil {
			http.Error(w, fmt.Sprintf("failed to close temp file: %v", err), http.StatusInternalServerError)
			return
		}

		// Load and validate the new snapshot
		newStore, loaded := loadOrInitStore(tmp.Name(), store.Count+1, store.Dim)
		if !loaded {
			http.Error(w, "import failed: unable to load snapshot", http.StatusBadRequest)
			return
		}

		// Validate dimensions match
		if newStore.Dim != store.Dim {
			http.Error(w, fmt.Sprintf("dimension mismatch: current=%d, import=%d", store.Dim, newStore.Dim), http.StatusBadRequest)
			return
		}

		// Validate checksum
		if !newStore.validateChecksum() {
			http.Error(w, "checksum validation failed", http.StatusBadRequest)
			return
		}

		// Phase 2: Atomically replace store
		// Create backup before replacement
		backupPath := indexPath + ".backup"
		store.RLock()
		if err := store.Save(backupPath); err != nil {
			store.RUnlock()
			http.Error(w, fmt.Sprintf("failed to create backup: %v", err), http.StatusInternalServerError)
			return
		}
		store.RUnlock()

		// Replace store atomically - swap data fields individually to avoid copying mutex
		store.Lock()
		// Save old state for potential rollback (data fields only, not mutex)
		oldData := store.Data
		oldDim := store.Dim
		oldCount := store.Count
		oldDocs := store.Docs
		oldIDs := store.IDs
		oldSeqs := store.Seqs
		oldNext := store.next
		oldIndexes := store.indexes
		oldIdToIx := store.idToIx
		oldMeta := store.Meta
		oldDeleted := store.Deleted
		oldColl := store.Coll
		oldNumMeta := store.NumMeta
		oldTimeMeta := store.TimeMeta
		oldNumIndex := store.numIndex
		oldTimeIndex := store.timeIndex
		oldLexTF := store.lexTF
		oldDocLen := store.docLen
		oldDF := store.df
		oldSumDocL := store.sumDocL
		oldTenantID := store.TenantID
		oldMetaIndex := store.metaIndex

		// Copy data from newStore (preserving store's mutex and config fields)
		store.Data = newStore.Data
		store.Dim = newStore.Dim
		store.Count = newStore.Count
		store.Docs = newStore.Docs
		store.IDs = newStore.IDs
		store.Seqs = newStore.Seqs
		store.next = newStore.next
		store.indexes = newStore.indexes
		store.idToIx = newStore.idToIx
		store.Meta = newStore.Meta
		store.Deleted = newStore.Deleted
		store.Coll = newStore.Coll
		store.NumMeta = newStore.NumMeta
		store.TimeMeta = newStore.TimeMeta
		store.numIndex = newStore.numIndex
		store.timeIndex = newStore.timeIndex
		store.lexTF = newStore.lexTF
		store.docLen = newStore.docLen
		store.df = newStore.df
		store.sumDocL = newStore.sumDocL
		store.TenantID = newStore.TenantID
		store.metaIndex = newStore.metaIndex
		// Note: walPath, apiToken, rl, acl, quotas, etc. are preserved from old store
		store.Unlock()

		// Save new store
		if err := store.Save(indexPath); err != nil {
			// Rollback on failure - restore old data fields
			store.Lock()
			store.Data = oldData
			store.Dim = oldDim
			store.Count = oldCount
			store.Docs = oldDocs
			store.IDs = oldIDs
			store.Seqs = oldSeqs
			store.next = oldNext
			store.indexes = oldIndexes
			store.idToIx = oldIdToIx
			store.Meta = oldMeta
			store.Deleted = oldDeleted
			store.Coll = oldColl
			store.NumMeta = oldNumMeta
			store.TimeMeta = oldTimeMeta
			store.numIndex = oldNumIndex
			store.timeIndex = oldTimeIndex
			store.lexTF = oldLexTF
			store.docLen = oldDocLen
			store.df = oldDF
			store.sumDocL = oldSumDocL
			store.TenantID = oldTenantID
			store.metaIndex = oldMetaIndex
			store.Unlock()
			http.Error(w, fmt.Sprintf("import failed, rolled back: %v", err), http.StatusInternalServerError)
			os.Remove(backupPath)
			return
		}

		// Success - remove backup
		os.Remove(backupPath)

		if err := encodeResponse(w, r, map[string]any{
			"ok":      true,
			"count":   store.Count,
			"deleted": len(store.Deleted),
		}); err != nil {
			fmt.Printf("error encoding response: %v\n", err)
		}
	})))

	// ==================================================================================
	// ADMIN API ENDPOINTS - ACL & Quota Management
	// ==================================================================================

	// Admin middleware - requires admin permission
	adminGuard := func(next http.HandlerFunc) http.HandlerFunc {
		return guard(func(w http.ResponseWriter, r *http.Request) {
			// Read tenant context from request context (set by guard middleware)
			tenantCtx, ok := security.GetTenantContextFromContext(r.Context())
			if !ok || !tenantCtx.IsAdmin {
				http.Error(w, "forbidden: admin permission required", http.StatusForbidden)
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
	collectionHTTP := NewCollectionHTTPServer(indexPath + ".collections")
	if err := collectionHTTP.Load(indexPath + ".collections"); err != nil {
		fmt.Printf("Warning: failed to load collection state: %v\n", err)
	}
	collectionHTTP.RegisterHandlers(mux, guard, adminGuard)

	// ==========================================================================
	// FEEDBACK API ENDPOINTS (v2)
	// ==========================================================================
	// Enables relevance feedback collection and boost-based re-ranking
	// Endpoints: /v2/feedback, /v2/feedback/batch, /v2/feedback/stats,
	//            /v2/feedback/boosts, /v2/feedback/implicit, /v2/interaction
	RegisterFeedbackHandlers(mux)

	// ==========================================================================
	// KNOWLEDGE GRAPH EXTRACTION API ENDPOINTS (v2)
	// ==========================================================================
	// LLM-based entity/relationship extraction from text
	// Endpoints: /v2/extract, /v2/extract/batch, /v2/extract/temporal, /v2/extract/status
	RegisterExtractionHandlers(mux)

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
			Type      string `json:"type"`      // "ollama", "openai", "gemini", "voyage", "jina", "cohere", "mistral", "hash"
			Model     string `json:"model"`     // model name
			URL       string `json:"url"`       // for ollama: base URL
			Key       string `json:"key"`       // API key for the provider
			Dimension int    `json:"dimension"` // for providers with configurable dimensions (gemini, voyage, jina)
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

		case "gemini":
			key := req.Key
			if key == "" {
				key = os.Getenv("GOOGLE_API_KEY")
			}
			if key == "" {
				key = os.Getenv("GEMINI_API_KEY")
			}
			if key == "" {
				errMsg = "Google/Gemini API key required (pass 'key' field or set GOOGLE_API_KEY)"
				break
			}
			dim := req.Dimension
			if dim <= 0 {
				dim = 3072
			}
			emb := NewGeminiEmbedder(key, dim)
			vec, err := emb.Embed("dimension test")
			if err != nil {
				errMsg = "gemini embed failed: " + err.Error()
				break
			}
			newDim = len(vec)
			newEmb = emb
			newEmbType = "gemini"
			newEmbModel = "gemini-embedding-2-preview"

		case "voyage":
			key := req.Key
			if key == "" {
				key = os.Getenv("VOYAGE_API_KEY")
			}
			if key == "" {
				errMsg = "Voyage API key required (pass 'key' field or set VOYAGE_API_KEY)"
				break
			}
			model := req.Model
			if model == "" {
				model = "voyage-4-large"
			}
			dim := req.Dimension
			if dim <= 0 {
				dim = 1024
			}
			emb := NewVoyageEmbedder(key, model, dim)
			vec, err := emb.Embed("dimension test")
			if err != nil {
				errMsg = "voyage embed failed: " + err.Error()
				break
			}
			newDim = len(vec)
			newEmb = emb
			newEmbType = "voyage"
			newEmbModel = model

		case "jina":
			key := req.Key
			if key == "" {
				key = os.Getenv("JINA_API_KEY")
			}
			if key == "" {
				errMsg = "Jina API key required (pass 'key' field or set JINA_API_KEY)"
				break
			}
			model := req.Model
			if model == "" {
				model = "jina-embeddings-v3"
			}
			dim := req.Dimension
			if dim <= 0 {
				dim = 1024
			}
			emb := NewJinaEmbedder(key, model, dim)
			vec, err := emb.Embed("dimension test")
			if err != nil {
				errMsg = "jina embed failed: " + err.Error()
				break
			}
			newDim = len(vec)
			newEmb = emb
			newEmbType = "jina"
			newEmbModel = model

		case "cohere":
			key := req.Key
			if key == "" {
				key = os.Getenv("COHERE_API_KEY")
			}
			if key == "" {
				errMsg = "Cohere API key required (pass 'key' field or set COHERE_API_KEY)"
				break
			}
			model := req.Model
			if model == "" {
				model = "embed-english-v3.0"
			}
			emb := NewCohereEmbedder(key, model)
			vec, err := emb.Embed("dimension test")
			if err != nil {
				errMsg = "cohere embed failed: " + err.Error()
				break
			}
			newDim = len(vec)
			newEmb = emb
			newEmbType = "cohere"
			newEmbModel = model

		case "mistral":
			key := req.Key
			if key == "" {
				key = os.Getenv("MISTRAL_API_KEY")
			}
			if key == "" {
				errMsg = "Mistral API key required (pass 'key' field or set MISTRAL_API_KEY)"
				break
			}
			model := req.Model
			if model == "" {
				model = "mistral-embed"
			}
			emb := NewMistralEmbedder(key, model)
			vec, err := emb.Embed("dimension test")
			if err != nil {
				errMsg = "mistral embed failed: " + err.Error()
				break
			}
			newDim = len(vec)
			newEmb = emb
			newEmbType = "mistral"
			newEmbModel = model

		default:
			errMsg = "unknown embedder type: " + req.Type + " (valid: ollama, openai, gemini, voyage, jina, cohere, mistral, hash)"
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

	// GET /api/costs - Returns cost tracking statistics (PRO mode only)
	mux.HandleFunc("/api/costs", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Find the cost tracker from the embedder if it's a TrackedEmbedder
		var costTracker *CostTracker
		if te, ok := embedder.(*TrackedEmbedder); ok {
			costTracker = te.costTracker
		}

		if costTracker == nil {
			// LOCAL mode - no cost tracking
			sendResponse(w, r, map[string]any{
				"mode":    "local",
				"message": "Cost tracking is only available in PRO mode",
				"session": map[string]any{
					"tokens": 0,
					"cost":   0,
					"ops":    0,
				},
			})
			return
		}

		stats := costTracker.GetStats()
		sendResponse(w, r, stats)
	})

	// GET /api/costs/daily - Returns daily cost breakdown (PRO mode only)
	mux.HandleFunc("/api/costs/daily", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var costTracker *CostTracker
		if te, ok := embedder.(*TrackedEmbedder); ok {
			costTracker = te.costTracker
		}

		if costTracker == nil {
			sendResponse(w, r, map[string]any{
				"mode":  "local",
				"daily": []DailyStats{},
			})
			return
		}

		days := 30 // Default to last 30 days
		dailyStats, err := costTracker.GetDailyStats(days)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to get daily stats: %v", err), http.StatusInternalServerError)
			return
		}

		sendResponse(w, r, map[string]any{
			"mode":  "pro",
			"days":  days,
			"daily": dailyStats,
		})
	})

	// GET /api/costs/export - Export costs as CSV (PRO mode only)
	mux.HandleFunc("/api/costs/export", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		var costTracker *CostTracker
		if te, ok := embedder.(*TrackedEmbedder); ok {
			costTracker = te.costTracker
		}

		if costTracker == nil {
			http.Error(w, "Cost tracking is only available in PRO mode", http.StatusBadRequest)
			return
		}

		csv, err := costTracker.ExportCSV()
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to export costs: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/csv")
		w.Header().Set("Content-Disposition", "attachment; filename=vectordb-costs.csv")
		w.Write([]byte(csv))
	})

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
			http.Error(w, fmt.Sprintf("embedding failed: %v", err), http.StatusInternalServerError)
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
				http.Error(w, fmt.Sprintf("embedding failed for text %d: %v", i, err), http.StatusInternalServerError)
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
					// Log the panic with stack trace
					logging.Default().Error("panic recovered in HTTP handler",
						"error", err,
						"path", r.URL.Path,
						"method", r.Method,
					)
					// Return 500 error to client instead of closing connection
					http.Error(w, fmt.Sprintf("internal server error: %v", err), http.StatusInternalServerError)
				}
			}()
			next.ServeHTTP(w, r)
		})
	}

	// ==========================================================================
	// OBSIDIAN ADMIN ENDPOINTS
	// ==========================================================================

	mux.HandleFunc("/admin/obsidian/detect", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		vaults := obsidian.DetectVaults()
		sendResponse(w, r, map[string]any{"vaults": vaults})
	}))

	mux.HandleFunc("/admin/obsidian/status", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		cfg := obsidian.LoadOrDetectConfig(configDir)
		obsidian.ApplyEnvOverrides(&cfg)

		// Count notes in the obsidian collection
		collection := cfg.Collection
		if collection == "" {
			collection = "obsidian"
		}
		store.RLock()
		noteCount := 0
		for _, coll := range store.Coll {
			if coll == collection {
				noteCount++
			}
		}
		store.RUnlock()

		sendResponse(w, r, map[string]any{
			"enabled":    cfg.Enabled,
			"vault":      cfg.VaultPath,
			"collection": collection,
			"interval":   cfg.Interval.String(),
			"note_count": noteCount,
		})
	}))

	mux.HandleFunc("/admin/obsidian/enable", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Vault      string `json:"vault"`
			Collection string `json:"collection"`
			Interval   string `json:"interval"`
		}
		if err := decodeRequest(r, &req); err != nil {
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if req.Vault == "" {
			http.Error(w, "vault path required", http.StatusBadRequest)
			return
		}
		// Validate vault path exists
		if info, err := os.Stat(req.Vault); err != nil || !info.IsDir() {
			http.Error(w, "vault path is not a valid directory", http.StatusBadRequest)
			return
		}

		cfg := obsidian.DefaultConfig()
		cfg.VaultPath = req.Vault
		cfg.Enabled = true
		if req.Collection != "" {
			cfg.Collection = req.Collection
		}
		if req.Interval != "" {
			if d, err := time.ParseDuration(req.Interval); err == nil {
				cfg.Interval = d
			}
		}

		if err := obsidian.SaveConfig(configDir, cfg); err != nil {
			http.Error(w, "failed to save config: "+err.Error(), http.StatusInternalServerError)
			return
		}

		sendResponse(w, r, map[string]any{
			"ok":         true,
			"vault":      cfg.VaultPath,
			"collection": cfg.Collection,
			"interval":   cfg.Interval.String(),
			"note":       "restart server to start sync, or set OBSIDIAN_VAULT env var",
		})
	}))

	mux.HandleFunc("/admin/obsidian/disable", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		cfg := obsidian.DefaultConfig()
		cfg.Enabled = false
		cfg.VaultPath = ""
		if err := obsidian.SaveConfig(configDir, cfg); err != nil {
			http.Error(w, "failed to save config: "+err.Error(), http.StatusInternalServerError)
			return
		}
		sendResponse(w, r, map[string]any{"ok": true, "note": "restart server to stop sync"})
	}))

	// ==========================================================================
	// VAULT FILE SERVING & ANNOTATIONS
	// ==========================================================================

	// getVaultRoot returns the configured obsidian vault path, or empty string.
	getVaultRoot := func() string {
		cfg := obsidian.LoadOrDetectConfig(configDir)
		obsidian.ApplyEnvOverrides(&cfg)
		return cfg.VaultPath
	}

	// validateVaultPath resolves a relative path within the vault, rejecting traversal.
	validateVaultPath := func(vaultRoot, relPath string) (string, error) {
		cleaned := filepath.Clean(relPath)
		if strings.Contains(cleaned, "..") {
			return "", fmt.Errorf("path traversal rejected")
		}
		abs := filepath.Join(vaultRoot, cleaned)
		// Ensure resolved path is still within vault
		realAbs, err := filepath.EvalSymlinks(abs)
		if err != nil {
			// File may not exist yet for new paths — check parent
			realAbs = abs
		}
		realVault, err := filepath.EvalSymlinks(vaultRoot)
		if err != nil {
			realVault = vaultRoot
		}
		if !strings.HasPrefix(realAbs, realVault+string(filepath.Separator)) && realAbs != realVault {
			return "", fmt.Errorf("path outside vault")
		}
		return abs, nil
	}

	// GET /vault/file?path=<relative_path> — serve a file from the vault
	mux.HandleFunc("/vault/file", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		vaultRoot := getVaultRoot()
		if vaultRoot == "" {
			http.Error(w, "no vault configured", http.StatusNotFound)
			return
		}
		relPath := r.URL.Query().Get("path")
		if relPath == "" {
			http.Error(w, "path parameter required", http.StatusBadRequest)
			return
		}
		absPath, err := validateVaultPath(vaultRoot, relPath)
		if err != nil {
			http.Error(w, "forbidden: "+err.Error(), http.StatusForbidden)
			return
		}
		info, err := os.Stat(absPath)
		if err != nil {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if info.IsDir() {
			http.Error(w, "path is a directory", http.StatusBadRequest)
			return
		}
		if info.Size() > 100*1024*1024 {
			http.Error(w, "file too large (>100MB)", http.StatusRequestEntityTooLarge)
			return
		}
		// Set MIME type from extension
		ext := filepath.Ext(absPath)
		mimeType := mime.TypeByExtension(ext)
		if mimeType != "" {
			w.Header().Set("Content-Type", mimeType)
		}
		http.ServeFile(w, r, absPath)
	}))

	// GET /vault/browse?dir=<relative_path> — list files in a vault subdirectory
	mux.HandleFunc("/vault/browse", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		vaultRoot := getVaultRoot()
		if vaultRoot == "" {
			http.Error(w, "no vault configured", http.StatusNotFound)
			return
		}
		relDir := r.URL.Query().Get("dir")
		if relDir == "" {
			relDir = "."
		}
		absDir, err := validateVaultPath(vaultRoot, relDir)
		if err != nil {
			http.Error(w, "forbidden: "+err.Error(), http.StatusForbidden)
			return
		}
		entries, err := os.ReadDir(absDir)
		if err != nil {
			http.Error(w, "cannot read directory", http.StatusNotFound)
			return
		}
		type fileEntry struct {
			Name    string `json:"name"`
			Path    string `json:"path"`
			Size    int64  `json:"size"`
			ModTime string `json:"mod_time"`
			IsDir   bool   `json:"is_dir"`
		}
		files := make([]fileEntry, 0, len(entries))
		for _, e := range entries {
			// Skip hidden files/dirs
			if strings.HasPrefix(e.Name(), ".") {
				continue
			}
			info, err := e.Info()
			if err != nil {
				continue
			}
			entryPath := relDir
			if entryPath == "." {
				entryPath = e.Name()
			} else {
				entryPath = filepath.Join(relDir, e.Name())
			}
			files = append(files, fileEntry{
				Name:    e.Name(),
				Path:    entryPath,
				Size:    info.Size(),
				ModTime: info.ModTime().UTC().Format(time.RFC3339),
				IsDir:   e.IsDir(),
			})
		}
		sendResponse(w, r, map[string]any{"files": files, "dir": relDir})
	}))

	// Annotation storage — simple JSON file persistence
	annotationFile := filepath.Join(configDir, "annotations.json")
	var annotationMu sync.Mutex

	loadAnnotations := func() map[string][]map[string]any {
		data, err := os.ReadFile(annotationFile)
		if err != nil {
			return make(map[string][]map[string]any)
		}
		var result map[string][]map[string]any
		if err := json.Unmarshal(data, &result); err != nil {
			return make(map[string][]map[string]any)
		}
		return result
	}
	saveAnnotations := func(anns map[string][]map[string]any) error {
		data, err := json.MarshalIndent(anns, "", "  ")
		if err != nil {
			return err
		}
		return os.WriteFile(annotationFile, data, 0644)
	}

	// GET /vault/annotations?doc_id=<id> — get annotations for a document
	// POST /vault/annotations — create/update an annotation
	// DELETE /vault/annotations?doc_id=<id>&id=<aid> — delete an annotation
	mux.HandleFunc("/vault/annotations", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			docID := r.URL.Query().Get("doc_id")
			annotationMu.Lock()
			anns := loadAnnotations()
			annotationMu.Unlock()
			if docID != "" {
				sendResponse(w, r, map[string]any{"annotations": anns[docID]})
			} else {
				sendResponse(w, r, map[string]any{"annotations": anns})
			}

		case http.MethodPost:
			var ann map[string]any
			if err := decodeRequest(r, &ann); err != nil {
				http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
				return
			}
			docID, _ := ann["doc_id"].(string)
			if docID == "" {
				http.Error(w, "doc_id required", http.StatusBadRequest)
				return
			}
			annotationMu.Lock()
			anns := loadAnnotations()
			anns[docID] = append(anns[docID], ann)
			err := saveAnnotations(anns)
			annotationMu.Unlock()
			if err != nil {
				http.Error(w, "save failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			sendResponse(w, r, map[string]any{"ok": true})

		case http.MethodDelete:
			docID := r.URL.Query().Get("doc_id")
			annID := r.URL.Query().Get("id")
			if docID == "" || annID == "" {
				http.Error(w, "doc_id and id required", http.StatusBadRequest)
				return
			}
			annotationMu.Lock()
			anns := loadAnnotations()
			if docAnns, ok := anns[docID]; ok {
				filtered := make([]map[string]any, 0, len(docAnns))
				for _, a := range docAnns {
					if aid, _ := a["id"].(string); aid != annID {
						filtered = append(filtered, a)
					}
				}
				anns[docID] = filtered
			}
			err := saveAnnotations(anns)
			annotationMu.Unlock()
			if err != nil {
				http.Error(w, "save failed: "+err.Error(), http.StatusInternalServerError)
				return
			}
			sendResponse(w, r, map[string]any{"ok": true})

		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))

	// ==========================================================================
	// WHISPER TRANSCRIPTION ENDPOINT
	// ==========================================================================

	// POST /vault/transcribe?path=<relative_path> — transcribe audio/video via OpenAI Whisper
	// Also accepts file upload via multipart form (field "file")
	mux.HandleFunc("/vault/transcribe", adminGuard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		openaiKey := os.Getenv("OPENAI_API_KEY")
		if openaiKey == "" {
			http.Error(w, "OPENAI_API_KEY not configured", http.StatusServiceUnavailable)
			return
		}

		var audioData io.Reader
		var filename string

		// Option 1: path query param — read from vault
		if relPath := r.URL.Query().Get("path"); relPath != "" {
			vaultRoot := getVaultRoot()
			if vaultRoot == "" {
				http.Error(w, "no vault configured", http.StatusNotFound)
				return
			}
			absPath, err := validateVaultPath(vaultRoot, relPath)
			if err != nil {
				http.Error(w, "forbidden: "+err.Error(), http.StatusForbidden)
				return
			}
			info, err := os.Stat(absPath)
			if err != nil {
				http.Error(w, "not found", http.StatusNotFound)
				return
			}
			if info.Size() > 25*1024*1024 { // Whisper limit is 25MB
				http.Error(w, "file too large for Whisper (>25MB)", http.StatusRequestEntityTooLarge)
				return
			}
			f, err := os.Open(absPath)
			if err != nil {
				http.Error(w, "cannot open file", http.StatusInternalServerError)
				return
			}
			defer f.Close()
			audioData = f
			filename = filepath.Base(absPath)
		} else {
			// Option 2: multipart upload
			if err := r.ParseMultipartForm(25 << 20); err != nil {
				http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
				return
			}
			file, header, err := r.FormFile("file")
			if err != nil {
				http.Error(w, "file field required", http.StatusBadRequest)
				return
			}
			defer file.Close()
			audioData = file
			filename = header.Filename
		}

		// Build multipart request for OpenAI Whisper API
		var buf bytes.Buffer
		mw := multipart.NewWriter(&buf)
		part, err := mw.CreateFormFile("file", filename)
		if err != nil {
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		if _, err := io.Copy(part, audioData); err != nil {
			http.Error(w, "read error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		mw.WriteField("model", "whisper-1")
		mw.WriteField("response_format", "verbose_json")

		// Optional language hint
		if lang := r.URL.Query().Get("language"); lang != "" {
			mw.WriteField("language", lang)
		}
		mw.Close()

		whisperReq, err := http.NewRequest("POST", "https://api.openai.com/v1/audio/transcriptions", &buf)
		if err != nil {
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		whisperReq.Header.Set("Authorization", "Bearer "+openaiKey)
		whisperReq.Header.Set("Content-Type", mw.FormDataContentType())

		client := &http.Client{Timeout: 120 * time.Second}
		resp, err := client.Do(whisperReq)
		if err != nil {
			http.Error(w, "whisper API error: "+err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			http.Error(w, "whisper API error "+fmt.Sprint(resp.StatusCode)+": "+string(body), resp.StatusCode)
			return
		}

		// Stream the JSON response back
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
	}))

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

	return recoveryMiddleware(requestTimeoutMiddleware(corsMiddleware(otelMiddleware(mux)))), collectionHTTP
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

// Compact rebuilds the index and purges tombstones, then saves a snapshot.
func (vs *VectorStore) Compact(path string) error {
	vs.Lock()
	defer vs.Unlock()

	cfg := loadHNSWConfig()
	newIdx, err := index.NewHNSWIndex(vs.Dim, map[string]interface{}{
		"m":         cfg.M,
		"ml":        cfg.Ml,
		"ef_search": cfg.EfSearch,
	})
	if err != nil {
		return fmt.Errorf("failed to create new index: %w", err)
	}

	vs.idToIx = make(map[uint64]int)
	newData := make([]float32, 0, len(vs.Data))
	newDocs := make([]string, 0, len(vs.Docs))
	newIDs := make([]string, 0, len(vs.IDs))
	for i, id := range vs.IDs {
		hid := hashID(id)
		if vs.Deleted[hid] {
			continue
		}
		vec := vs.Data[i*vs.Dim : (i+1)*vs.Dim]
		base := len(newDocs)
		newData = append(newData, vec...)
		newDocs = append(newDocs, vs.Docs[i])
		newIDs = append(newIDs, id)
		if err := newIdx.Add(context.Background(), hid, vec); err != nil {
			return fmt.Errorf("failed to add vector to new index: %w", err)
		}
		vs.idToIx[hid] = base
	}
	vs.Data = newData
	vs.Docs = newDocs
	vs.IDs = newIDs
	vs.Count = len(newDocs)
	vs.indexes["default"] = newIdx
	vs.Deleted = make(map[uint64]bool) // Clear tombstones
	if err := vs.Save(path); err != nil {
		return err
	}
	return nil
}
