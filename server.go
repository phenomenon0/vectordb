package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"agentscope/sjson/codec"

	"github.com/coder/hnsw"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// encodeResponse encodes v using the codec preferred by the client (Accept header).
// SJSON is used when Accept: application/sjson is present, JSON otherwise.
// For small responses, both formats are similar; for responses with []float32 arrays
// (like query scores), SJSON provides ~48% size reduction.
func encodeResponse(w http.ResponseWriter, r *http.Request, v any) error {
	responseCodec := codec.FromRequest(r)
	w.Header().Set("Content-Type", responseCodec.ContentType())
	return responseCodec.Encode(w, v)
}

// decodeRequest decodes the request body using the appropriate codec based on Content-Type.
// Supports both JSON (default) and SJSON (Content-Type: application/sjson).
func decodeRequest(r *http.Request, v any) error {
	requestCodec := codec.FromContentType(r.Header.Get("Content-Type"))
	return requestCodec.Decode(r.Body, v)
}

// newHTTPHandler builds the HTTP mux for insert/query/delete/health/metrics.
func newHTTPHandler(store *VectorStore, embedder Embedder, reranker Reranker, indexPath string) http.Handler {
	mux := http.NewServeMux()
	if store.rl == nil {
		rps := envInt("API_RPS", 100)
		store.rl = newRateLimiter(rps, rps, time.Minute)
	}
	guard := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			token := r.Header.Get("Authorization")
			if token == "" {
				token = r.URL.Query().Get("token")
			}

			// JWT authentication (when enabled)
			if store.requireAuth && store.jwtMgr != nil {
				if token == "" {
					http.Error(w, "unauthorized: missing token", http.StatusUnauthorized)
					return
				}
				// Extract Bearer token
				jwtToken := strings.TrimPrefix(token, "Bearer ")
				if _, err := store.jwtMgr.ValidateTenantToken(jwtToken); err != nil {
					http.Error(w, "unauthorized: "+err.Error(), http.StatusUnauthorized)
					return
				}
			}

			// Simple API token authentication (legacy)
			if store.apiToken != "" {
				if token != "Bearer "+store.apiToken && token != store.apiToken {
					http.Error(w, "unauthorized", http.StatusUnauthorized)
					return
				}
			}
			if store.rl != nil {
				// Use IP address for anonymous users instead of shared "anon" key
				key := token
				if key == "" {
					// Extract client IP (handle X-Forwarded-For and X-Real-IP headers)
					clientIP := r.Header.Get("X-Forwarded-For")
					if clientIP == "" {
						clientIP = r.Header.Get("X-Real-IP")
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
			next(w, r)
		}
	}

	mux.HandleFunc("/insert", withMetrics("insert", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context
		tenantCtx := GetTenantContextFromRequest(r, store.jwtMgr)
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
		r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024) // 10MB limit

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
		const (
			MaxDocLength       = 1_000_000 // 1MB
			MaxMetaKeys        = 100
			MaxMetaValueLength = 10_000
		)

		if req.Doc == "" {
			http.Error(w, "doc required", http.StatusBadRequest)
			return
		}
		if len(req.Doc) > MaxDocLength {
			http.Error(w, fmt.Sprintf("doc too large: max %d bytes", MaxDocLength), http.StatusBadRequest)
			return
		}
		if len(req.Meta) > MaxMetaKeys {
			http.Error(w, fmt.Sprintf("too many metadata keys: max %d", MaxMetaKeys), http.StatusBadRequest)
			return
		}
		for k, v := range req.Meta {
			if len(k) > MaxMetaValueLength || len(v) > MaxMetaValueLength {
				http.Error(w, fmt.Sprintf("metadata key or value too large: max %d bytes", MaxMetaValueLength), http.StatusBadRequest)
				return
			}
		}

		// Check collection access via ACL
		if req.Collection == "" {
			req.Collection = "default"
		}
		if !tenantCtx.IsAdmin && len(tenantCtx.Collections) > 0 && !tenantCtx.Collections[req.Collection] {
			http.Error(w, fmt.Sprintf("forbidden: no access to collection '%s'", req.Collection), http.StatusForbidden)
			return
		}

		vec, err := embedder.Embed(req.Doc)
		if err != nil {
			http.Error(w, "embed error: "+err.Error(), http.StatusInternalServerError)
			return
		}

		var id string
		if req.Upsert {
			id, err = store.Upsert(vec, req.Doc, req.ID, req.Meta, req.Collection, tenantID)
		} else {
			id, err = store.Add(vec, req.Doc, req.ID, req.Meta, req.Collection, tenantID)
		}
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to insert document: %v", err), http.StatusInternalServerError)
			return
		}

		if err := encodeResponse(w, r, map[string]any{"id": id}); err != nil {
			fmt.Printf("error encoding response: %v\n", err)
		}
	})))

	mux.HandleFunc("/batch_insert", withMetrics("batch_insert", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context
		tenantCtx := GetTenantContextFromRequest(r, store.jwtMgr)
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
		r.Body = http.MaxBytesReader(w, r.Body, 50*1024*1024) // 50MB limit for batch

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

		const (
			MaxDocLength       = 1_000_000 // 1MB
			MaxMetaKeys        = 100
			MaxMetaValueLength = 10_000
			MaxBatchSize       = 10_000
		)

		if len(req.Docs) == 0 {
			http.Error(w, "no docs provided", http.StatusBadRequest)
			return
		}
		if len(req.Docs) > MaxBatchSize {
			http.Error(w, fmt.Sprintf("batch too large: max %d docs", MaxBatchSize), http.StatusBadRequest)
			return
		}

		ids := make([]string, 0, len(req.Docs))
		var errors []string

		for i, d := range req.Docs {
			if d.Doc == "" {
				errors = append(errors, fmt.Sprintf("doc %d: empty document", i))
				continue
			}
			if len(d.Doc) > MaxDocLength {
				errors = append(errors, fmt.Sprintf("doc %d: too large", i))
				continue
			}
			if len(d.Meta) > MaxMetaKeys {
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
		}

		if err := encodeResponse(w, r, response); err != nil {
			fmt.Printf("error encoding response: %v\n", err)
		}
	})))

	mux.HandleFunc("/query", withMetrics("query", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context
		tenantCtx := GetTenantContextFromRequest(r, store.jwtMgr)
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
		r.Body = http.MaxBytesReader(w, r.Body, 1*1024*1024) // 1MB limit for query

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
		if req.Mode == "" {
			req.Mode = "ann"
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
		offset := req.Offset
		if req.PageToken != "" {
			if v, err := decodePageToken(req.PageToken); err == nil {
				offset = v.Offset
				if v.FilterHash != hashFilters(req.Meta, req.MetaAny, req.MetaNot, req.MetaRanges, req.Collection, req.Mode, req.ScoreMode) {
					http.Error(w, "page token invalid for current query", http.StatusBadRequest)
					return
				}
			}
		}

		qTokens := tokenize(req.Query)
		qVec := []float32{}
		var err error
		if req.Mode != "lex" {
			qVec, err = embedder.Embed(req.Query)
			if err != nil {
				http.Error(w, "embed error: "+err.Error(), http.StatusInternalServerError)
				return
			}
		}
		var ids []int
		if req.Mode == "scan" {
			ids = store.Search(qVec, req.TopK)
		} else if req.Mode == "lex" {
			ids = store.SearchLex(qTokens, req.TopK)
		} else {
			if req.EfSearch > 0 {
				orig := store.hnsw.EfSearch
				store.hnsw.EfSearch = req.EfSearch
				ids = store.SearchANN(qVec, req.TopK)
				store.hnsw.EfSearch = orig
			} else {
				ids = store.SearchANN(qVec, req.TopK)
			}
		}
		docs := make([]string, 0, len(ids))
		respIDs := make([]string, 0, len(ids))
		respMeta := make([]map[string]string, 0, len(ids))
		respScores := make([]float32, 0, len(ids))
		type scored struct {
			docIdx int
			score  float64
			seq    uint64
		}
		scoresOrdered := make([]scored, 0, len(ids))
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
			docs = append(docs, store.GetDoc(idx))
			respIDs = append(respIDs, store.GetID(idx))
			if req.IncludeMeta {
				cp := make(map[string]string, len(meta))
				for k, v := range meta {
					cp[k] = v
				}
				respMeta = append(respMeta, cp)
			}
			switch req.ScoreMode {
			case "hybrid":
				respScores = append(respScores, float32(store.hybridScore(hid, qVec, qTokens, req.HybridAlpha)))
			case "lexical":
				respScores = append(respScores, float32(store.bm25(hid, qTokens)))
			default: // vector
				respScores = append(respScores, DotProduct(qVec, store.Data[idx*store.Dim:(idx+1)*store.Dim]))
			}
			scoresOrdered = append(scoresOrdered, scored{docIdx: len(docs) - 1, score: float64(respScores[len(respScores)-1]), seq: store.Seqs[idx]})
		}
		if len(scoresOrdered) > 0 {
			sort.Slice(scoresOrdered, func(i, j int) bool {
				if scoresOrdered[i].score == scoresOrdered[j].score {
					return scoresOrdered[i].seq < scoresOrdered[j].seq
				}
				return scoresOrdered[i].score > scoresOrdered[j].score
			})
			reDocs := make([]string, 0, len(scoresOrdered))
			reIDs := make([]string, 0, len(scoresOrdered))
			reScores := make([]float32, 0, len(scoresOrdered))
			var reMeta []map[string]string
			if req.IncludeMeta {
				reMeta = make([]map[string]string, 0, len(scoresOrdered))
			}
			for _, s := range scoresOrdered {
				reDocs = append(reDocs, docs[s.docIdx])
				reIDs = append(reIDs, respIDs[s.docIdx])
				reScores = append(reScores, float32(s.score))
				if req.IncludeMeta {
					reMeta = append(reMeta, respMeta[s.docIdx])
				}
			}
			docs = reDocs
			respIDs = reIDs
			respScores = reScores
			if req.IncludeMeta {
				respMeta = reMeta
			}
		}
		start := req.Offset
		if start > len(docs) {
			start = len(docs)
		}
		end := start + pageSize
		if end > len(docs) {
			end = len(docs)
		}
		docs = docs[start:end]
		respIDs = respIDs[start:end]
		if len(respScores) > 0 {
			respScores = respScores[start:end]
		} else {
			respScores = nil
		}
		if req.IncludeMeta && len(respMeta) > 0 {
			respMeta = respMeta[start:end]
		} else {
			respMeta = nil
		}

		nextPage := ""
		if end < len(respIDs) {
			nextPage = encodePageToken(offset+pageSize, hashFilters(req.Meta, req.MetaAny, req.MetaNot, req.MetaRanges, req.Collection, req.Mode, req.ScoreMode), store.Seqs[ids[end-1]])
		}

		rDocs, rerankScores, stats, err := reranker.Rerank(req.Query, docs, req.Limit)
		if err != nil {
			http.Error(w, "rerank error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if len(respScores) == 0 {
			respScores = rerankScores
		}
		// Select codec based on Accept header (JSON default, SJSON opt-in)
		responseCodec := codec.FromRequest(r)
		w.Header().Set("Content-Type", responseCodec.ContentType())

		response := map[string]any{
			"ids":    respIDs,
			"docs":   rDocs,
			"scores": respScores,
			"stats":  stats,
			"meta":   respMeta,
			"next":   nextPage,
		}

		if err := responseCodec.Encode(w, response); err != nil {
			// Log error but don't change response (already started writing)
			fmt.Printf("error encoding response: %v\n", err)
		}
		return
	})))

	mux.HandleFunc("/delete", withMetrics("delete", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

		// Extract tenant context
		tenantCtx := GetTenantContextFromRequest(r, store.jwtMgr)
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

		r.Body = http.MaxBytesReader(w, r.Body, 1*1024*1024) // 1MB limit

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
			http.Error(w, fmt.Sprintf("failed to delete document: %v", err), http.StatusInternalServerError)
			return
		}

		// Removed synchronous save - rely on WAL + background snapshots

		if err := encodeResponse(w, r, map[string]any{"deleted": req.ID}); err != nil {
			fmt.Printf("error encoding response: %v\n", err)
		}
	})))

	mux.HandleFunc("/health", withMetrics("health", guard(func(w http.ResponseWriter, r *http.Request) {
		store.RLock()
		total := store.Count
		deleted := len(store.Deleted)
		active := total - deleted
		lastSaved := store.lastSaved
		_, embedderIsONNX := embedder.(*OnnxEmbedder)
		_, rerankerIsONNX := reranker.(*OnnxCrossEncoderReranker)
		store.RUnlock()
		snapAge := ageMillis(indexPath, lastSaved)
		walAge := ageMillis(store.walPath, time.Time{})

		_ = encodeResponse(w, r, map[string]any{
			"ok":              true,
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
				"type": map[bool]string{true: "onnx", false: "hash"}[embedderIsONNX],
			},
			"reranker": map[string]any{
				"type": map[bool]string{true: "onnx", false: "simple"}[rerankerIsONNX],
			},
		})
	})))

	mux.Handle("/metrics", promhttp.Handler())

	mux.HandleFunc("/integrity", withMetrics("integrity", guard(func(w http.ResponseWriter, r *http.Request) {
		store.RLock()
		ck := store.validateChecksum()
		hnswOK := true
		if store.hnsw == nil {
			hnswOK = false
		}
		store.RUnlock()
		_ = encodeResponse(w, r, map[string]any{
			"checksum_ok": ck,
			"hnsw_ok":     hnswOK,
		})
	})))

	mux.HandleFunc("/compact", withMetrics("compact", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		if err := store.Compact(indexPath); err != nil {
			http.Error(w, fmt.Sprintf("compact failed: %v", err), http.StatusInternalServerError)
			return
		}
		_ = encodeResponse(w, r, map[string]any{"ok": true})
	})))

	// Export snapshot (read-only)
	mux.HandleFunc("/export", withMetrics("export", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.WriteHeader(http.StatusMethodNotAllowed)
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

		// Add request size limit (max 1GB for snapshot)
		r.Body = http.MaxBytesReader(w, r.Body, 1*1024*1024*1024)

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

		// Replace store atomically
		store.Lock()
		oldStore := *store
		*store = *newStore
		store.walPath = oldStore.walPath
		store.apiToken = oldStore.apiToken
		store.rl = oldStore.rl
		store.Unlock()

		// Save new store
		if err := store.Save(indexPath); err != nil {
			// Rollback on failure
			store.Lock()
			*store = oldStore
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
			tenantCtx := GetTenantContextFromRequest(r, store.jwtMgr)
			if !tenantCtx.IsAdmin {
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

		_ = encodeResponse(w, r, map[string]any{
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

		_ = encodeResponse(w, r, map[string]any{
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

		_ = encodeResponse(w, r, map[string]any{
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

		_ = encodeResponse(w, r, map[string]any{
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

		_ = encodeResponse(w, r, map[string]any{
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

		_ = encodeResponse(w, r, map[string]any{
			"tenant_id":        tenantID,
			"used_bytes":       usedBytes,
			"max_bytes":        maxBytes,
			"vector_count":     vectorCount,
			"utilization_pct":  utilizationPct,
			"has_quota_limit":  maxBytes > 0,
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

		_ = encodeResponse(w, r, map[string]any{
			"ok":        true,
			"tenant_id": req.TenantID,
			"rps":       req.RPS,
			"burst":     req.Burst,
		})
	}))

	return mux
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

func hashFilters(meta map[string]string, any []map[string]string, not map[string]string, ranges []RangeFilter, coll string, mode string, scoreMode string) string {
	type filterHash struct {
		Meta      map[string]string   `json:"meta"`
		Any       []map[string]string `json:"any"`
		Not       map[string]string   `json:"not"`
		Ranges    []RangeFilter       `json:"ranges"`
		Coll      string              `json:"coll"`
		Mode      string              `json:"mode"`
		ScoreMode string              `json:"score_mode"`
	}
	payload := filterHash{
		Meta:      meta,
		Any:       any,
		Not:       not,
		Ranges:    ranges,
		Coll:      coll,
		Mode:      mode,
		ScoreMode: scoreMode,
	}
	b, _ := json.Marshal(payload)
	sum := fnv.New64a()
	_, _ = sum.Write(b)
	return fmt.Sprintf("%x", sum.Sum64())
}

// Compact rebuilds the in-memory HNSW and purges tombstones, then saves a snapshot.
func (vs *VectorStore) Compact(path string) error {
	vs.Lock()
	defer vs.Unlock()
	g := hnsw.NewGraph[uint64]()
	g.Distance = vs.hnsw.Distance
	g.M = vs.hnsw.M
	g.Ml = vs.hnsw.Ml
	g.EfSearch = vs.hnsw.EfSearch

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
		g.Add(hnsw.MakeNode(hid, vec))
		vs.idToIx[hid] = base
	}
	vs.Data = newData
	vs.Docs = newDocs
	vs.IDs = newIDs
	vs.Count = len(newDocs)
	vs.hnsw = g
	if err := vs.Save(path); err != nil {
		return err
	}
	return nil
}
