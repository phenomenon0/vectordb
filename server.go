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
	"time"

	"github.com/coder/hnsw"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

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
			if store.apiToken != "" {
				if token != "Bearer "+store.apiToken && token != store.apiToken {
					http.Error(w, "unauthorized", http.StatusUnauthorized)
					return
				}
			}
			if store.rl != nil {
				key := token
				if key == "" {
					key = "anon"
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
		var req struct {
			ID         string            `json:"id"`
			Doc        string            `json:"doc"`
			Meta       map[string]string `json:"meta"`
			Upsert     bool              `json:"upsert"`
			Collection string            `json:"collection"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.Doc == "" {
			http.Error(w, "doc required", http.StatusBadRequest)
			return
		}
		vec, err := embedder.Embed(req.Doc)
		if err != nil {
			http.Error(w, "embed error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		var id string
		if req.Upsert {
			id = store.Upsert(vec, req.Doc, req.ID, req.Meta, req.Collection)
		} else {
			id = store.Add(vec, req.Doc, req.ID, req.Meta, req.Collection)
		}
		if err := store.Save(indexPath); err != nil {
			fmt.Printf("warning: save failed: %v\n", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"id": id})
	})))

	mux.HandleFunc("/batch_insert", withMetrics("batch_insert", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Docs []struct {
				ID         string            `json:"id"`
				Doc        string            `json:"doc"`
				Meta       map[string]string `json:"meta"`
				Collection string            `json:"collection"`
			} `json:"docs"`
			Upsert bool `json:"upsert"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if len(req.Docs) == 0 {
			http.Error(w, "no docs provided", http.StatusBadRequest)
			return
		}
		ids := make([]string, 0, len(req.Docs))
		for _, d := range req.Docs {
			if d.Doc == "" {
				continue
			}
			vec, err := embedder.Embed(d.Doc)
			if err != nil {
				continue
			}
			if req.Upsert {
				ids = append(ids, store.Upsert(vec, d.Doc, d.ID, d.Meta, d.Collection))
			} else {
				ids = append(ids, store.Add(vec, d.Doc, d.ID, d.Meta, d.Collection))
			}
		}
		if err := store.Save(indexPath); err != nil {
			fmt.Printf("warning: save failed: %v\n", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"ids": ids})
	})))

	mux.HandleFunc("/query", withMetrics("query", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
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
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.TopK == 0 {
			req.TopK = 3
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

		qVec, err := embedder.Embed(req.Query)
		if err != nil {
			http.Error(w, "embed error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		var ids []int
		if req.Mode == "scan" {
			ids = store.Search(qVec, req.TopK)
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
		qTokens := tokenize(req.Query)
		docs := make([]string, 0, len(ids))
		respIDs := make([]string, 0, len(ids))
		respMeta := make([]map[string]string, 0, len(ids))
		respScores := make([]float32, 0, len(ids))
		type scored struct {
			docIdx int
			score  float64
		}
		hybridScores := make([]scored, 0, len(ids))
		for _, idx := range ids {
			hid := hashID(store.GetID(idx))
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
			if req.Mode == "hybrid" {
				score := store.hybridScore(hid, qVec, qTokens, req.HybridAlpha)
				hybridScores = append(hybridScores, scored{docIdx: len(docs) - 1, score: score})
			}
		}
		if req.Mode == "hybrid" && len(hybridScores) > 0 {
			sort.Slice(hybridScores, func(i, j int) bool { return hybridScores[i].score > hybridScores[j].score })
			reDocs := make([]string, 0, len(hybridScores))
			reIDs := make([]string, 0, len(hybridScores))
			reScores := make([]float32, 0, len(hybridScores))
			var reMeta []map[string]string
			if req.IncludeMeta {
				reMeta = make([]map[string]string, 0, len(hybridScores))
			}
			for _, s := range hybridScores {
				reDocs = append(reDocs, docs[s.docIdx])
				reIDs = append(reIDs, respIDs[s.docIdx])
				if req.ScoreMode == "hybrid" {
					reScores = append(reScores, float32(s.score))
				} else if req.ScoreMode == "vector" {
					reScores = append(reScores, respScores[s.docIdx])
				}
				if req.IncludeMeta {
					reMeta = append(reMeta, respMeta[s.docIdx])
				}
			}
			docs = reDocs
			respIDs = reIDs
			if req.ScoreMode == "hybrid" || req.ScoreMode == "vector" {
				respScores = reScores
			}
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
			nextPage = encodePageToken(offset+pageSize, hashFilters(req.Meta, req.MetaAny, req.MetaNot, req.MetaRanges, req.Collection, req.Mode, req.ScoreMode))
		}

		rDocs, rerankScores, stats, err := reranker.Rerank(req.Query, docs, req.Limit)
		if err != nil {
			http.Error(w, "rerank error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if len(respScores) == 0 {
			respScores = rerankScores
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ids":    respIDs,
			"docs":   rDocs,
			"scores": respScores,
			"stats":  stats,
			"meta":   respMeta,
			"next":   nextPage,
		})
		return
	})))

	mux.HandleFunc("/delete", withMetrics("delete", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			ID string `json:"id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.ID == "" {
			http.Error(w, "id required", http.StatusBadRequest)
			return
		}
		store.Delete(req.ID)
		if err := store.Save(indexPath); err != nil {
			fmt.Printf("warning: save failed: %v\n", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"deleted": req.ID})
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

		_ = json.NewEncoder(w).Encode(map[string]any{
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
		_ = json.NewEncoder(w).Encode(map[string]any{
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
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
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

	// Import snapshot (overwrites current index)
	mux.HandleFunc("/import", withMetrics("import", guard(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		tmp, err := os.CreateTemp("", "vectordb-import-*.gob")
		if err != nil {
			http.Error(w, "tmp file error", http.StatusInternalServerError)
			return
		}
		defer os.Remove(tmp.Name())
		if _, err := io.Copy(tmp, r.Body); err != nil {
			http.Error(w, "copy failed", http.StatusInternalServerError)
			return
		}
		if err := tmp.Close(); err != nil {
			http.Error(w, "close failed", http.StatusInternalServerError)
			return
		}
		newStore, loaded := loadOrInitStore(tmp.Name(), store.Count+1, store.Dim)
		if !loaded {
			http.Error(w, "import failed", http.StatusBadRequest)
			return
		}
		store.Lock()
		*store = *newStore
		store.Unlock()
		if err := store.Save(indexPath); err != nil {
			http.Error(w, "save failed", http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})))

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
}

func encodePageToken(offset int, filterHash string) string {
	cur := pageCursor{Offset: offset, FilterHash: filterHash}
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
