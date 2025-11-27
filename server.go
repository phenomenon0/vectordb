package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

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

		qVec, err := embedder.Embed(req.Query)
		if err != nil {
			http.Error(w, "embed error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		var ids []int
		if req.Mode == "scan" {
			ids = store.Search(qVec, req.TopK)
		} else {
			ids = store.SearchANN(qVec, req.TopK)
		}
		docs := make([]string, 0, len(ids))
		respIDs := make([]string, 0, len(ids))
		respMeta := make([]map[string]string, 0, len(ids))
		for _, idx := range ids {
			hid := hashID(store.GetID(idx))
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
		}
		start := req.Offset
		if start > len(docs) {
			start = len(docs)
		}
		end := start + req.Limit
		if end > len(docs) {
			end = len(docs)
		}
		docs = docs[start:end]
		respIDs = respIDs[start:end]
		if req.IncludeMeta && len(respMeta) > 0 {
			respMeta = respMeta[start:end]
		} else {
			respMeta = nil
		}

		rDocs, scores, stats, err := reranker.Rerank(req.Query, docs, req.Limit)
		if err != nil {
			http.Error(w, "rerank error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"ids":    respIDs,
			"docs":   rDocs,
			"scores": scores,
			"stats":  stats,
			"meta":   respMeta,
		})
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
		})
	})))

	mux.Handle("/metrics", promhttp.Handler())
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
