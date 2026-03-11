package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// ======================================================================================
// HTTP Endpoint Integration Tests
// ======================================================================================

func TestHTTPInsertEndpoint(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	// Test successful insert
	reqBody := map[string]any{
		"doc":  "test document",
		"meta": map[string]string{"tag": "test"},
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/insert", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp["id"] == nil {
		t.Error("expected id in response")
	}
}

func TestHTTPInsertValidation(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	tests := []struct {
		name       string
		body       map[string]any
		wantStatus int
	}{
		{
			name:       "empty document",
			body:       map[string]any{"doc": ""},
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "document too large",
			body:       map[string]any{"doc": string(make([]byte, 2_000_000))},
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "too many metadata keys",
			body:       map[string]any{"doc": "test", "meta": func() map[string]string {
				m := make(map[string]string)
				for i := 0; i < 150; i++ {
					m[fmt.Sprintf("key%d", i)] = "value"
				}
				return m
			}()},
			wantStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, _ := json.Marshal(tt.body)
			req := httptest.NewRequest("POST", "/insert", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Errorf("expected status %d, got %d: %s", tt.wantStatus, w.Code, w.Body.String())
			}
		})
	}
}

func TestHTTPBatchInsertEndpoint(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	reqBody := map[string]any{
		"docs": []map[string]any{
			{"doc": "doc1", "meta": map[string]string{"tag": "a"}},
			{"doc": "doc2", "meta": map[string]string{"tag": "b"}},
			{"doc": "doc3", "meta": map[string]string{"tag": "c"}},
		},
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/batch_insert", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	ids, ok := resp["ids"].([]any)
	if !ok {
		t.Fatal("expected ids array in response")
	}

	if len(ids) != 3 {
		t.Errorf("expected 3 ids, got %d", len(ids))
	}
}

func TestHTTPQueryEndpoint(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	// Insert some documents first
	for i := 0; i < 10; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		store.Add(vec, fmt.Sprintf("content-%d", i), "", map[string]string{"tag": "test"}, "default", "")
	}

	reqBody := map[string]any{
		"query": "test query",
		"top_k": 5,
		"mode":  "ann",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	ids, ok := resp["ids"].([]any)
	if !ok {
		t.Fatal("expected ids array in response")
	}

	if len(ids) > 5 {
		t.Errorf("expected at most 5 results, got %d", len(ids))
	}
}

func TestHTTPQueryValidation(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	tests := []struct {
		name       string
		body       map[string]any
		wantStatus int
	}{
		{
			name:       "query too long",
			body:       map[string]any{"query": string(make([]byte, 20_000)), "top_k": 10},
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "top_k too large",
			body:       map[string]any{"query": "test", "top_k": 2000},
			wantStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, _ := json.Marshal(tt.body)
			req := httptest.NewRequest("POST", "/query", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Errorf("expected status %d, got %d: %s", tt.wantStatus, w.Code, w.Body.String())
			}
		})
	}
}

func TestHTTPDeleteEndpoint(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	// Insert a document
	vec, _ := emb.Embed("test doc")
	id, _ := store.Add(vec, "test content", "test-id", nil, "default", "")

	reqBody := map[string]any{"id": id}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/delete", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify deletion
	if !store.Deleted[hashID(id)] {
		t.Error("document was not deleted")
	}
}

func TestHTTPHealthEndpoint(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	// Add some documents
	for i := 0; i < 5; i++ {
		vec, _ := emb.Embed(fmt.Sprintf("doc-%d", i))
		store.Add(vec, fmt.Sprintf("content-%d", i), "", nil, "default", "")
	}

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp["ok"] != true {
		t.Error("expected ok: true in health response")
	}

	if resp["total"] == nil {
		t.Error("expected total in health response")
	}
}

func TestHTTPRateLimiting(t *testing.T) {
	store := NewVectorStore(100, 3)
	store.rl = newRateLimiter(2, 2, 0, time.Minute) // 2 requests per minute
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	// Make 3 requests from same IP
	for i := 0; i < 3; i++ {
		reqBody := map[string]any{"doc": fmt.Sprintf("doc-%d", i)}
		body, _ := json.Marshal(reqBody)

		req := httptest.NewRequest("POST", "/insert", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.RemoteAddr = "192.168.1.1:12345"
		w := httptest.NewRecorder()

		handler.ServeHTTP(w, req)

		// First 2 requests should succeed, 3rd should be rate limited
		if i < 2 {
			if w.Code == http.StatusTooManyRequests {
				t.Errorf("request %d should not be rate limited", i)
			}
		} else {
			if w.Code != http.StatusTooManyRequests {
				t.Errorf("request %d should be rate limited, got status %d", i, w.Code)
			}
		}
	}
}

func TestHTTPRequestSizeLimits(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	// Create a request larger than 10MB (insert endpoint limit)
	largeDoc := string(make([]byte, 11*1024*1024))
	reqBody := map[string]any{"doc": largeDoc}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/insert", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400 for oversized request, got %d", w.Code)
	}
}

func TestHTTPUpsertFunctionality(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler := newHTTPHandler(store, emb, reranker, "")

	// Insert with ID
	reqBody := map[string]any{
		"id":     "test-id",
		"doc":    "original content",
		"upsert": false,
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/insert", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("initial insert failed: %d", w.Code)
	}

	// Upsert same ID with new content
	reqBody2 := map[string]any{
		"id":     "test-id",
		"doc":    "updated content",
		"upsert": true,
	}
	body2, _ := json.Marshal(reqBody2)

	req2 := httptest.NewRequest("POST", "/insert", bytes.NewReader(body2))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()

	handler.ServeHTTP(w2, req2)

	if w2.Code != http.StatusOK {
		t.Fatalf("upsert failed: %d", w2.Code)
	}

	// Verify upsert replaced content
	hid := hashID("test-id")
	idx := store.idToIx[hid]
	if store.Docs[idx] != "updated content" {
		t.Errorf("expected 'updated content', got '%s'", store.Docs[idx])
	}

	// Verify only one document exists
	if store.Count != 1 {
		t.Errorf("expected 1 document, got %d", store.Count)
	}
}
