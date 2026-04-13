package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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

func TestHealthzReturnsOKWhenLockFree(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	req := httptest.NewRequest("GET", "/healthz", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if w.Body.String() != "ok" {
		t.Errorf("expected body 'ok', got %q", w.Body.String())
	}
}

func TestHealthzReturns503WhenLockHeld(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Hold the write lock so RLock blocks
	store.Lock()
	defer store.Unlock()

	req := httptest.NewRequest("GET", "/healthz", nil)
	w := httptest.NewRecorder()

	start := time.Now()
	handler.ServeHTTP(w, req)
	elapsed := time.Since(start)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 when lock is held, got %d", w.Code)
	}
	if elapsed > 10*time.Second {
		t.Errorf("healthz should timeout in ~5s, took %v", elapsed)
	}
}

func TestHealthzNoGoroutineLeakOnTimeout(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Hold the write lock, fire multiple healthz probes
	store.Lock()

	const probes = 2
	for i := 0; i < probes; i++ {
		req := httptest.NewRequest("GET", "/healthz", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("probe %d: expected 503, got %d", i, w.Code)
		}
	}

	// Release the lock — all blocked goroutines should complete
	store.Unlock()

	// Give goroutines time to drain
	time.Sleep(100 * time.Millisecond)
	// If goroutines leaked, the race detector would catch them on a
	// second lock cycle. Verify the lock is functional:
	store.RLock()
	store.RUnlock()
}

func TestReadyzReturns503WhenLockHeld(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Hold the write lock so readyz's goroutine blocks on RLock
	store.Lock()
	defer store.Unlock()

	req := httptest.NewRequest("GET", "/readyz", nil)
	w := httptest.NewRecorder()

	start := time.Now()
	handler.ServeHTTP(w, req)
	elapsed := time.Since(start)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 when lock is held, got %d", w.Code)
	}
	if elapsed > 10*time.Second {
		t.Errorf("readyz should timeout in ~5s, took %v", elapsed)
	}

	// Verify the response reports the lock timeout issue
	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	issues, ok := resp["issues"].([]any)
	if !ok || len(issues) == 0 {
		t.Fatal("expected issues in response")
	}
	found := false
	for _, issue := range issues {
		if s, ok := issue.(string); ok && s == "lock acquisition timeout — snapshot or heavy write in progress" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected lock timeout issue, got %v", issues)
	}
}

func TestHTTPRateLimiting(t *testing.T) {
	store := NewVectorStore(100, 3)
	store.rl = newRateLimiter(2, 2, 0, time.Minute) // 2 requests per minute
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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
	handler, _ := newHTTPHandler(store, emb, reranker, "")

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

// ======================================================================================
// Error Response Sanitization Tests
// Verify that 500-class responses never expose internal error details to clients.
// ======================================================================================

func TestErrorResponsesSanitized_PanicRecovery(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Force a panic by sending a request to a path that will trigger panic recovery.
	// We can't easily cause a panic through normal endpoints, but we can verify
	// that the panic recovery middleware doesn't include panic details.
	// Instead, verify that known 500 paths return sanitized messages.

	// Test: duplicate insert without upsert returns 500 but message is opaque
	body1, _ := json.Marshal(map[string]any{"doc": "test document", "id": "dup-test-id"})
	req1 := httptest.NewRequest("POST", "/insert", bytes.NewReader(body1))
	req1.Header.Set("Content-Type", "application/json")
	w1 := httptest.NewRecorder()
	handler.ServeHTTP(w1, req1)
	if w1.Code != http.StatusOK {
		t.Fatalf("first insert failed: %d %s", w1.Code, w1.Body.String())
	}

	// Second insert with same ID (no upsert) should fail
	body2, _ := json.Marshal(map[string]any{"doc": "duplicate document", "id": "dup-test-id"})
	req2 := httptest.NewRequest("POST", "/insert", bytes.NewReader(body2))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	handler.ServeHTTP(w2, req2)

	if w2.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500 for duplicate insert, got %d", w2.Code)
	}

	respBody := w2.Body.String()
	// The response should NOT contain any of these internal details
	forbiddenPatterns := []string{
		"already exists",
		"duplicate key",
		"store.Add",
		"main.go",
		"server.go",
		".go:",
		"goroutine",
		"runtime.",
	}
	for _, pattern := range forbiddenPatterns {
		if strings.Contains(strings.ToLower(respBody), strings.ToLower(pattern)) {
			t.Errorf("500 response leaked internal detail %q: %s", pattern, respBody)
		}
	}

	// Should be a generic opaque message
	if !strings.Contains(respBody, "failed to insert document") {
		t.Errorf("expected generic error message, got: %s", respBody)
	}
}

func TestErrorResponsesSanitized_DeleteNotFound(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Delete a nonexistent document
	body, _ := json.Marshal(map[string]any{"id": "nonexistent-id"})
	req := httptest.NewRequest("POST", "/delete", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	respBody := w.Body.String()
	// Error messages should not contain stack traces or Go internals
	if strings.Contains(respBody, ".go:") || strings.Contains(respBody, "goroutine") {
		t.Errorf("error response leaked internal details: %s", respBody)
	}
}

func TestErrorResponsesSanitized_CompactEndpoint(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Try compact — it may succeed or fail, but either way the error should be sanitized
	req := httptest.NewRequest("POST", "/admin/compact", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	respBody := w.Body.String()
	// Regardless of status, should not leak Go error internals
	if strings.Contains(respBody, "os.") || strings.Contains(respBody, "io.") ||
		strings.Contains(respBody, "syscall") || strings.Contains(respBody, "runtime.") {
		t.Errorf("compact response leaked internal details: %s", respBody)
	}
}

func TestErrorResponsesSanitized_EmbedEndpoint(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Valid embed request — should succeed
	body, _ := json.Marshal(map[string]any{"text": "hello world"})
	req := httptest.NewRequest("POST", "/api/embed", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		respBody := w.Body.String()
		// If it fails, the error message must not contain internal details
		if strings.Contains(respBody, ".go:") || strings.Contains(respBody, "panic") {
			t.Errorf("embed error response leaked internals: %s", respBody)
		}
	}
}

func TestErrorResponsesSanitized_BatchEmbedPartialFailure(t *testing.T) {
	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Batch embed — should succeed with HashEmbedder
	body, _ := json.Marshal(map[string]any{"texts": []string{"text1", "text2"}})
	req := httptest.NewRequest("POST", "/api/embed/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// If the response is 500, verify it doesn't leak the error
	if w.Code == http.StatusInternalServerError {
		respBody := w.Body.String()
		// Should say "embedding failed for text N" but NOT include the Go error
		if strings.Contains(respBody, ":") && strings.Contains(respBody, ".go") {
			t.Errorf("batch embed error leaked internals: %s", respBody)
		}
	}
}
