package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/logging"
)

// TestRequestIDMiddlewareGeneratesID verifies that every response carries
// an X-Request-ID header even when the client doesn't provide one.
func TestRequestIDMiddlewareGeneratesID(t *testing.T) {
	store := NewVectorStore(100, 4)
	emb := NewHashEmbedder(4)
	handler, _ := newHTTPHandler(store, emb, nil, "")

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	id := rec.Header().Get("X-Request-ID")
	if id == "" {
		t.Fatal("expected X-Request-ID header on response, got empty")
	}
	if len(id) != 32 {
		t.Errorf("expected 32-char hex request ID, got %d chars: %q", len(id), id)
	}
}

// TestRequestIDMiddlewareReusesClientID verifies that a client-provided
// X-Request-ID is echoed back instead of generating a new one.
func TestRequestIDMiddlewareReusesClientID(t *testing.T) {
	store := NewVectorStore(100, 4)
	emb := NewHashEmbedder(4)
	handler, _ := newHTTPHandler(store, emb, nil, "")

	clientID := "my-custom-trace-id-12345"
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set("X-Request-ID", clientID)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	got := rec.Header().Get("X-Request-ID")
	if got != clientID {
		t.Errorf("expected echoed X-Request-ID %q, got %q", clientID, got)
	}
}

// TestRequestIDMiddlewareUniquePerRequest verifies that two requests
// get different auto-generated IDs.
func TestRequestIDMiddlewareUniquePerRequest(t *testing.T) {
	store := NewVectorStore(100, 4)
	emb := NewHashEmbedder(4)
	handler, _ := newHTTPHandler(store, emb, nil, "")

	req1 := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec1 := httptest.NewRecorder()
	handler.ServeHTTP(rec1, req1)

	req2 := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, req2)

	id1 := rec1.Header().Get("X-Request-ID")
	id2 := rec2.Header().Get("X-Request-ID")
	if id1 == id2 {
		t.Errorf("expected unique request IDs, both got %q", id1)
	}
}

// TestRequestIDOnErrorResponse verifies that error responses also carry
// the X-Request-ID header (important for client-side error correlation).
func TestRequestIDOnErrorResponse(t *testing.T) {
	store := NewVectorStore(100, 4)
	emb := NewHashEmbedder(4)
	handler, _ := newHTTPHandler(store, emb, nil, "")

	// POST to /insert with invalid body — should get an error response
	req := httptest.NewRequest(http.MethodPost, "/insert", strings.NewReader("not-json"))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	id := rec.Header().Get("X-Request-ID")
	if id == "" {
		t.Fatal("expected X-Request-ID on error response, got empty")
	}
}

// TestRequestIDFromContext verifies the requestIDFromContext helper.
func TestRequestIDFromContext(t *testing.T) {
	t.Run("empty context returns empty string", func(t *testing.T) {
		if got := requestIDFromContext(context.Background()); got != "" {
			t.Errorf("expected empty, got %q", got)
		}
	})

	t.Run("context with ID returns the ID", func(t *testing.T) {
		ctx := context.WithValue(context.Background(), logging.RequestIDKey, "test-id-abc")
		if got := requestIDFromContext(ctx); got != "test-id-abc" {
			t.Errorf("expected %q, got %q", "test-id-abc", got)
		}
	})
}

// TestGenerateRequestID verifies the ID generator produces valid hex strings.
func TestGenerateRequestID(t *testing.T) {
	id := generateRequestID()
	if len(id) != 32 {
		t.Errorf("expected 32-char hex string, got %d chars: %q", len(id), id)
	}
	for _, c := range id {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			t.Errorf("non-hex character %q in request ID %q", c, id)
			break
		}
	}
}
