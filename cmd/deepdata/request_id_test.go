package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/phenomenon0/vectordb/internal/logging"
)

// TestRequestIDMiddlewareGeneratesID verifies that every response carries
// an X-Request-ID header even when the client doesn't provide one.
func TestRequestIDMiddlewareGeneratesID(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)

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
	handler := newCanonicalSurfaceTestHandler(t)

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
	handler := newCanonicalSurfaceTestHandler(t)

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
	handler := newCanonicalSurfaceTestHandler(t)

	// A route the canonical surface doesn't serve — 404 from canonicalRCSurface.
	req := httptest.NewRequest(http.MethodGet, "/definitely-not-a-route", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 from canonicalRCSurface, got %d", rec.Code)
	}
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

// TestRequestIDTruncationKeepsValidUTF8 verifies that a client-supplied ID
// longer than the cap is truncated to a valid UTF-8 string rather than being
// sliced at an arbitrary byte boundary (which could corrupt the echoed header
// and the structured log field).
func TestRequestIDTruncationKeepsValidUTF8(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)

	// 127 ASCII bytes + a 2-byte rune + another 2-byte rune: a naive 128-byte
	// slice would cut the first é mid-rune.
	clientID := strings.Repeat("a", 127) + "é" + "é"
	if len(clientID) <= 128 {
		t.Fatalf("test setup requires a >128-byte ID, got %d", len(clientID))
	}

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set("X-Request-ID", clientID)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	got := rec.Header().Get("X-Request-ID")
	if len(got) > 128 {
		t.Errorf("truncated X-Request-ID length = %d, want <= 128", len(got))
	}
	if !utf8.ValidString(got) {
		t.Errorf("X-Request-ID is not valid UTF-8 after truncation: %q", got)
	}
}

// TestTruncateRequestID verifies the truncation helper directly.
func TestTruncateRequestID(t *testing.T) {
	if got := truncateRequestID("short", 128); got != "short" {
		t.Errorf("truncateRequestID under limit = %q, want unchanged", got)
	}
	multi := strings.Repeat("é", 70) // 140 bytes, five 2-byte runes over 128
	got := truncateRequestID(multi, 128)
	if len(got) > 128 {
		t.Errorf("truncateRequestID length = %d, want <= 128", len(got))
	}
	if !utf8.ValidString(got) {
		t.Errorf("truncateRequestID produced invalid UTF-8: %q", got)
	}
	if len(got)%2 != 0 {
		t.Errorf("truncateRequestID split a 2-byte rune: %q", got)
	}
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
