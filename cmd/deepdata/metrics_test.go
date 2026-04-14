package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMetricsEndpointServesCustomMetrics(t *testing.T) {
	// Ensure globalMetrics is initialized (mirrors main() calling initMetrics())
	initMetrics()

	store := NewVectorStore(100, 3)
	emb := NewHashEmbedder(3)
	reranker := &SimpleReranker{Embedder: emb}
	handler, _ := newHTTPHandler(store, emb, reranker, "")

	// Record synthetic metrics so they appear in the scrape output
	globalMetrics.RecordHTTPRequest("GET", "/query", 200, 0)
	globalMetrics.RecordOperation("insert", 0, 0, nil)
	globalMetrics.RecordQuery("dense", []string{"default"}, 0, 10, 1)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("GET /metrics returned %d, want 200", w.Code)
	}

	body, err := io.ReadAll(w.Body)
	if err != nil {
		t.Fatalf("reading response body: %v", err)
	}
	text := string(body)

	// These metrics are registered on the custom registry (mc.registry).
	// Before the fix, promhttp.Handler() served the default global registry
	// which contained none of these — only Go runtime metrics.
	requiredMetrics := []string{
		"vectordb_http_requests_total",
		"vectordb_http_request_duration_seconds",
		"vectordb_operations_total",
		"vectordb_operation_duration_seconds",
		"vectordb_query_duration_seconds",
		"vectordb_query_results",
	}

	for _, m := range requiredMetrics {
		if !strings.Contains(text, m) {
			t.Errorf("GET /metrics missing %q — endpoint may be serving the wrong Prometheus registry", m)
		}
	}

	// Verify the recorded synthetic metric has data with correct labels
	if !strings.Contains(text, `vectordb_http_requests_total{endpoint="/query",method="GET",status="200"} 1`) {
		t.Error("GET /metrics: recorded http_requests_total metric not found with expected labels and value")
	}

	// Verify that Go runtime metrics from the default registry are NOT present
	// (our custom registry should not include them unless explicitly registered)
	if strings.Contains(text, "go_goroutines") {
		t.Error("GET /metrics: contains go_goroutines from default registry — should only serve custom vectordb metrics")
	}
}

func TestMetricsPathNormalization(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"/v1/health", "/v1/health"},
		{"/v2/collections/my-coll", "/v2/collections/:name"},
		{"/v3/tenants/abc123/collections", "/v3/tenants/:id/collections"},
		{"/health", "/health"},
		{"/metrics", "/metrics"},
	}

	for _, tc := range tests {
		got := normalizeMetricsPath(tc.input)
		if got != tc.want {
			t.Errorf("normalizeMetricsPath(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestMetricsHTTPMiddleware(t *testing.T) {
	initMetrics()

	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	})
	wrapped := globalMetrics.HTTPMiddleware(inner)

	req := httptest.NewRequest("POST", "/insert", nil)
	w := httptest.NewRecorder()
	wrapped.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("expected 201, got %d", w.Code)
	}
}

func TestWithMetricsHelper(t *testing.T) {
	initMetrics()

	called := false
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})

	wrapped := withMetrics("/test-endpoint", inner)
	req := httptest.NewRequest("GET", "/test-endpoint", nil)
	w := httptest.NewRecorder()
	wrapped.ServeHTTP(w, req)

	if !called {
		t.Error("inner handler was not called")
	}
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}
