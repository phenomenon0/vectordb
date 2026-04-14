package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sort"
	"testing"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// End-to-end HTTP search benchmark.
// Tests the full /v2/search path: JSON decode → collection search → JSON encode.
// This validates Fix 1 (ef_search passthrough), Fix 2 (vector stripping),
// Fix 3 (lightweight doc copy), and Fix 4 (JSON struct tags).

func setupE2EBench(b *testing.B, numDocs, dim int) *CollectionHTTPServer {
	b.Helper()

	srv := NewCollectionHTTPServer(b.TempDir())

	// Create collection
	schema := vcollection.CollectionSchema{
		Name: "bench",
		Fields: []vcollection.VectorField{
			{
				Name: "embedding",
				Type: vcollection.VectorTypeDense,
				Dim:  dim,
				Index: vcollection.IndexConfig{
					Type: vcollection.IndexTypeHNSW,
					Params: map[string]interface{}{
						"m": 16, "ef_construction": 200,
					},
				},
			},
		},
	}

	ctx := b.Context()
	if _, err := srv.manager.CreateCollection(ctx, schema); err != nil {
		b.Fatalf("create collection: %v", err)
	}

	// Insert documents with vectors + metadata
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < numDocs; i++ {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rng.Float32()*2 - 1
		}
		doc := vcollection.Document{
			Vectors:  map[string]interface{}{"embedding": vec},
			Metadata: map[string]interface{}{"category": fmt.Sprintf("cat-%d", i%10), "score": float64(i)},
		}
		if err := srv.manager.AddDocument(ctx, "bench", &doc); err != nil {
			b.Fatalf("insert doc %d: %v", i, err)
		}
	}

	return srv
}

func makeSearchBody(dim, topK, efSearch int, includeVectors bool, rng *rand.Rand) []byte {
	query := make([]float64, dim)
	for i := range query {
		query[i] = float64(rng.Float32()*2 - 1)
	}
	req := map[string]interface{}{
		"collection": "bench",
		"queries":    map[string]interface{}{"embedding": query},
		"top_k":      topK,
	}
	if efSearch > 0 {
		req["ef_search"] = efSearch
	}
	if includeVectors {
		req["include_vectors"] = true
	}
	body, _ := json.Marshal(req)
	return body
}

// BenchmarkE2E_Search_NoVectors benchmarks full HTTP search with vectors stripped (default).
func BenchmarkE2E_Search_NoVectors(b *testing.B) {
	for _, tc := range []struct {
		name    string
		numDocs int
		dim     int
		topK    int
	}{
		{"1K_384d_top10", 1000, 384, 10},
		{"1K_1536d_top100", 1000, 1536, 100},
		{"10K_384d_top10", 10_000, 384, 10},
		{"10K_1536d_top100", 10_000, 1536, 100},
	} {
		b.Run(tc.name, func(b *testing.B) {
			srv := setupE2EBench(b, tc.numDocs, tc.dim)
			mux := http.NewServeMux()
			passthrough := func(h http.HandlerFunc) http.HandlerFunc { return h }
			srv.RegisterHandlers(mux, passthrough, passthrough)

			rng := rand.New(rand.NewSource(99))
			bodies := make([][]byte, 100)
			for i := range bodies {
				bodies[i] = makeSearchBody(tc.dim, tc.topK, 0, false, rng)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				body := bodies[i%len(bodies)]
				req := httptest.NewRequest(http.MethodPost, "/v2/search", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				mux.ServeHTTP(w, req)
				if w.Code != 200 {
					b.Fatalf("status %d: %s", w.Code, w.Body.String())
				}
			}
		})
	}
}

// BenchmarkE2E_Search_WithVectors benchmarks with include_vectors=true for comparison.
func BenchmarkE2E_Search_WithVectors(b *testing.B) {
	for _, tc := range []struct {
		name    string
		numDocs int
		dim     int
		topK    int
	}{
		{"1K_384d_top10", 1000, 384, 10},
		{"1K_1536d_top100", 1000, 1536, 100},
		{"10K_384d_top10", 10_000, 384, 10},
		{"10K_1536d_top100", 10_000, 1536, 100},
	} {
		b.Run(tc.name, func(b *testing.B) {
			srv := setupE2EBench(b, tc.numDocs, tc.dim)
			mux := http.NewServeMux()
			passthrough := func(h http.HandlerFunc) http.HandlerFunc { return h }
			srv.RegisterHandlers(mux, passthrough, passthrough)

			rng := rand.New(rand.NewSource(99))
			bodies := make([][]byte, 100)
			for i := range bodies {
				bodies[i] = makeSearchBody(tc.dim, tc.topK, 0, true, rng)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				body := bodies[i%len(bodies)]
				req := httptest.NewRequest(http.MethodPost, "/v2/search", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				mux.ServeHTTP(w, req)
				if w.Code != 200 {
					b.Fatalf("status %d: %s", w.Code, w.Body.String())
				}
			}
		})
	}
}

// BenchmarkE2E_Search_EfSearch benchmarks different ef_search values to validate Fix 1.
func BenchmarkE2E_Search_EfSearch(b *testing.B) {
	srv := setupE2EBench(b, 10_000, 384)
	mux := http.NewServeMux()
	passthrough := func(h http.HandlerFunc) http.HandlerFunc { return h }
	srv.RegisterHandlers(mux, passthrough, passthrough)

	for _, ef := range []int{32, 64, 128, 256, 512} {
		b.Run(fmt.Sprintf("ef=%d", ef), func(b *testing.B) {
			rng := rand.New(rand.NewSource(99))
			bodies := make([][]byte, 100)
			for i := range bodies {
				bodies[i] = makeSearchBody(384, 10, ef, false, rng)
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				body := bodies[i%len(bodies)]
				req := httptest.NewRequest(http.MethodPost, "/v2/search", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()
				mux.ServeHTTP(w, req)
				if w.Code != 200 {
					b.Fatalf("status %d: %s", w.Code, w.Body.String())
				}
			}
		})
	}
}

// BenchmarkE2E_ResponseSize measures JSON response payload sizes.
func BenchmarkE2E_ResponseSize(b *testing.B) {
	srv := setupE2EBench(b, 1000, 1536)
	mux := http.NewServeMux()
	passthrough := func(h http.HandlerFunc) http.HandlerFunc { return h }
	srv.RegisterHandlers(mux, passthrough, passthrough)

	rng := rand.New(rand.NewSource(99))

	for _, tc := range []struct {
		name           string
		topK           int
		includeVectors bool
	}{
		{"top10_no_vectors", 10, false},
		{"top10_with_vectors", 10, true},
		{"top100_no_vectors", 100, false},
		{"top100_with_vectors", 100, true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			body := makeSearchBody(1536, tc.topK, 0, tc.includeVectors, rng)
			req := httptest.NewRequest(http.MethodPost, "/v2/search", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			mux.ServeHTTP(w, req)
			if w.Code != 200 {
				b.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			b.ReportMetric(float64(w.Body.Len()), "response_bytes")
			b.ReportMetric(float64(w.Body.Len())/1024, "response_KB")
		})
	}
}

// TestE2E_SearchLatencyDistribution runs a latency distribution test (not a benchmark)
// to show p50/p95/p99 for the full HTTP search path.
func TestE2E_SearchLatencyDistribution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e latency distribution test in short mode")
	}

	for _, tc := range []struct {
		name           string
		numDocs        int
		dim            int
		topK           int
		includeVectors bool
		numQueries     int
	}{
		{"10K_1536d_top100_no_vectors", 10_000, 1536, 100, false, 500},
		{"10K_1536d_top100_with_vectors", 10_000, 1536, 100, true, 500},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := NewCollectionHTTPServer(t.TempDir())
			schema := vcollection.CollectionSchema{
				Name: "bench",
				Fields: []vcollection.VectorField{
					{
						Name: "embedding",
						Type: vcollection.VectorTypeDense,
						Dim:  tc.dim,
						Index: vcollection.IndexConfig{
							Type: vcollection.IndexTypeHNSW,
							Params: map[string]interface{}{
								"m": 16, "ef_construction": 200,
							},
						},
					},
				},
			}
			ctx := t.Context()
			if _, err := srv.manager.CreateCollection(ctx, schema); err != nil {
				t.Fatalf("create collection: %v", err)
			}

			rng := rand.New(rand.NewSource(42))
			for i := 0; i < tc.numDocs; i++ {
				vec := make([]float32, tc.dim)
				for j := range vec {
					vec[j] = rng.Float32()*2 - 1
				}
				doc := vcollection.Document{
					Vectors:  map[string]interface{}{"embedding": vec},
					Metadata: map[string]interface{}{"idx": float64(i)},
				}
				if err := srv.manager.AddDocument(ctx, "bench", &doc); err != nil {
					t.Fatalf("insert doc %d: %v", i, err)
				}
			}

			mux := http.NewServeMux()
			passthrough := func(h http.HandlerFunc) http.HandlerFunc { return h }
			srv.RegisterHandlers(mux, passthrough, passthrough)

			latencies := make([]time.Duration, tc.numQueries)
			var totalResponseBytes int64

			for i := 0; i < tc.numQueries; i++ {
				body := makeSearchBody(tc.dim, tc.topK, 128, tc.includeVectors, rng)
				req := httptest.NewRequest(http.MethodPost, "/v2/search", bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				w := httptest.NewRecorder()

				start := time.Now()
				mux.ServeHTTP(w, req)
				latencies[i] = time.Since(start)

				if w.Code != 200 {
					t.Fatalf("query %d: status %d: %s", i, w.Code, w.Body.String())
				}
				totalResponseBytes += int64(w.Body.Len())
			}

			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
			p50 := latencies[len(latencies)*50/100]
			p95 := latencies[len(latencies)*95/100]
			p99 := latencies[len(latencies)*99/100]
			avgRespKB := float64(totalResponseBytes) / float64(tc.numQueries) / 1024

			t.Logf("=== %s ===", tc.name)
			t.Logf("  Queries:       %d", tc.numQueries)
			t.Logf("  P50:           %v", p50)
			t.Logf("  P95:           %v", p95)
			t.Logf("  P99:           %v", p99)
			t.Logf("  Avg response:  %.1f KB", avgRespKB)
		})
	}
}
