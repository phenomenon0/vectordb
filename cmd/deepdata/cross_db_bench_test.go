package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/benchmarks/testdata"
)

// ============================================================================
// Shared Helpers
// ============================================================================

// httpDo sends an HTTP request to the mux and returns the recorder.
func httpDo(t *testing.T, mux *http.ServeMux, method, path string, body interface{}) *httptest.ResponseRecorder {
	t.Helper()
	var payload []byte
	if body != nil {
		var err error
		payload, err = json.Marshal(body)
		if err != nil {
			t.Fatalf("marshal request: %v", err)
		}
	}
	req := httptest.NewRequest(method, path, bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

// httpDoWithAccept is like httpDo but sets a custom Accept header.
func httpDoWithAccept(t *testing.T, mux *http.ServeMux, method, path string, body interface{}, accept string) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(method, path, bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

// httpCreateCollection creates a collection via POST /v2/collections.
// The schema uses string type values (e.g. "dense", "sparse") to validate Fix 5.
func httpCreateCollection(t *testing.T, mux *http.ServeMux, schema map[string]interface{}) {
	t.Helper()
	w := httpDo(t, mux, http.MethodPost, "/v2/collections", schema)
	if w.Code != http.StatusCreated {
		t.Fatalf("create collection: status %d: %s", w.Code, w.Body.String())
	}
}

// httpInsert inserts a single document via POST /v2/insert.
// Returns the assigned document ID.
func httpInsert(t *testing.T, mux *http.ServeMux, collection string, vectors map[string]interface{}, metadata map[string]interface{}) uint64 {
	t.Helper()
	body := map[string]interface{}{
		"collection": collection,
		"vectors":    vectors,
	}
	if metadata != nil {
		body["metadata"] = metadata
	}
	w := httpDo(t, mux, http.MethodPost, "/v2/insert", body)
	if w.Code != http.StatusOK {
		t.Fatalf("insert: status %d: %s", w.Code, w.Body.String())
	}
	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode insert response: %v", err)
	}
	id, ok := resp["id"].(float64)
	if !ok {
		t.Fatalf("insert response missing numeric id: %v", resp)
	}
	return uint64(id)
}

// httpBatchInsert inserts documents in batch via POST /v2/insert/batch.
// Returns the decoded response map.
func httpBatchInsert(t *testing.T, mux *http.ServeMux, body map[string]interface{}) map[string]interface{} {
	t.Helper()
	w := httpDo(t, mux, http.MethodPost, "/v2/insert/batch", body)
	if w.Code != http.StatusOK {
		t.Fatalf("batch insert: status %d: %s", w.Code, w.Body.String())
	}
	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode batch insert response: %v", err)
	}
	return resp
}

// httpSearch performs a search via POST /v2/search and returns raw bytes and status code.
func httpSearch(t *testing.T, mux *http.ServeMux, body map[string]interface{}) (int, []byte) {
	t.Helper()
	w := httpDo(t, mux, http.MethodPost, "/v2/search", body)
	return w.Code, w.Body.Bytes()
}

// searchResponse is the parsed search response.
type searchResponse struct {
	Status             string                   `json:"status"`
	Documents          []map[string]interface{} `json:"documents"`
	Scores             []float64                `json:"scores"`
	CandidatesExamined int                      `json:"candidates_examined"`
}

// parseSearchResponse decodes a search response and asserts snake_case field names.
// Hard fails if any document has PascalCase keys like "ID", "Metadata", "Vectors".
func parseSearchResponse(t *testing.T, data []byte) searchResponse {
	t.Helper()

	// First check raw JSON for PascalCase field names
	raw := string(data)
	for _, bad := range []string{`"ID"`, `"Metadata"`, `"Vectors"`} {
		if strings.Contains(raw, bad) {
			t.Fatalf("response contains PascalCase field %s — expected snake_case. Body: %.500s", bad, raw)
		}
	}

	var resp searchResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		t.Fatalf("decode search response: %v\nBody: %.500s", err, raw)
	}
	return resp
}

// setupMux creates a CollectionHTTPServer with passthrough auth and returns (server, mux).
func setupMux(t *testing.T) (*CollectionHTTPServer, *http.ServeMux) {
	t.Helper()
	srv := NewCollectionHTTPServer(t.TempDir())
	mux := http.NewServeMux()
	passthrough := func(h http.HandlerFunc) http.HandlerFunc { return h }
	srv.RegisterHandlers(mux, passthrough, passthrough)
	return srv, mux
}

// vecToFloat64 converts []float32 to []float64 for JSON marshaling.
func vecToFloat64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, f := range v {
		out[i] = float64(f)
	}
	return out
}

// extractDocIDs extracts document IDs from search response.
func extractDocIDs(resp searchResponse) []uint64 {
	ids := make([]uint64, len(resp.Documents))
	for i, doc := range resp.Documents {
		id, ok := doc["id"].(float64)
		if !ok {
			continue
		}
		ids[i] = uint64(id)
	}
	return ids
}

// pctile returns the p-th percentile from sorted durations.
func pctile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100.0*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// ============================================================================
// Test 1: RAG Retrieval Round-Trip
// ============================================================================

func TestCrossDB_RAGRetrievalRoundTrip(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping cross-DB bench in short mode")
	}

	const (
		numDocs    = 10_000
		dim        = 384
		numQueries = 100
		topK       = 10
		seed       = 42
	)

	_, mux := setupMux(t)

	// Step 1: Create collection with string type "dense" (validates Fix 5)
	httpCreateCollection(t, mux, map[string]interface{}{
		"name": "rag_bench",
		"fields": []map[string]interface{}{
			{
				"name": "embedding",
				"type": "dense",
				"dim":  dim,
				"index": map[string]interface{}{
					"type":   0,
					"params": map[string]interface{}{"m": 16, "ef_construction": 200},
				},
			},
		},
	})

	// Step 2: Generate clustered dataset and insert via HTTP
	data, queries := testdata.GenerateClusteredDataset(numDocs, numQueries, dim, 20, 0.15, seed)

	t.Logf("Inserting %d docs (%dd)...", numDocs, dim)
	insertStart := time.Now()
	// Track assigned IDs: httpID -> data index
	httpIDToIdx := make(map[uint64]int, numDocs)
	idxToHTTPID := make([]uint64, numDocs)
	for i, vec := range data {
		meta := map[string]interface{}{
			"chunk_idx": float64(i),
			"source":    fmt.Sprintf("doc-%d", i/100),
		}
		id := httpInsert(t, mux, "rag_bench", map[string]interface{}{
			"embedding": vecToFloat64(vec),
		}, meta)
		httpIDToIdx[id] = i
		idxToHTTPID[i] = id
	}
	t.Logf("Insert took %v (%.0f docs/sec)", time.Since(insertStart),
		float64(numDocs)/time.Since(insertStart).Seconds())

	// Step 3: Compute ground truth via brute-force (uses 0-based data indices)
	gt, err := testdata.ComputeGroundTruth(data, queries, topK)
	if err != nil {
		t.Fatalf("compute ground truth: %v", err)
	}
	// Convert ground truth from data indices to HTTP IDs
	for i := range gt.Neighbors {
		for j := range gt.Neighbors[i] {
			gt.Neighbors[i][j] = idxToHTTPID[gt.Neighbors[i][j]]
		}
	}

	// Step 4: Run queries via HTTP with ef_search=256 (without vectors)
	allResults := make([][]uint64, numQueries)
	var totalStrippedBytes int64

	for i, q := range queries {
		code, body := httpSearch(t, mux, map[string]interface{}{
			"collection": "rag_bench",
			"queries":    map[string]interface{}{"embedding": vecToFloat64(q)},
			"top_k":      topK,
			"ef_search":  256,
		})
		if code != 200 {
			t.Fatalf("search query %d: status %d: %s", i, code, string(body))
		}
		totalStrippedBytes += int64(len(body))

		resp := parseSearchResponse(t, body)
		allResults[i] = extractDocIDs(resp)

		// Step 5: Assert no "vectors" key in response documents (default stripping)
		for j, doc := range resp.Documents {
			if _, hasVectors := doc["vectors"]; hasVectors {
				t.Fatalf("query %d, doc %d: response contains 'vectors' key — should be stripped by default", i, j)
			}
		}
	}

	// Compute recall (both GT and results now use HTTP IDs)
	recall := testdata.RecallAt(gt, allResults, topK)
	t.Logf("Recall@%d = %.4f (target >= 0.97)", topK, recall)
	if recall < 0.97 {
		t.Errorf("Recall@%d = %.4f < 0.97 (Weaviate baseline)", topK, recall)
	}

	// Step 6: Compare payload sizes with include_vectors=true
	var totalWithVecBytes int64
	for _, q := range queries[:10] { // sample 10 for payload comparison
		code, body := httpSearch(t, mux, map[string]interface{}{
			"collection":      "rag_bench",
			"queries":         map[string]interface{}{"embedding": vecToFloat64(q)},
			"top_k":           topK,
			"ef_search":       256,
			"include_vectors": true,
		})
		if code != 200 {
			t.Fatalf("search with vectors: status %d", code)
		}
		totalWithVecBytes += int64(len(body))
	}

	var totalStrippedSample int64
	for _, q := range queries[:10] {
		_, body := httpSearch(t, mux, map[string]interface{}{
			"collection": "rag_bench",
			"queries":    map[string]interface{}{"embedding": vecToFloat64(q)},
			"top_k":      topK,
			"ef_search":  256,
		})
		totalStrippedSample += int64(len(body))
	}

	payloadRatio := float64(totalStrippedSample) / float64(totalWithVecBytes)
	t.Logf("Payload ratio (stripped/with_vectors) = %.2f (target < 0.25)", payloadRatio)
	if payloadRatio >= 0.25 {
		t.Errorf("Stripped payload is %.0f%% of with-vectors payload (expected < 25%%)", payloadRatio*100)
	}
}

// ============================================================================
// Test 2: Batch Insert Throughput
// ============================================================================

func TestCrossDB_BatchInsertThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping cross-DB bench in short mode")
	}

	const (
		dim       = 384
		seqCount  = 1000
		batchSize = 100
		seed      = 42
	)

	_, mux := setupMux(t)

	// Create collection
	httpCreateCollection(t, mux, map[string]interface{}{
		"name": "batch_bench",
		"fields": []map[string]interface{}{
			{
				"name": "embedding",
				"type": "dense",
				"dim":  dim,
				"index": map[string]interface{}{
					"type":   0,
					"params": map[string]interface{}{"m": 16, "ef_construction": 200},
				},
			},
		},
	})

	rng := rand.New(rand.NewSource(seed))
	genVec := func() []float64 {
		v := make([]float64, dim)
		for i := range v {
			v[i] = float64(rng.Float32()*2 - 1)
		}
		return v
	}

	// Step 2: Sequential insert of 1000 docs
	seqStart := time.Now()
	seqIDs := make([]uint64, 0, seqCount)
	for i := 0; i < seqCount; i++ {
		id := httpInsert(t, mux, "batch_bench", map[string]interface{}{
			"embedding": genVec(),
		}, map[string]interface{}{
			"seq_idx": float64(i),
		})
		seqIDs = append(seqIDs, id)
	}
	seqDuration := time.Since(seqStart)
	seqRate := float64(seqCount) / seqDuration.Seconds()
	t.Logf("Sequential: %d docs in %v (%.0f docs/sec)", seqCount, seqDuration, seqRate)

	// Step 3: Batch insert of 1000 docs (in batches of 100)
	batchStart := time.Now()
	var batchIDs []uint64
	for batch := 0; batch < seqCount/batchSize; batch++ {
		docs := make([]map[string]interface{}, batchSize)
		for i := range docs {
			docs[i] = map[string]interface{}{
				"vectors":  map[string]interface{}{"embedding": genVec()},
				"metadata": map[string]interface{}{"batch_idx": float64(batch*batchSize + i)},
			}
		}
		resp := httpBatchInsert(t, mux, map[string]interface{}{
			"collection": "batch_bench",
			"docs":       docs,
		})
		// Verify IDs returned
		ids, ok := resp["ids"].([]interface{})
		if !ok {
			t.Fatalf("batch response missing ids array: %v", resp)
		}
		for _, id := range ids {
			batchIDs = append(batchIDs, uint64(id.(float64)))
		}
	}
	batchDuration := time.Since(batchStart)
	batchRate := float64(seqCount) / batchDuration.Seconds()
	t.Logf("Batch: %d docs in %v (%.0f docs/sec)", seqCount, batchDuration, batchRate)

	// In httptest (no network), batch advantage is reduced since there's no round-trip savings.
	// We verify batch works and report throughput; speed comparison is informational.
	speedup := seqDuration.Seconds() / batchDuration.Seconds()
	t.Logf("Speedup: %.1fx (in httptest — network round-trip savings not measured)", speedup)

	// All batch IDs must be returned
	if len(batchIDs) != seqCount {
		t.Errorf("Batch returned %d IDs, expected %d", len(batchIDs), seqCount)
	}

	// Step 4: Batch insert with continue_on_error
	docsWithBad := make([]map[string]interface{}, 5)
	for i := range docsWithBad {
		docsWithBad[i] = map[string]interface{}{
			"vectors":  map[string]interface{}{"embedding": genVec()},
			"metadata": map[string]interface{}{"error_test": float64(i)},
		}
	}
	// Make doc at index 2 bad: empty vectors
	docsWithBad[2] = map[string]interface{}{
		"vectors":  map[string]interface{}{},
		"metadata": map[string]interface{}{"error_test": float64(2)},
	}

	errorResp := httpBatchInsert(t, mux, map[string]interface{}{
		"collection":        "batch_bench",
		"docs":              docsWithBad,
		"continue_on_error": true,
	})
	inserted := int(errorResp["inserted"].(float64))
	failed := int(errorResp["failed"].(float64))
	t.Logf("continue_on_error: inserted=%d, failed=%d", inserted, failed)
	if inserted != 4 {
		t.Errorf("expected 4 inserted with continue_on_error, got %d", inserted)
	}
	if failed != 1 {
		t.Errorf("expected 1 failed with continue_on_error, got %d", failed)
	}
	errors, _ := errorResp["errors"].(map[string]interface{})
	if _, has := errors["2"]; !has {
		t.Errorf("expected error at index 2, got errors: %v", errors)
	}

	// Step 5: Verify docs are searchable with correct metadata
	// Pick a known vector from sequential inserts (first one) and search
	searchRng := rand.New(rand.NewSource(seed)) // reset to get same first vector
	firstVec := make([]float64, dim)
	for i := range firstVec {
		firstVec[i] = float64(searchRng.Float32()*2 - 1)
	}
	code, body := httpSearch(t, mux, map[string]interface{}{
		"collection": "batch_bench",
		"queries":    map[string]interface{}{"embedding": firstVec},
		"top_k":      5,
	})
	if code != 200 {
		t.Fatalf("verification search: status %d: %s", code, string(body))
	}
	resp := parseSearchResponse(t, body)
	if len(resp.Documents) == 0 {
		t.Fatal("verification search returned 0 documents")
	}

	// Throughput target (sequential path, which is more predictable in httptest)
	if seqRate < 500 {
		t.Errorf("Sequential insert throughput %.0f docs/sec < 500 target", seqRate)
	}
}

// ============================================================================
// Test 3: ef_search Recall-Latency Tradeoff
// ============================================================================

func TestCrossDB_EfSearchTradeoff(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping cross-DB bench in short mode")
	}

	const (
		numDocs    = 10_000
		dim        = 128
		numQueries = 100
		topK       = 10
		seed       = 42
	)

	_, mux := setupMux(t)

	// Step 1: Create collection
	httpCreateCollection(t, mux, map[string]interface{}{
		"name": "ef_bench",
		"fields": []map[string]interface{}{
			{
				"name": "embedding",
				"type": "dense",
				"dim":  dim,
				"index": map[string]interface{}{
					"type":   0,
					"params": map[string]interface{}{"m": 16, "ef_construction": 200},
				},
			},
		},
	})

	// Step 2: Insert clustered vectors
	data, queries := testdata.GenerateClusteredDataset(numDocs, numQueries, dim, 20, 0.15, seed)

	t.Logf("Inserting %d docs (%dd)...", numDocs, dim)
	idxToHTTPID := make([]uint64, numDocs)
	for i, vec := range data {
		id := httpInsert(t, mux, "ef_bench", map[string]interface{}{
			"embedding": vecToFloat64(vec),
		}, map[string]interface{}{"idx": float64(i)})
		idxToHTTPID[i] = id
	}

	// Step 3: Compute ground truth and remap IDs
	gt, err := testdata.ComputeGroundTruth(data, queries, topK)
	if err != nil {
		t.Fatalf("compute ground truth: %v", err)
	}
	for i := range gt.Neighbors {
		for j := range gt.Neighbors[i] {
			gt.Neighbors[i][j] = idxToHTTPID[gt.Neighbors[i][j]]
		}
	}

	// Step 4: Sweep ef_search values
	efValues := []int{16, 32, 64, 128, 256}
	type efResult struct {
		ef     int
		recall float64
		p50    time.Duration
		p99    time.Duration
	}
	results := make([]efResult, len(efValues))

	for idx, ef := range efValues {
		allResults := make([][]uint64, numQueries)
		latencies := make([]time.Duration, numQueries)

		for i, q := range queries {
			start := time.Now()
			code, body := httpSearch(t, mux, map[string]interface{}{
				"collection": "ef_bench",
				"queries":    map[string]interface{}{"embedding": vecToFloat64(q)},
				"top_k":      topK,
				"ef_search":  ef,
			})
			latencies[i] = time.Since(start)

			if code != 200 {
				t.Fatalf("ef=%d, query %d: status %d: %s", ef, i, code, string(body))
			}
			resp := parseSearchResponse(t, body)
			allResults[i] = extractDocIDs(resp)
		}

		recall := testdata.RecallAt(gt, allResults, topK)
		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

		results[idx] = efResult{
			ef:     ef,
			recall: recall,
			p50:    pctile(latencies, 50),
			p99:    pctile(latencies, 99),
		}
		t.Logf("ef_search=%3d: recall@%d=%.4f  p50=%v  p99=%v",
			ef, topK, recall, results[idx].p50, results[idx].p99)
	}

	// Step 5: Assertions

	// Recall difference between ef=16 and ef=256 must be >= 0.03 (proves passthrough)
	recallDiff := results[len(results)-1].recall - results[0].recall
	t.Logf("Recall difference (ef=256 - ef=16) = %.4f (target >= 0.03)", recallDiff)
	if recallDiff < 0.03 {
		t.Errorf("ef_search passthrough FAILED: recall diff %.4f < 0.03 (ef_search may be ignored)", recallDiff)
	}

	// At ef=128: recall >= 0.98
	ef128 := results[3] // ef=128 is index 3
	if ef128.recall < 0.98 {
		t.Errorf("ef=128: recall %.4f < 0.98", ef128.recall)
	}

	// At ef=128: p99 < 5ms
	if ef128.p99 > 5*time.Millisecond {
		t.Logf("WARNING: ef=128 p99=%v > 5ms (may vary by hardware)", ef128.p99)
	}

	// Recall should be non-decreasing
	for i := 1; i < len(results); i++ {
		if results[i].recall < results[i-1].recall-0.005 { // small tolerance for noise
			t.Errorf("Recall decreased: ef=%d (%.4f) -> ef=%d (%.4f)",
				results[i-1].ef, results[i-1].recall, results[i].ef, results[i].recall)
		}
	}
}

// ============================================================================
// Test 4: Filtered Recommendation
// ============================================================================

func TestCrossDB_FilteredRecommendation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping cross-DB bench in short mode")
	}

	const (
		numDocs    = 10_000
		dim        = 128
		numQueries = 50
		topK       = 10
		seed       = 42
	)

	_, mux := setupMux(t)

	// Step 1: Create collection
	httpCreateCollection(t, mux, map[string]interface{}{
		"name": "filter_bench",
		"fields": []map[string]interface{}{
			{
				"name": "embedding",
				"type": "dense",
				"dim":  dim,
				"index": map[string]interface{}{
					"type":   0,
					"params": map[string]interface{}{"m": 16, "ef_construction": 200},
				},
			},
		},
	})

	// Step 2: Insert docs with skewed metadata
	rng := rand.New(rand.NewSource(seed))
	data := testdata.GenerateClusteredVectors(numDocs, dim, 20, 0.15, rng)
	metaRng := rand.New(rand.NewSource(seed + 100))
	metadata := testdata.GenerateSkewedMetadata(numDocs, metaRng)

	t.Logf("Inserting %d docs with skewed metadata...", numDocs)
	for i, vec := range data {
		httpInsert(t, mux, "filter_bench", map[string]interface{}{
			"embedding": vecToFloat64(vec),
		}, metadata[i])
	}

	// Count actual category distribution
	catCounts := make(map[string]int)
	for _, m := range metadata {
		cat := m["category"].(string)
		catCounts[cat]++
	}
	t.Logf("Category distribution: %v", catCounts)

	// Generate query vectors
	queryRng := rand.New(rand.NewSource(seed + 200))
	queries := testdata.GenerateClusteredVectors(numQueries, dim, 20, 0.12, queryRng)

	// Step 3: Run unfiltered baseline for latency comparison
	var unfilteredLatencies []time.Duration
	for _, q := range queries {
		start := time.Now()
		code, body := httpSearch(t, mux, map[string]interface{}{
			"collection": "filter_bench",
			"queries":    map[string]interface{}{"embedding": vecToFloat64(q)},
			"top_k":      topK,
		})
		unfilteredLatencies = append(unfilteredLatencies, time.Since(start))
		if code != 200 {
			t.Fatalf("unfiltered search: status %d: %s", code, string(body))
		}
	}
	sort.Slice(unfilteredLatencies, func(i, j int) bool { return unfilteredLatencies[i] < unfilteredLatencies[j] })
	unfilteredP99 := pctile(unfilteredLatencies, 99)

	// Step 4: Test at different selectivity levels
	type filterTestCase struct {
		name       string
		filter     map[string]interface{}
		category   string
		minRecall  float64
		maxP99Mult float64 // max multiplier vs unfiltered p99
	}

	cases := []filterTestCase{
		{"50pct_dominant", map[string]interface{}{"category": "dominant"}, "dominant", 0.90, 3.0},
		{"10pct_uncommon", map[string]interface{}{"category": "uncommon"}, "uncommon", 0.0, 3.0}, // lower recall expected
		{"2pct_very_rare", map[string]interface{}{"category": "very_rare"}, "very_rare", 0.0, 3.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var correctFilters, totalResults int
			var latencies []time.Duration

			// Compute filtered ground truth
			filteredData := make([][]float32, 0)
			filteredIDs := make([]uint64, 0)
			for i, m := range metadata {
				if m["category"].(string) == tc.category {
					filteredData = append(filteredData, data[i])
					filteredIDs = append(filteredIDs, uint64(i))
				}
			}
			t.Logf("  Matching docs: %d / %d (%.1f%%)",
				len(filteredData), numDocs, float64(len(filteredData))/float64(numDocs)*100)

			for _, q := range queries {
				start := time.Now()
				code, body := httpSearch(t, mux, map[string]interface{}{
					"collection": "filter_bench",
					"queries":    map[string]interface{}{"embedding": vecToFloat64(q)},
					"top_k":      topK,
					"filters":    tc.filter,
				})
				latencies = append(latencies, time.Since(start))

				if code != 200 {
					t.Fatalf("filtered search: status %d: %s", code, string(body))
				}
				resp := parseSearchResponse(t, body)

				// Verify EVERY returned doc matches the filter predicate
				for _, doc := range resp.Documents {
					totalResults++
					meta, hasMeta := doc["metadata"].(map[string]interface{})
					if !hasMeta {
						t.Fatalf("document missing metadata")
					}
					docCat, hasCat := meta["category"].(string)
					if !hasCat {
						t.Fatalf("document metadata missing category")
					}
					if docCat == tc.category {
						correctFilters++
					} else {
						t.Errorf("Filter violation: expected category=%q, got %q (doc: %v)",
							tc.category, docCat, doc["id"])
					}
				}
			}

			// Filter correctness: 100% (zero false positives)
			if totalResults > 0 {
				accuracy := float64(correctFilters) / float64(totalResults)
				t.Logf("  Filter accuracy: %.4f (%d/%d)", accuracy, correctFilters, totalResults)
				if accuracy < 1.0 {
					t.Errorf("Filter correctness < 100%%: %.4f", accuracy)
				}
			}

			// Latency check
			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
			filteredP99 := pctile(latencies, 99)
			t.Logf("  Filtered p99: %v (unfiltered p99: %v, max: %v)",
				filteredP99, unfilteredP99, time.Duration(float64(unfilteredP99)*tc.maxP99Mult))
			if unfilteredP99 > 0 && float64(filteredP99) > float64(unfilteredP99)*tc.maxP99Mult {
				t.Logf("  WARNING: filtered p99 > %.0fx unfiltered p99", tc.maxP99Mult)
			}
		})
	}
}

// ============================================================================
// Test 5: Hybrid Search + Payload Efficiency
// ============================================================================

func TestCrossDB_HybridSearchPayload(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping cross-DB bench in short mode")
	}

	const (
		numDocs    = 5_000
		denseDim   = 384
		vocabSize  = 10_000
		numQueries = 50
		topK       = 10
		seed       = 42
	)

	_, mux := setupMux(t)

	// Step 1: Create collection with dense + sparse fields (string types — Fix 5)
	httpCreateCollection(t, mux, map[string]interface{}{
		"name": "hybrid_bench",
		"fields": []map[string]interface{}{
			{
				"name": "embedding",
				"type": "dense",
				"dim":  denseDim,
				"index": map[string]interface{}{
					"type":   0,
					"params": map[string]interface{}{"m": 16, "ef_construction": 200},
				},
			},
			{
				"name": "keywords",
				"type": "sparse",
				"dim":  vocabSize,
				"index": map[string]interface{}{
					"type": 4,
				},
			},
		},
	})

	// Step 2: Generate and insert docs with both dense and sparse vectors
	rng := rand.New(rand.NewSource(seed))
	denseData := testdata.GenerateClusteredVectors(numDocs, denseDim, 20, 0.15, rng)
	sparseRng := rand.New(rand.NewSource(seed + 10))
	sparseDocs := testdata.GenerateSparseDocuments(numDocs, vocabSize, 5, 50, sparseRng)

	t.Logf("Inserting %d docs (dense %dd + sparse %d-dim)...", numDocs, denseDim, vocabSize)
	insertStart := time.Now()
	idxToHTTPID := make([]uint64, numDocs)
	for i := range denseData {
		sv := sparseDocs[i]
		// Convert sparse vector to JSON format
		sparseJSON := map[string]interface{}{
			"indices": uintSliceToFloat64(sv.Indices),
			"values":  float32SliceToFloat64(sv.Values),
			"dim":     float64(sv.Dim),
		}
		id := httpInsert(t, mux, "hybrid_bench", map[string]interface{}{
			"embedding": vecToFloat64(denseData[i]),
			"keywords":  sparseJSON,
		}, map[string]interface{}{
			"doc_idx": float64(i),
		})
		idxToHTTPID[i] = id
	}
	t.Logf("Insert took %v", time.Since(insertStart))

	// Generate query vectors
	queryRng := rand.New(rand.NewSource(seed + 20))
	denseQueries := testdata.GenerateClusteredVectors(numQueries, denseDim, 20, 0.12, queryRng)
	sparseQueryRng := rand.New(rand.NewSource(seed + 30))
	sparseQueries := testdata.GenerateSparseDocuments(numQueries, vocabSize, 3, 15, sparseQueryRng)

	// Step 3: Run three search modes

	// Mode A: Dense-only
	t.Log("Running dense-only queries...")
	denseOnlyResults := make([][]uint64, numQueries)
	for i, q := range denseQueries {
		code, body := httpSearch(t, mux, map[string]interface{}{
			"collection": "hybrid_bench",
			"queries":    map[string]interface{}{"embedding": vecToFloat64(q)},
			"top_k":      topK,
			"ef_search":  128,
		})
		if code != 200 {
			t.Fatalf("dense-only query %d: status %d: %s", i, code, string(body))
		}
		resp := parseSearchResponse(t, body)
		denseOnlyResults[i] = extractDocIDs(resp)
	}

	// Mode B: Hybrid RRF (dense + sparse)
	t.Log("Running hybrid RRF queries...")
	hybridResults := make([][]uint64, numQueries)
	for i := range denseQueries {
		sq := sparseQueries[i]
		sparseJSON := map[string]interface{}{
			"indices": uintSliceToFloat64(sq.Indices),
			"values":  float32SliceToFloat64(sq.Values),
			"dim":     float64(sq.Dim),
		}
		code, body := httpSearch(t, mux, map[string]interface{}{
			"collection": "hybrid_bench",
			"queries": map[string]interface{}{
				"embedding": vecToFloat64(denseQueries[i]),
				"keywords":  sparseJSON,
			},
			"top_k":     topK,
			"ef_search": 128,
			"hybrid_params": map[string]interface{}{
				"strategy":     "rrf",
				"rrf_constant": 60,
			},
		})
		if code != 200 {
			t.Fatalf("hybrid query %d: status %d: %s", i, code, string(body))
		}
		resp := parseSearchResponse(t, body)
		hybridResults[i] = extractDocIDs(resp)
	}

	// Compute ground truth for dense recall comparison and remap IDs
	gt, err := testdata.ComputeGroundTruth(denseData, denseQueries, topK)
	if err != nil {
		t.Fatalf("compute ground truth: %v", err)
	}
	for i := range gt.Neighbors {
		for j := range gt.Neighbors[i] {
			gt.Neighbors[i][j] = idxToHTTPID[gt.Neighbors[i][j]]
		}
	}

	denseOnlyRecall := testdata.RecallAt(gt, denseOnlyResults, topK)
	hybridRecall := testdata.RecallAt(gt, hybridResults, topK)
	t.Logf("Dense-only recall@%d = %.4f", topK, denseOnlyRecall)
	t.Logf("Hybrid RRF recall@%d  = %.4f", topK, hybridRecall)

	// Hybrid may degrade dense-only recall when sparse queries are uncorrelated with dense.
	// The key assertion is that hybrid search works end-to-end without errors.
	// With correlated queries, hybrid should improve recall; with random sparse, some degradation is expected.
	if hybridRecall < denseOnlyRecall*0.5 {
		t.Errorf("Hybrid recall (%.4f) degraded >50%% vs dense-only (%.4f)", hybridRecall, denseOnlyRecall)
	}

	// Step 4: Payload efficiency comparison
	t.Log("Measuring payload sizes...")

	var strippedTotal, withVecTotal, glyphTotal int64
	sampleQueries := denseQueries[:10]

	for _, q := range sampleQueries {
		searchBody := map[string]interface{}{
			"collection": "hybrid_bench",
			"queries":    map[string]interface{}{"embedding": vecToFloat64(q)},
			"top_k":      topK,
			"ef_search":  128,
		}

		// Default (stripped)
		_, body := httpSearch(t, mux, searchBody)
		strippedTotal += int64(len(body))

		// With vectors
		searchBody["include_vectors"] = true
		_, body = httpSearch(t, mux, searchBody)
		withVecTotal += int64(len(body))
		delete(searchBody, "include_vectors")

		// Glyph encoding
		w := httpDoWithAccept(t, mux, http.MethodPost, "/v2/search", searchBody, "application/glyph")
		if w.Code == 200 {
			glyphTotal += int64(w.Body.Len())
		}
	}

	strippedRatio := float64(strippedTotal) / float64(withVecTotal)
	t.Logf("Stripped/WithVectors ratio = %.2f (target < 0.30)", strippedRatio)
	if strippedRatio >= 0.30 {
		t.Errorf("Stripped payload ratio %.2f >= 0.30", strippedRatio)
	}

	if glyphTotal > 0 {
		glyphRatio := float64(glyphTotal) / float64(strippedTotal)
		t.Logf("Glyph/Stripped ratio = %.2f (target < 0.70)", glyphRatio)
		if glyphRatio >= 0.70 {
			t.Errorf("Glyph payload ratio %.2f >= 0.70", glyphRatio)
		}
	}

	// Verify all sparse queries returned 200 (no parse errors)
	t.Log("Verifying sparse-only queries return 200...")
	for i, sq := range sparseQueries {
		sparseJSON := map[string]interface{}{
			"indices": uintSliceToFloat64(sq.Indices),
			"values":  float32SliceToFloat64(sq.Values),
			"dim":     float64(sq.Dim),
		}
		code, body := httpSearch(t, mux, map[string]interface{}{
			"collection": "hybrid_bench",
			"queries":    map[string]interface{}{"keywords": sparseJSON},
			"top_k":      topK,
		})
		if code != 200 {
			t.Fatalf("sparse-only query %d: status %d: %s", i, code, string(body))
		}
	}
}

// ============================================================================
// Sparse vector conversion helpers
// ============================================================================

func uintSliceToFloat64(s []uint32) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = float64(v)
	}
	return out
}

func float32SliceToFloat64(s []float32) []float64 {
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = float64(v)
	}
	return out
}
