package competitive

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/index"
)

// BenchmarkScenario describes a parameterized benchmark configuration.
type BenchmarkScenario struct {
	Name      string
	IndexType string
	Dimension int
	Scale     int
	Config    map[string]interface{}
	SearchParams index.SearchParams
	K         int
	NumQueries int
}

// RunScenario executes a benchmark scenario: builds index, runs searches, collects metrics.
// Returns a BenchmarkResult with latency, throughput, and memory stats.
func RunScenario(b *testing.B, scenario BenchmarkScenario, vectors, queries [][]float32) BenchmarkResult {
	b.Helper()

	ctx := context.Background()
	dim := scenario.Dimension
	k := scenario.K
	if k == 0 {
		k = 10
	}

	// Create index
	idx, err := index.Create(scenario.IndexType, dim, scenario.Config)
	if err != nil {
		b.Fatalf("creating %s index: %v", scenario.IndexType, err)
	}

	// Insert vectors
	for i, v := range vectors {
		if err := idx.Add(ctx, uint64(i), v); err != nil {
			b.Fatalf("inserting vector %d: %v", i, err)
		}
	}

	// Force GC and measure baseline memory
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	// Warm up
	warmup := 10
	if warmup > len(queries) {
		warmup = len(queries)
	}
	for i := 0; i < warmup; i++ {
		_, _ = idx.Search(ctx, queries[i%len(queries)], k, scenario.SearchParams)
	}

	// Benchmark search
	numQueries := len(queries)
	latencies := make([]time.Duration, 0, b.N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%numQueries]
		start := time.Now()
		_, err := idx.Search(ctx, q, k, scenario.SearchParams)
		elapsed := time.Since(start)
		if err != nil {
			b.Fatalf("search error: %v", err)
		}
		latencies = append(latencies, elapsed)
	}
	b.StopTimer()

	// Measure memory after
	runtime.GC()
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	memUsedMB := float64(memAfter.Alloc-memBefore.Alloc) / (1024 * 1024)
	if memUsedMB < 0 {
		// GC reclaimed some memory, use index stats instead
		stats := idx.Stats()
		memUsedMB = float64(stats.MemoryUsed) / (1024 * 1024)
	}

	// Compute stats
	mean, p50, p95, p99, max := LatencyStats(latencies)
	totalTime := MeanDuration(latencies) * time.Duration(len(latencies))
	qps := float64(len(latencies)) / totalTime.Seconds()

	result := BenchmarkResult{
		Scenario:     scenario.Name,
		Index:        scenario.IndexType,
		Dimension:    dim,
		Scale:        scenario.Scale,
		Timestamp:    time.Now(),
		MeanLatencyUs: mean,
		P50LatencyUs: p50,
		P95LatencyUs: p95,
		P99LatencyUs: p99,
		MaxLatencyUs: max,
		QPS:          qps,
		MemoryMB:     memUsedMB,
	}

	if scenario.Scale > 0 {
		result.BytesPerVec = (memUsedMB * 1024 * 1024) / float64(scenario.Scale)
	}

	// Report to Go benchmark
	b.ReportMetric(qps, "qps")
	b.ReportMetric(p50, "p50_us")
	b.ReportMetric(p99, "p99_us")
	b.ReportMetric(memUsedMB, "mem_mb")

	return result
}

// MeasureInsertThroughput measures insert performance.
func MeasureInsertThroughput(b *testing.B, indexType string, dim int, config map[string]interface{}, vectors [][]float32) BenchmarkResult {
	b.Helper()
	ctx := context.Background()

	b.ResetTimer()
	idx, err := index.Create(indexType, dim, config)
	if err != nil {
		b.Fatalf("creating index: %v", err)
	}

	start := time.Now()
	for i := 0; i < b.N; i++ {
		v := vectors[i%len(vectors)]
		if err := idx.Add(ctx, uint64(i), v); err != nil {
			b.Fatalf("insert error at %d: %v", i, err)
		}
	}
	elapsed := time.Since(start)
	b.StopTimer()

	qps := float64(b.N) / elapsed.Seconds()
	b.ReportMetric(qps, "insert_qps")

	return BenchmarkResult{
		Scenario:  "insert",
		Index:     indexType,
		Dimension: dim,
		Scale:     b.N,
		InsertQPS: qps,
		Timestamp: time.Now(),
	}
}

// MeasureRecall runs queries and computes recall against ground truth.
func MeasureRecall(idx index.Index, queries [][]float32, groundTruth [][]uint64, k int, params index.SearchParams) (recall1, recall10, recall100 float64, err error) {
	ctx := context.Background()
	results := make([][]uint64, len(queries))

	maxK := 100
	if k > maxK {
		maxK = k
	}

	for i, q := range queries {
		res, searchErr := idx.Search(ctx, q, maxK, params)
		if searchErr != nil {
			return 0, 0, 0, searchErr
		}
		ids := make([]uint64, len(res))
		for j, r := range res {
			ids[j] = r.ID
		}
		results[i] = ids
	}

	// Compute recall at different k values
	recall1 = computeRecall(results, groundTruth, 1)
	recall10 = computeRecall(results, groundTruth, 10)
	recall100 = computeRecall(results, groundTruth, 100)
	return
}

func computeRecall(results, groundTruth [][]uint64, k int) float64 {
	if len(results) == 0 {
		return 0
	}
	totalRecall := 0.0
	n := len(results)
	if len(groundTruth) < n {
		n = len(groundTruth)
	}
	for i := 0; i < n; i++ {
		trueSet := make(map[uint64]bool)
		limit := k
		if limit > len(groundTruth[i]) {
			limit = len(groundTruth[i])
		}
		for _, id := range groundTruth[i][:limit] {
			trueSet[id] = true
		}
		found := 0
		resLimit := k
		if resLimit > len(results[i]) {
			resLimit = len(results[i])
		}
		for _, id := range results[i][:resLimit] {
			if trueSet[id] {
				found++
			}
		}
		if limit > 0 {
			totalRecall += float64(found) / float64(limit)
		}
	}
	return totalRecall / float64(n)
}

// MeasureMemory measures memory footprint of an index at a given scale.
func MeasureMemory(indexType string, dim int, config map[string]interface{}, vectors [][]float32) (memMB float64, bytesPerVec float64, err error) {
	ctx := context.Background()

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	idx, err := index.Create(indexType, dim, config)
	if err != nil {
		return 0, 0, err
	}

	for i, v := range vectors {
		if err := idx.Add(ctx, uint64(i), v); err != nil {
			return 0, 0, err
		}
	}

	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	memBytes := int64(after.Alloc) - int64(before.Alloc)
	if memBytes < 0 {
		stats := idx.Stats()
		memBytes = stats.MemoryUsed
	}
	memMB = float64(memBytes) / (1024 * 1024)
	if len(vectors) > 0 {
		bytesPerVec = float64(memBytes) / float64(len(vectors))
	}
	return
}
