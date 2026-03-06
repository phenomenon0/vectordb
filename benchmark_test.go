package main

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/index"
)

// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================

const (
	// Scale levels for benchmarks
	Scale1K   = 1_000
	Scale10K  = 10_000
	Scale100K = 100_000
	Scale1M   = 1_000_000
	Scale10M  = 10_000_000

	// Default benchmark parameters
	DefaultDim    = 384
	DefaultK      = 10
	DefaultEF     = 100
	DefaultM      = 16
	DefaultEfCons = 200
)

// BenchmarkConfig holds configuration for benchmark runs
type BenchmarkConfig struct {
	NumVectors     int
	Dimension      int
	K              int
	EF             int
	M              int
	EfConstruction int
	NumQueries     int
	Parallel       int
}

func DefaultBenchmarkConfig(scale int) BenchmarkConfig {
	return BenchmarkConfig{
		NumVectors:     scale,
		Dimension:      DefaultDim,
		K:              DefaultK,
		EF:             DefaultEF,
		M:              DefaultM,
		EfConstruction: DefaultEfCons,
		NumQueries:     1000,
		Parallel:       runtime.NumCPU(),
	}
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// generateRandomVectors creates random float32 vectors
func generateRandomVectors(count, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for i := 0; i < count; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()*2 - 1 // [-1, 1]
		}
		vectors[i] = vec
	}
	return vectors
}

// generateRandomVector creates a single random vector
func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vec[i] = rand.Float32()*2 - 1
	}
	return vec
}

// memStats returns current memory usage in MB
func memStats() (allocMB, totalMB float64) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	allocMB = float64(m.Alloc) / 1024 / 1024
	totalMB = float64(m.TotalAlloc) / 1024 / 1024
	return
}

// =============================================================================
// INSERTION BENCHMARKS
// =============================================================================

func BenchmarkInsert_1K(b *testing.B) {
	benchmarkInsert(b, Scale1K)
}

func BenchmarkInsert_10K(b *testing.B) {
	benchmarkInsert(b, Scale10K)
}

func BenchmarkInsert_100K(b *testing.B) {
	benchmarkInsert(b, Scale100K)
}

func benchmarkInsert(b *testing.B, scale int) {
	cfg := DefaultBenchmarkConfig(scale)
	vectors := generateRandomVectors(cfg.NumVectors, cfg.Dimension)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		store := newTestStore(cfg.Dimension)
		b.StartTimer()

		for j, vec := range vectors {
			store.Add(vec, fmt.Sprintf("doc-%d", j), "", nil, "default", "default")
		}
	}
}

func BenchmarkInsertParallel_10K(b *testing.B) {
	benchmarkInsertParallel(b, Scale10K)
}

func BenchmarkInsertParallel_100K(b *testing.B) {
	benchmarkInsertParallel(b, Scale100K)
}

func benchmarkInsertParallel(b *testing.B, scale int) {
	cfg := DefaultBenchmarkConfig(scale)
	vectors := generateRandomVectors(cfg.NumVectors, cfg.Dimension)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		store := newTestStore(cfg.Dimension)
		b.StartTimer()

		var wg sync.WaitGroup
		batchSize := cfg.NumVectors / cfg.Parallel

		for p := 0; p < cfg.Parallel; p++ {
			wg.Add(1)
			go func(partition int) {
				defer wg.Done()
				start := partition * batchSize
				end := start + batchSize
				if partition == cfg.Parallel-1 {
					end = cfg.NumVectors
				}
				for j := start; j < end; j++ {
					store.Add(vectors[j], fmt.Sprintf("doc-%d", j), "", nil, "default", "default")
				}
			}(p)
		}
		wg.Wait()
	}
}

// =============================================================================
// SEARCH BENCHMARKS
// =============================================================================

func BenchmarkSearch_1K(b *testing.B) {
	benchmarkSearch(b, Scale1K)
}

func BenchmarkSearch_10K(b *testing.B) {
	benchmarkSearch(b, Scale10K)
}

func BenchmarkSearch_100K(b *testing.B) {
	benchmarkSearch(b, Scale100K)
}

func benchmarkSearch(b *testing.B, scale int) {
	cfg := DefaultBenchmarkConfig(scale)
	store := setupBenchmarkStore(cfg)
	queries := generateRandomVectors(cfg.NumQueries, cfg.Dimension)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		store.SearchANN(query, cfg.K)
	}
}

func BenchmarkSearchParallel_10K(b *testing.B) {
	benchmarkSearchParallel(b, Scale10K)
}

func BenchmarkSearchParallel_100K(b *testing.B) {
	benchmarkSearchParallel(b, Scale100K)
}

func benchmarkSearchParallel(b *testing.B, scale int) {
	cfg := DefaultBenchmarkConfig(scale)
	store := setupBenchmarkStore(cfg)
	queries := generateRandomVectors(cfg.NumQueries, cfg.Dimension)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			query := queries[i%len(queries)]
			store.SearchANN(query, cfg.K)
			i++
		}
	})
}

// =============================================================================
// HYBRID SEARCH BENCHMARKS
// =============================================================================

func BenchmarkHybridSearch_10K(b *testing.B) {
	benchmarkHybridSearch(b, Scale10K)
}

func BenchmarkHybridSearch_100K(b *testing.B) {
	benchmarkHybridSearch(b, Scale100K)
}

func benchmarkHybridSearch(b *testing.B, scale int) {
	cfg := DefaultBenchmarkConfig(scale)
	store := setupBenchmarkStoreWithDocs(cfg)
	queries := []string{
		"vector search optimization",
		"machine learning algorithms",
		"neural network architecture",
		"database performance tuning",
		"distributed systems design",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		qTokens := tokenize(query)
		store.SearchLex(qTokens, cfg.K)
	}
}

// =============================================================================
// MEMORY BENCHMARKS
// =============================================================================

func BenchmarkMemory_10K(b *testing.B) {
	benchmarkMemory(b, Scale10K)
}

func BenchmarkMemory_100K(b *testing.B) {
	benchmarkMemory(b, Scale100K)
}

func BenchmarkMemory_1M(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping 1M memory benchmark in short mode")
	}
	benchmarkMemory(b, Scale1M)
}

func benchmarkMemory(b *testing.B, scale int) {
	cfg := DefaultBenchmarkConfig(scale)

	runtime.GC()
	beforeAlloc, _ := memStats()

	store := newTestStore(cfg.Dimension)
	vectors := generateRandomVectors(cfg.NumVectors, cfg.Dimension)

	for i, vec := range vectors {
		store.Add(vec, fmt.Sprintf("doc-%d content", i), "", nil, "default", "default")
	}

	runtime.GC()
	afterAlloc, _ := memStats()

	bytesPerVector := (afterAlloc - beforeAlloc) * 1024 * 1024 / float64(cfg.NumVectors)
	b.ReportMetric(afterAlloc-beforeAlloc, "MB_total")
	b.ReportMetric(bytesPerVector, "bytes/vector")
	b.ReportMetric(float64(cfg.NumVectors), "vectors")
}

// =============================================================================
// LATENCY DISTRIBUTION BENCHMARKS
// =============================================================================

func BenchmarkSearchLatencyDistribution_100K(b *testing.B) {
	cfg := DefaultBenchmarkConfig(Scale100K)
	store := setupBenchmarkStore(cfg)
	queries := generateRandomVectors(1000, cfg.Dimension)

	latencies := make([]time.Duration, 0, b.N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		start := time.Now()
		store.SearchANN(query, cfg.K)
		latencies = append(latencies, time.Since(start))
	}

	if len(latencies) > 0 {
		// Calculate percentiles
		p50 := percentile(latencies, 0.50)
		p95 := percentile(latencies, 0.95)
		p99 := percentile(latencies, 0.99)

		b.ReportMetric(float64(p50.Microseconds()), "p50_us")
		b.ReportMetric(float64(p95.Microseconds()), "p95_us")
		b.ReportMetric(float64(p99.Microseconds()), "p99_us")
	}
}

// =============================================================================
// THROUGHPUT BENCHMARKS
// =============================================================================

func BenchmarkThroughput_Search_100K(b *testing.B) {
	cfg := DefaultBenchmarkConfig(Scale100K)
	store := setupBenchmarkStore(cfg)
	queries := generateRandomVectors(1000, cfg.Dimension)

	start := time.Now()
	b.ResetTimer()

	var ops int64
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			query := queries[i%len(queries)]
			store.SearchANN(query, cfg.K)
			i++
			ops++
		}
	})

	elapsed := time.Since(start)
	qps := float64(b.N) / elapsed.Seconds()
	b.ReportMetric(qps, "qps")
}

// =============================================================================
// SCALE TEST (10M)
// =============================================================================

func TestScale10M(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 10M scale test in short mode")
	}

	cfg := BenchmarkConfig{
		NumVectors:     Scale10M,
		Dimension:      DefaultDim,
		K:              DefaultK,
		EF:             DefaultEF,
		M:              DefaultM,
		EfConstruction: DefaultEfCons,
		NumQueries:     1000,
		Parallel:       runtime.NumCPU(),
	}

	t.Logf("Starting 10M scale test with dimension=%d", cfg.Dimension)

	// Memory before
	runtime.GC()
	beforeAlloc, _ := memStats()
	t.Logf("Memory before: %.2f MB", beforeAlloc)

	// Create store
	store := newTestStore(cfg.Dimension)

	// Insert in batches
	batchSize := 100000
	insertStart := time.Now()

	for batch := 0; batch < cfg.NumVectors/batchSize; batch++ {
		vectors := generateRandomVectors(batchSize, cfg.Dimension)
		for i, vec := range vectors {
			idx := batch*batchSize + i
			store.Add(vec, fmt.Sprintf("doc-%d", idx), "", nil, "default", "default")
		}
		if (batch+1)%10 == 0 {
			t.Logf("Inserted %d vectors...", (batch+1)*batchSize)
		}
	}

	insertDuration := time.Since(insertStart)
	insertRate := float64(cfg.NumVectors) / insertDuration.Seconds()
	t.Logf("Insert complete: %d vectors in %v (%.0f vec/s)", cfg.NumVectors, insertDuration, insertRate)

	// Memory after insert
	runtime.GC()
	afterAlloc, _ := memStats()
	t.Logf("Memory after: %.2f MB (%.2f bytes/vector)", afterAlloc, (afterAlloc-beforeAlloc)*1024*1024/float64(cfg.NumVectors))

	// Search benchmark
	queries := generateRandomVectors(cfg.NumQueries, cfg.Dimension)
	latencies := make([]time.Duration, cfg.NumQueries)

	searchStart := time.Now()
	for i, query := range queries {
		start := time.Now()
		results := store.SearchANN(query, cfg.K)
		latencies[i] = time.Since(start)
		if len(results) != cfg.K {
			t.Logf("Warning: query %d returned %d results (expected %d)", i, len(results), cfg.K)
		}
	}
	searchDuration := time.Since(searchStart)

	// Calculate percentiles
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)
	qps := float64(cfg.NumQueries) / searchDuration.Seconds()

	t.Logf("Search results:")
	t.Logf("  Total queries: %d", cfg.NumQueries)
	t.Logf("  Total time: %v", searchDuration)
	t.Logf("  QPS: %.2f", qps)
	t.Logf("  P50 latency: %v", p50)
	t.Logf("  P95 latency: %v", p95)
	t.Logf("  P99 latency: %v", p99)

	// Validation
	if p99 > 100*time.Millisecond {
		t.Errorf("P99 latency too high: %v (target: <100ms)", p99)
	}
	if qps < 100 {
		t.Errorf("QPS too low: %.2f (target: >100)", qps)
	}
}

// =============================================================================
// SCALE TEST (50M) - Uses DiskANN for memory efficiency
// =============================================================================

const (
	Scale50M  = 50_000_000
	Scale100M = 100_000_000
)

func TestScale50M(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 50M scale test in short mode")
	}

	// Check available memory - this test is memory-intensive
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	cfg := BenchmarkConfig{
		NumVectors:     Scale50M,
		Dimension:      256, // Smaller dimension for 50M scale
		K:              DefaultK,
		EF:             DefaultEF,
		M:              12, // Lower M for memory efficiency
		EfConstruction: 100,
		NumQueries:     1000,
		Parallel:       runtime.NumCPU(),
	}

	t.Logf("Starting 50M scale test with dimension=%d", cfg.Dimension)
	t.Logf("Expected memory: ~%.0f GB", float64(cfg.NumVectors)*float64(cfg.Dimension)*4/(1024*1024*1024))

	// Memory before
	runtime.GC()
	beforeAlloc, _ := memStats()
	t.Logf("Memory before: %.2f MB", beforeAlloc)

	// Create DiskANN index for large scale
	diskannConfig := map[string]interface{}{
		"memory_limit":    1000000, // Keep 1M vectors in memory
		"max_degree":      64,
		"ef_construction": 100,
		"ef_search":       cfg.EF,
		"metric":          "cosine",
	}

	diskann, err := index.NewDiskANNIndex(cfg.Dimension, diskannConfig)
	if err != nil {
		t.Fatalf("Failed to create DiskANN: %v", err)
	}

	// Insert in large batches with progress tracking
	batchSize := 500000 // 500K per batch
	insertStart := time.Now()
	totalInserted := 0

	for batch := 0; batch < cfg.NumVectors/batchSize; batch++ {
		vectors := generateRandomVectors(batchSize, cfg.Dimension)
		for i, vec := range vectors {
			idx := uint64(batch*batchSize + i)
			if err := diskann.Add(context.Background(), idx, vec); err != nil {
				t.Fatalf("Insert failed at vector %d: %v", idx, err)
			}
		}
		totalInserted += batchSize

		// Report progress every 5M vectors
		if totalInserted%(5*1000000) == 0 {
			elapsed := time.Since(insertStart)
			rate := float64(totalInserted) / elapsed.Seconds()
			currentAlloc, _ := memStats()
			t.Logf("Progress: %dM vectors, rate: %.0f vec/s, memory: %.0f MB",
				totalInserted/1000000, rate, currentAlloc)
		}
	}

	insertDuration := time.Since(insertStart)
	insertRate := float64(cfg.NumVectors) / insertDuration.Seconds()
	t.Logf("Insert complete: %d vectors in %v (%.0f vec/s)", cfg.NumVectors, insertDuration, insertRate)

	// Memory after insert
	runtime.GC()
	afterAlloc, _ := memStats()
	bytesPerVector := (afterAlloc - beforeAlloc) * 1024 * 1024 / float64(cfg.NumVectors)
	t.Logf("Memory after: %.2f MB (%.2f bytes/vector)", afterAlloc, bytesPerVector)

	// Search benchmark
	queries := generateRandomVectors(cfg.NumQueries, cfg.Dimension)
	latencies := make([]time.Duration, cfg.NumQueries)

	searchStart := time.Now()
	for i, query := range queries {
		start := time.Now()
		results, err := diskann.Search(context.Background(), query, cfg.K, nil)
		latencies[i] = time.Since(start)
		if err != nil {
			t.Errorf("Search failed: %v", err)
		}
		if len(results) != cfg.K {
			t.Logf("Warning: query %d returned %d results (expected %d)", i, len(results), cfg.K)
		}
	}
	searchDuration := time.Since(searchStart)

	// Calculate percentiles
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)
	qps := float64(cfg.NumQueries) / searchDuration.Seconds()

	t.Logf("Search results (50M scale):")
	t.Logf("  Total queries: %d", cfg.NumQueries)
	t.Logf("  Total time: %v", searchDuration)
	t.Logf("  QPS: %.2f", qps)
	t.Logf("  P50 latency: %v", p50)
	t.Logf("  P95 latency: %v", p95)
	t.Logf("  P99 latency: %v", p99)

	// Validation - relaxed targets for 50M
	if p99 > 200*time.Millisecond {
		t.Errorf("P99 latency too high: %v (target: <200ms at 50M scale)", p99)
	}
	if qps < 50 {
		t.Errorf("QPS too low: %.2f (target: >50 at 50M scale)", qps)
	}
}

// =============================================================================
// SCALE TEST (100M) - Requires disk-backed storage
// =============================================================================

func TestScale100M(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 100M scale test in short mode")
	}

	cfg := BenchmarkConfig{
		NumVectors:     Scale100M,
		Dimension:      128, // Smaller dimension for 100M scale
		K:              DefaultK,
		EF:             DefaultEF,
		M:              8, // Lower M for memory efficiency
		EfConstruction: 64,
		NumQueries:     500, // Fewer queries for 100M
		Parallel:       runtime.NumCPU(),
	}

	t.Logf("Starting 100M scale test with dimension=%d", cfg.Dimension)
	t.Logf("Expected memory: ~%.0f GB", float64(cfg.NumVectors)*float64(cfg.Dimension)*4/(1024*1024*1024))

	// Memory before
	runtime.GC()
	beforeAlloc, _ := memStats()
	t.Logf("Memory before: %.2f MB", beforeAlloc)

	// Create DiskANN index for large scale
	diskannConfig := map[string]interface{}{
		"memory_limit":    2000000, // Keep 2M vectors in memory
		"max_degree":      48,
		"ef_construction": 64,
		"ef_search":       cfg.EF,
		"metric":          "cosine",
	}

	diskann, err := index.NewDiskANNIndex(cfg.Dimension, diskannConfig)
	if err != nil {
		t.Fatalf("Failed to create DiskANN: %v", err)
	}

	// Insert in large batches with progress tracking
	batchSize := 1000000 // 1M per batch
	insertStart := time.Now()
	totalInserted := 0
	lastReportTime := insertStart

	for batch := 0; batch < cfg.NumVectors/batchSize; batch++ {
		vectors := generateRandomVectors(batchSize, cfg.Dimension)
		for i, vec := range vectors {
			idx := uint64(batch*batchSize + i)
			if err := diskann.Add(context.Background(), idx, vec); err != nil {
				t.Fatalf("Insert failed at vector %d: %v", idx, err)
			}
		}
		totalInserted += batchSize

		// Report progress every 10M vectors
		if totalInserted%(10*1000000) == 0 {
			elapsed := time.Since(insertStart)
			batchElapsed := time.Since(lastReportTime)
			rate := float64(10*1000000) / batchElapsed.Seconds()
			currentAlloc, _ := memStats()
			eta := time.Duration(float64(cfg.NumVectors-totalInserted) / rate * float64(time.Second))
			t.Logf("Progress: %dM vectors, rate: %.0f vec/s, memory: %.0f MB, elapsed: %v, ETA: %v",
				totalInserted/1000000, rate, currentAlloc, elapsed.Round(time.Second), eta.Round(time.Second))
			lastReportTime = time.Now()
		}
	}

	insertDuration := time.Since(insertStart)
	insertRate := float64(cfg.NumVectors) / insertDuration.Seconds()
	t.Logf("Insert complete: %d vectors in %v (%.0f vec/s)", cfg.NumVectors, insertDuration, insertRate)

	// Memory after insert
	runtime.GC()
	afterAlloc, _ := memStats()
	bytesPerVector := (afterAlloc - beforeAlloc) * 1024 * 1024 / float64(cfg.NumVectors)
	t.Logf("Memory after: %.2f MB (%.2f bytes/vector)", afterAlloc, bytesPerVector)

	// Search benchmark
	queries := generateRandomVectors(cfg.NumQueries, cfg.Dimension)
	latencies := make([]time.Duration, cfg.NumQueries)

	t.Logf("Running %d search queries...", cfg.NumQueries)
	searchStart := time.Now()
	for i, query := range queries {
		start := time.Now()
		results, err := diskann.Search(context.Background(), query, cfg.K, nil)
		latencies[i] = time.Since(start)
		if err != nil {
			t.Errorf("Search failed: %v", err)
		}
		if len(results) != cfg.K {
			t.Logf("Warning: query %d returned %d results (expected %d)", i, len(results), cfg.K)
		}
	}
	searchDuration := time.Since(searchStart)

	// Calculate percentiles
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)
	p99 := percentile(latencies, 0.99)
	qps := float64(cfg.NumQueries) / searchDuration.Seconds()

	t.Logf("Search results (100M scale):")
	t.Logf("  Total queries: %d", cfg.NumQueries)
	t.Logf("  Total time: %v", searchDuration)
	t.Logf("  QPS: %.2f", qps)
	t.Logf("  P50 latency: %v", p50)
	t.Logf("  P95 latency: %v", p95)
	t.Logf("  P99 latency: %v", p99)
	t.Logf("  Memory: %.0f MB (%.2f bytes/vector)", afterAlloc, bytesPerVector)

	// Validation - relaxed targets for 100M
	if p99 > 500*time.Millisecond {
		t.Errorf("P99 latency too high: %v (target: <500ms at 100M scale)", p99)
	}
	if qps < 20 {
		t.Errorf("QPS too low: %.2f (target: >20 at 100M scale)", qps)
	}

	// Summary
	t.Logf("\n=== 100M SCALE VALIDATION SUMMARY ===")
	t.Logf("Vectors:        %d (%dM)", cfg.NumVectors, cfg.NumVectors/1000000)
	t.Logf("Dimension:      %d", cfg.Dimension)
	t.Logf("Insert rate:    %.0f vec/s", insertRate)
	t.Logf("Memory:         %.0f MB (%.2f bytes/vector)", afterAlloc, bytesPerVector)
	t.Logf("QPS:            %.2f", qps)
	t.Logf("P50:            %v", p50)
	t.Logf("P95:            %v", p95)
	t.Logf("P99:            %v", p99)
	t.Logf("=====================================")
}

// =============================================================================
// MEMORY-CONSTRAINED SCALE TEST
// Uses quantization for limited memory environments
// =============================================================================

func TestScaleLargeWithQuantization(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large scale quantization test in short mode")
	}

	// This test validates memory efficiency using product quantization compression
	// Can run 10M vectors in ~4GB instead of ~15GB

	cfg := BenchmarkConfig{
		NumVectors: Scale10M,
		Dimension:  384,
		K:          DefaultK,
		EF:         DefaultEF,
		NumQueries: 500,
		Parallel:   runtime.NumCPU(),
	}

	t.Logf("Starting 10M quantized scale test")

	runtime.GC()
	beforeAlloc, _ := memStats()
	t.Logf("Memory before: %.2f MB", beforeAlloc)

	// Create ProductQuantizer for compression
	numSubVec := 48 // 8 dimensions per subvector
	ksub := 256     // 1 byte per subvector
	pq, err := index.NewProductQuantizer(cfg.Dimension, numSubVec, ksub)
	if err != nil {
		t.Fatalf("Failed to create PQ: %v", err)
	}

	// Generate training data - PQ requires flattened vectors
	trainSize := 100000
	trainVectors := generateRandomVectors(trainSize, cfg.Dimension)
	flattenedTrain := make([]float32, 0, trainSize*cfg.Dimension)
	for _, v := range trainVectors {
		flattenedTrain = append(flattenedTrain, v...)
	}

	t.Log("Training product quantizer...")
	trainStart := time.Now()
	if err := pq.Train(flattenedTrain, 10); err != nil {
		t.Fatalf("PQ training failed: %v", err)
	}
	t.Logf("Training complete: %v", time.Since(trainStart))

	// Store compressed codes
	codes := make([][]byte, 0, cfg.NumVectors)

	// Insert vectors
	batchSize := 100000
	insertStart := time.Now()

	for batch := 0; batch < cfg.NumVectors/batchSize; batch++ {
		vectors := generateRandomVectors(batchSize, cfg.Dimension)
		for _, vec := range vectors {
			// Quantize returns ([]byte, error)
			code, err := pq.Quantize(vec)
			if err != nil {
				t.Fatalf("Quantize failed: %v", err)
			}
			codes = append(codes, code)
		}
		if (batch+1)%10 == 0 {
			currentAlloc, _ := memStats()
			t.Logf("Inserted %dM vectors, memory: %.0f MB", (batch+1)*batchSize/1000000, currentAlloc)
		}
	}

	insertDuration := time.Since(insertStart)

	runtime.GC()
	afterAlloc, _ := memStats()
	bytesPerVector := (afterAlloc - beforeAlloc) * 1024 * 1024 / float64(cfg.NumVectors)
	compressionRatio := float64(cfg.Dimension*4) / bytesPerVector

	t.Logf("Insert complete: %v (%.0f vec/s)", insertDuration, float64(cfg.NumVectors)/insertDuration.Seconds())
	t.Logf("Memory: %.0f MB (%.2f bytes/vector, %.1fx compression)", afterAlloc, bytesPerVector, compressionRatio)

	// Search using quantization and dequantization
	queries := generateRandomVectors(cfg.NumQueries, cfg.Dimension)
	latencies := make([]time.Duration, cfg.NumQueries)

	searchStart := time.Now()
	for i, query := range queries {
		start := time.Now()
		// Quantize and dequantize to measure codec overhead
		code, _ := pq.Quantize(query)
		_, _ = pq.Dequantize(code)
		latencies[i] = time.Since(start)
	}
	searchDuration := time.Since(searchStart)

	p50 := percentile(latencies, 0.50)
	p99 := percentile(latencies, 0.99)
	qps := float64(cfg.NumQueries) / searchDuration.Seconds()

	t.Logf("Search (quantization round-trip): QPS=%.2f, P50=%v, P99=%v", qps, p50, p99)
	t.Logf("Codes stored: %d, code size: %d bytes", len(codes), len(codes[0]))

	// Validation
	if compressionRatio < 3 {
		t.Errorf("Compression ratio too low: %.1fx (target: >3x)", compressionRatio)
	}
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

func newTestStore(dim int) *VectorStore {
	// Create default index
	defaultIdx, _ := index.NewHNSWIndex(dim, map[string]interface{}{
		"m":         16,
		"ml":        0.25,
		"ef_search": 64,
	})
	return &VectorStore{
		Data:     make([]float32, 0, 1000*dim),
		Dim:      dim,
		Docs:     make([]string, 0, 1000),
		IDs:      make([]string, 0, 1000),
		Seqs:     make([]uint64, 0, 1000),
		indexes:  map[string]index.Index{"default": defaultIdx},
		idToIx:   make(map[uint64]int),
		Meta:     make(map[uint64]map[string]string),
		Deleted:  make(map[uint64]bool),
		Coll:     make(map[uint64]string),
		NumMeta:  make(map[uint64]map[string]float64),
		TimeMeta: make(map[uint64]map[string]time.Time),
		lexTF:    make(map[uint64]map[string]int),
		docLen:   make(map[uint64]int),
		df:       make(map[string]int),
		TenantID: make(map[uint64]string),
		quotas:   NewTenantQuota(),
	}
}

func setupBenchmarkStore(cfg BenchmarkConfig) *VectorStore {
	store := newTestStore(cfg.Dimension)
	vectors := generateRandomVectors(cfg.NumVectors, cfg.Dimension)

	for i, vec := range vectors {
		store.Add(vec, fmt.Sprintf("doc-%d", i), "", nil, "default", "default")
	}

	return store
}

func setupBenchmarkStoreWithDocs(cfg BenchmarkConfig) *VectorStore {
	store := newTestStore(cfg.Dimension)
	vectors := generateRandomVectors(cfg.NumVectors, cfg.Dimension)

	docs := []string{
		"vector search optimization techniques for large scale systems",
		"machine learning algorithms and neural network architectures",
		"database performance tuning and query optimization strategies",
		"distributed systems design patterns and fault tolerance",
		"natural language processing and text embedding methods",
	}

	for i, vec := range vectors {
		doc := docs[i%len(docs)]
		store.Add(vec, doc, "", nil, "default", "default")
	}

	return store
}

func percentile(latencies []time.Duration, p float64) time.Duration {
	if len(latencies) == 0 {
		return 0
	}

	// Sort (simple insertion sort for benchmark)
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}

	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}
