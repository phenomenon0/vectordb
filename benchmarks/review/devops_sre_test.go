package review

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/benchmarks/competitive"
	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/index"
)

// DevOps/SRE Persona: latency shape, memory growth, GC pressure, goroutine leaks.

func TestDevOpsSREReview(t *testing.T) {
	review := NewReview("DevOps/SRE",
		"Latency distribution, memory growth, GC pressure, goroutine leaks, sustained load")

	ctx := context.Background()
	rng := rand.New(rand.NewSource(42))
	dim := 128
	scale := 10_000
	if testing.Short() {
		scale = 2000
	}

	vectors := testdata.GenerateClusteredVectors(scale, dim, 20, 0.15, rng)
	queries := testdata.GenerateQueries(200, dim, 20, 0.15, rng)

	idx, err := index.Create("hnsw", dim, map[string]interface{}{
		"m": 16, "ef_search": 64, "ef_construction": 200,
	})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vectors {
		idx.Add(ctx, uint64(i), v)
	}

	// Check 1: P99/P50 ratio < 10x
	t.Run("latency_shape", func(t *testing.T) {
		numOps := 500
		if testing.Short() {
			numOps = 100
		}

		latencies := make([]time.Duration, numOps)
		for i := 0; i < numOps; i++ {
			q := queries[i%len(queries)]
			start := time.Now()
			_, _ = idx.Search(ctx, q, 10, &index.HNSWSearchParams{EfSearch: 64})
			latencies[i] = time.Since(start)
		}

		_, p50, _, p99, _ := competitive.LatencyStats(latencies)

		ratio := p99 / p50
		t.Logf("P50=%.0fus, P99=%.0fus, ratio=%.1f", p50, p99, ratio)

		if ratio < 10.0 {
			review.Pass("latency_shape", "P99/P50 ratio < 10x", SeverityHigh,
				fmt.Sprintf("ratio=%.1f (p50=%.0fus, p99=%.0fus)", ratio, p50, p99))
		} else {
			review.Fail("latency_shape", "P99/P50 ratio < 10x", SeverityHigh,
				fmt.Sprintf("ratio=%.1f indicates tail latency spikes", ratio))
		}
	})

	// Check 2: Memory growth linearity
	t.Run("memory_growth", func(t *testing.T) {
		checkpoints := []int{1000, 2000, 5000}
		if testing.Short() {
			checkpoints = []int{500, 1000}
		}
		memSamples := make([]float64, 0)

		for _, cp := range checkpoints {
			memMB, _, err := competitive.MeasureMemory("hnsw", dim,
				map[string]interface{}{"m": 16, "ef_construction": 200},
				vectors[:cp])
			if err != nil {
				continue
			}
			memSamples = append(memSamples, memMB)
			t.Logf("Scale %d: %.2f MB", cp, memMB)
		}

		if len(memSamples) >= 2 {
			// Check that growth is roughly linear (not exponential)
			// Ratio of last/first should be < 2x the ratio of scales
			scaleRatio := float64(checkpoints[len(checkpoints)-1]) / float64(checkpoints[0])
			memRatio := memSamples[len(memSamples)-1] / memSamples[0]
			superlinearRatio := memRatio / scaleRatio

			if superlinearRatio < 2.0 {
				review.Pass("memory_growth", "Memory growth is approximately linear", SeverityMedium,
					fmt.Sprintf("scale ratio=%.1f, mem ratio=%.1f", scaleRatio, memRatio))
			} else {
				review.Fail("memory_growth", "Memory growth is approximately linear", SeverityMedium,
					fmt.Sprintf("superlinear: scale ratio=%.1f, mem ratio=%.1f", scaleRatio, memRatio))
			}
		} else {
			review.Fail("memory_growth", "Enough data points for memory growth check", SeverityLow,
				"Not enough checkpoints to measure")
		}
	})

	// Check 3: GC pause tracking
	t.Run("gc_pressure", func(t *testing.T) {
		runtime.GC()
		var beforeGC runtime.MemStats
		runtime.ReadMemStats(&beforeGC)
		beforePauses := beforeGC.NumGC

		// Do 200 search operations
		for i := 0; i < 200; i++ {
			q := queries[i%len(queries)]
			_, _ = idx.Search(ctx, q, 10, &index.HNSWSearchParams{EfSearch: 64})
		}

		var afterGC runtime.MemStats
		runtime.ReadMemStats(&afterGC)
		gcCount := afterGC.NumGC - beforePauses
		pauseTotal := afterGC.PauseTotalNs - beforeGC.PauseTotalNs

		t.Logf("GC during 200 searches: %d collections, %dms total pause", gcCount, pauseTotal/1e6)

		// During 200 searches, shouldn't trigger excessive GC
		if gcCount < 20 {
			review.Pass("gc_pressure", "GC pressure acceptable during search", SeverityMedium,
				fmt.Sprintf("%d collections in 200 searches", gcCount))
		} else {
			review.Fail("gc_pressure", "GC pressure acceptable during search", SeverityMedium,
				fmt.Sprintf("%d GC collections in 200 searches (excessive)", gcCount))
		}
	})

	// Check 4: Goroutine leak detection
	t.Run("goroutine_leak", func(t *testing.T) {
		beforeGoroutines := runtime.NumGoroutine()

		// Run search operations
		for i := 0; i < 100; i++ {
			q := queries[i%len(queries)]
			_, _ = idx.Search(ctx, q, 10, &index.HNSWSearchParams{EfSearch: 64})
		}

		// Small sleep to let any leaked goroutines settle
		time.Sleep(100 * time.Millisecond)
		afterGoroutines := runtime.NumGoroutine()

		leaked := afterGoroutines - beforeGoroutines
		t.Logf("Goroutines: before=%d, after=%d, diff=%d", beforeGoroutines, afterGoroutines, leaked)

		if leaked <= 2 { // Allow small variance
			review.Pass("goroutine_leak", "No goroutine leaks during search", SeverityHigh,
				fmt.Sprintf("delta=%d", leaked))
		} else {
			review.Fail("goroutine_leak", "No goroutine leaks during search", SeverityHigh,
				fmt.Sprintf("%d goroutines leaked", leaked))
		}
	})

	// Check 5: Sustained load (10 seconds)
	t.Run("sustained_load", func(t *testing.T) {
		duration := 10 * time.Second
		if testing.Short() {
			duration = 3 * time.Second
		}

		errorCount := 0
		totalOps := 0
		start := time.Now()

		for time.Since(start) < duration {
			q := queries[totalOps%len(queries)]
			_, err := idx.Search(ctx, q, 10, &index.HNSWSearchParams{EfSearch: 64})
			if err != nil {
				errorCount++
			}
			totalOps++
		}
		elapsed := time.Since(start)
		qps := float64(totalOps) / elapsed.Seconds()
		errorRate := float64(errorCount) / float64(totalOps)

		t.Logf("Sustained load: %.0f QPS over %v, %d errors (%.2f%%)",
			qps, elapsed.Round(time.Second), errorCount, errorRate*100)

		if errorRate < 0.001 { // < 0.1% error rate
			review.Pass("sustained_load", "Error rate < 0.1% under sustained load", SeverityHigh,
				fmt.Sprintf("%.0f QPS, %.4f%% errors", qps, errorRate*100))
		} else {
			review.Fail("sustained_load", "Error rate < 0.1% under sustained load", SeverityHigh,
				fmt.Sprintf("error rate=%.2f%%", errorRate*100))
		}
	})

	// Check 6: Insert doesn't block reads excessively
	t.Run("insert_read_isolation", func(t *testing.T) {
		// Measure baseline read latency
		baseLatencies := make([]time.Duration, 50)
		for i := 0; i < 50; i++ {
			start := time.Now()
			_, _ = idx.Search(ctx, queries[i%len(queries)], 10, &index.HNSWSearchParams{EfSearch: 64})
			baseLatencies[i] = time.Since(start)
		}
		_, baseP50, _, _, _ := competitive.LatencyStats(baseLatencies)

		// Now insert while reading
		done := make(chan bool)
		go func() {
			for i := 0; i < 500; i++ {
				v := vectors[i%len(vectors)]
				_ = idx.Add(ctx, uint64(scale+i), v)
			}
			done <- true
		}()

		loadLatencies := make([]time.Duration, 50)
		for i := 0; i < 50; i++ {
			start := time.Now()
			_, _ = idx.Search(ctx, queries[i%len(queries)], 10, &index.HNSWSearchParams{EfSearch: 64})
			loadLatencies[i] = time.Since(start)
		}
		<-done

		_, loadP50, _, _, _ := competitive.LatencyStats(loadLatencies)
		ratio := loadP50 / baseP50

		t.Logf("Read P50: baseline=%.0fus, under write load=%.0fus, ratio=%.1fx", baseP50, loadP50, ratio)

		if ratio < 5.0 {
			review.Pass("insert_read_isolation", "Read latency < 5x during concurrent writes", SeverityMedium,
				fmt.Sprintf("%.1fx slowdown", ratio))
		} else {
			review.Fail("insert_read_isolation", "Read latency < 5x during concurrent writes", SeverityMedium,
				fmt.Sprintf("%.1fx slowdown (excessive)", ratio))
		}
	})

	review.Report(t)
}
