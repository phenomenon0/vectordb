package hnsw

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
	"time"
)

// TestParameterSweep runs a grid search over (workers, ef_construction) to find
// the sweet spot balancing insert QPS and recall.
func TestParameterSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping parameter sweep in short mode")
	}

	const (
		n          = 50000
		dim        = 128
		k          = 10
		numQueries = 200
		m          = 16
	)

	workerCounts := []int{1, 2, 4, 8, 12, 16}
	efConstructions := []int{100, 150, 200, 250, 300}

	// Generate vectors once.
	rng := rand.New(rand.NewSource(42))
	vectors := generateNormalizedVectors(n, dim, rng)

	// Precompute brute-force ground truth for recall measurement.
	queryRng := rand.New(rand.NewSource(99))
	queryIndices := make([]int, numQueries)
	for i := range queryIndices {
		queryIndices[i] = queryRng.Intn(n)
	}
	groundTruth := make([]map[uint64]bool, numQueries)
	for qi, idx := range queryIndices {
		query := vectors[idx]
		type idDist struct {
			id   uint64
			dist float32
		}
		dists := make([]idDist, n)
		for i := 0; i < n; i++ {
			dists[i] = idDist{uint64(i), CosineDistance(query, vectors[i])}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })
		truth := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			truth[dists[i].id] = true
		}
		groundTruth[qi] = truth
	}

	type result struct {
		workers        int
		efConstruction int
		insertQPS      float64
		recallAt10     float64
	}

	var results []result

	for _, efC := range efConstructions {
		for _, nw := range workerCounts {
			g := &Graph[uint64]{
				M:        m,
				Ml:       1.0 / math.Log(float64(m)),
				Distance: CosineDistance,
				EfSearch: efC, // ef_construction is used as EfSearch during build
				Rng:      rand.New(rand.NewSource(42)),
			}

			start := time.Now()

			if nw == 1 {
				// Sequential baseline.
				nodes := make([]Node[uint64], n)
				for i := 0; i < n; i++ {
					nodes[i] = MakeNode(uint64(i), vectors[i])
				}
				g.Add(nodes...)
			} else {
				// Concurrent.
				ch := make(chan int, n)
				for i := 0; i < n; i++ {
					ch <- i
				}
				close(ch)

				var wg sync.WaitGroup
				for w := 0; w < nw; w++ {
					wg.Add(1)
					go func(seed int64) {
						defer wg.Done()
						workerRng := rand.New(rand.NewSource(seed))
						for i := range ch {
							g.AddConcurrent(MakeNode(uint64(i), vectors[i]), workerRng)
						}
					}(int64(w*1000) + int64(efC))
				}
				wg.Wait()
			}

			elapsed := time.Since(start)
			insertQPS := float64(n) / elapsed.Seconds()

			// Verify graph size.
			if g.Len() != n {
				t.Fatalf("workers=%d efC=%d: graph size=%d, want %d", nw, efC, g.Len(), n)
			}

			// Measure recall@10 with ef_search=200 (production-like setting).
			var totalRecall float64
			for qi, idx := range queryIndices {
				query := vectors[idx]
				results := g.SearchWithEf(query, k, 200, nil)
				hits := 0
				for _, r := range results {
					if groundTruth[qi][r.Key] {
						hits++
					}
				}
				totalRecall += float64(hits) / float64(k)
			}
			recall := totalRecall / float64(numQueries)

			t.Logf("workers=%2d  ef_construction=%3d  QPS=%8.0f  recall@10=%.4f  time=%v",
				nw, efC, insertQPS, recall, elapsed.Round(time.Millisecond))

			results = append(results, result{
				workers:        nw,
				efConstruction: efC,
				insertQPS:      insertQPS,
				recallAt10:     recall,
			})
		}
	}

	// Print summary table.
	t.Log("")
	t.Log("=== PARAMETER SWEEP SUMMARY (50K vectors, 128d, M=16) ===")
	t.Log("")
	t.Logf("%-8s %-6s %10s %10s %10s", "Workers", "efC", "QPS", "Recall@10", "Score")
	t.Logf("%-8s %-6s %10s %10s %10s", "-------", "---", "---", "---------", "-----")

	// Score = QPS * recall^2 (penalizes low recall heavily).
	bestScore := 0.0
	var bestResult result
	for _, r := range results {
		score := r.insertQPS * r.recallAt10 * r.recallAt10
		if score > bestScore {
			bestScore = score
			bestResult = r
		}
		t.Logf("%-8d %-6d %10.0f %10.4f %10.0f", r.workers, r.efConstruction, r.insertQPS, r.recallAt10, score)
	}

	t.Log("")
	t.Logf("BEST: workers=%d ef_construction=%d → QPS=%.0f recall@10=%.4f",
		bestResult.workers, bestResult.efConstruction, bestResult.insertQPS, bestResult.recallAt10)

	// Print per-efC best worker count.
	t.Log("")
	t.Log("Best worker count per ef_construction:")
	for _, efC := range efConstructions {
		var best result
		bestS := 0.0
		for _, r := range results {
			if r.efConstruction != efC {
				continue
			}
			s := r.insertQPS * r.recallAt10 * r.recallAt10
			if s > bestS {
				bestS = s
				best = r
			}
		}
		speedup := best.insertQPS / results[0].insertQPS // vs sequential efC=100
		t.Logf("  efC=%3d → workers=%d  QPS=%.0f  recall=%.4f  (%.1fx vs baseline)",
			efC, best.workers, best.insertQPS, best.recallAt10, speedup)
	}
}

// TestMParameterSweep sweeps M values with the best worker/efC from above.
func TestMParameterSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping M sweep in short mode")
	}

	const (
		n          = 50000
		dim        = 128
		k          = 10
		numQueries = 200
		workers    = 8
	)

	mValues := []int{8, 12, 16, 24, 32}
	efConstructions := []int{150, 200, 250}

	rng := rand.New(rand.NewSource(42))
	vectors := generateNormalizedVectors(n, dim, rng)

	// Ground truth.
	queryRng := rand.New(rand.NewSource(99))
	queryIndices := make([]int, numQueries)
	for i := range queryIndices {
		queryIndices[i] = queryRng.Intn(n)
	}
	groundTruth := make([]map[uint64]bool, numQueries)
	for qi, idx := range queryIndices {
		query := vectors[idx]
		type idDist struct {
			id   uint64
			dist float32
		}
		dists := make([]idDist, n)
		for i := 0; i < n; i++ {
			dists[i] = idDist{uint64(i), CosineDistance(query, vectors[i])}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })
		truth := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			truth[dists[i].id] = true
		}
		groundTruth[qi] = truth
	}

	t.Log("=== M PARAMETER SWEEP (50K vectors, 128d, 8 workers) ===")
	t.Log("")
	t.Logf("%-4s %-6s %10s %10s %12s", "M", "efC", "QPS", "Recall@10", "SearchQPS")
	t.Logf("%-4s %-6s %10s %10s %12s", "--", "---", "---", "---------", "---------")

	for _, mVal := range mValues {
		for _, efC := range efConstructions {
			g := &Graph[uint64]{
				M:        mVal,
				Ml:       1.0 / math.Log(float64(mVal)),
				Distance: CosineDistance,
				EfSearch: efC,
				Rng:      rand.New(rand.NewSource(42)),
			}

			// Concurrent insert.
			ch := make(chan int, n)
			for i := 0; i < n; i++ {
				ch <- i
			}
			close(ch)

			start := time.Now()
			var wg sync.WaitGroup
			for w := 0; w < workers; w++ {
				wg.Add(1)
				go func(seed int64) {
					defer wg.Done()
					workerRng := rand.New(rand.NewSource(seed))
					for i := range ch {
						g.AddConcurrent(MakeNode(uint64(i), vectors[i]), workerRng)
					}
				}(int64(w*1000) + int64(efC) + int64(mVal))
			}
			wg.Wait()
			insertElapsed := time.Since(start)
			insertQPS := float64(n) / insertElapsed.Seconds()

			// Measure recall and search QPS.
			searchStart := time.Now()
			var totalRecall float64
			for qi, idx := range queryIndices {
				query := vectors[idx]
				results := g.SearchWithEf(query, k, 200, nil)
				hits := 0
				for _, r := range results {
					if groundTruth[qi][r.Key] {
						hits++
					}
				}
				totalRecall += float64(hits) / float64(k)
			}
			searchElapsed := time.Since(searchStart)
			recall := totalRecall / float64(numQueries)
			searchQPS := float64(numQueries) / searchElapsed.Seconds()

			t.Logf("%-4d %-6d %10.0f %10.4f %12.0f", mVal, efC, insertQPS, recall, searchQPS)
		}
	}
}

// TestEfSearchSweep measures recall at different ef_search values for a fixed graph.
func TestEfSearchSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping ef_search sweep in short mode")
	}

	const (
		n              = 50000
		dim            = 128
		k              = 10
		numQueries     = 200
		workers        = 8
		efConstruction = 200
		m              = 16
	)

	rng := rand.New(rand.NewSource(42))
	vectors := generateNormalizedVectors(n, dim, rng)

	// Build graph concurrently.
	g := &Graph[uint64]{
		M:        m,
		Ml:       1.0 / math.Log(float64(m)),
		Distance: CosineDistance,
		EfSearch: efConstruction,
		Rng:      rand.New(rand.NewSource(42)),
	}

	ch := make(chan int, n)
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			workerRng := rand.New(rand.NewSource(seed))
			for i := range ch {
				g.AddConcurrent(MakeNode(uint64(i), vectors[i]), workerRng)
			}
		}(int64(w * 1000))
	}
	wg.Wait()

	// Ground truth.
	queryRng := rand.New(rand.NewSource(99))
	queryIndices := make([]int, numQueries)
	for i := range queryIndices {
		queryIndices[i] = queryRng.Intn(n)
	}
	groundTruth := make([]map[uint64]bool, numQueries)
	for qi, idx := range queryIndices {
		query := vectors[idx]
		type idDist struct {
			id   uint64
			dist float32
		}
		dists := make([]idDist, n)
		for i := 0; i < n; i++ {
			dists[i] = idDist{uint64(i), CosineDistance(query, vectors[i])}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })
		truth := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			truth[dists[i].id] = true
		}
		groundTruth[qi] = truth
	}

	efSearchValues := []int{50, 100, 150, 200, 300, 400, 500}

	t.Logf("=== ef_search SWEEP (50K concurrent graph, M=%d, efC=%d, %d workers) ===", m, efConstruction, workers)
	t.Log("")
	t.Logf("%-10s %10s %12s", "ef_search", "Recall@10", "SearchQPS")
	t.Logf("%-10s %10s %12s", "---------", "---------", "---------")

	for _, ef := range efSearchValues {
		searchStart := time.Now()
		var totalRecall float64
		for qi, idx := range queryIndices {
			query := vectors[idx]
			results := g.SearchWithEf(query, k, ef, nil)
			hits := 0
			for _, r := range results {
				if groundTruth[qi][r.Key] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
		searchElapsed := time.Since(searchStart)
		recall := totalRecall / float64(numQueries)
		searchQPS := float64(numQueries) / searchElapsed.Seconds()

		t.Logf("%-10d %10.4f %12.0f", ef, recall, searchQPS)
	}
}

// TestDimensionSweep checks how dimension affects concurrent insert performance.
func TestDimensionSweep(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping dimension sweep in short mode")
	}

	const (
		n              = 20000
		k              = 10
		numQueries     = 100
		workers        = 8
		efConstruction = 200
		m              = 16
	)

	dims := []int{64, 128, 256, 384, 768, 1536}

	t.Log("=== DIMENSION SWEEP (20K vectors, 8 workers, M=16, efC=200) ===")
	t.Log("")
	t.Logf("%-6s %10s %10s %10s", "Dim", "QPS", "Recall@10", "Time")
	t.Logf("%-6s %10s %10s %10s", "---", "---", "---------", "----")

	for _, dim := range dims {
		rng := rand.New(rand.NewSource(42))
		vectors := generateNormalizedVectors(n, dim, rng)

		g := &Graph[uint64]{
			M:        m,
			Ml:       1.0 / math.Log(float64(m)),
			Distance: CosineDistance,
			EfSearch: efConstruction,
			Rng:      rand.New(rand.NewSource(42)),
		}

		ch := make(chan int, n)
		for i := 0; i < n; i++ {
			ch <- i
		}
		close(ch)

		start := time.Now()
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(seed int64) {
				defer wg.Done()
				workerRng := rand.New(rand.NewSource(seed))
				for i := range ch {
					g.AddConcurrent(MakeNode(uint64(i), vectors[i]), workerRng)
				}
			}(int64(w * 1000))
		}
		wg.Wait()
		elapsed := time.Since(start)
		insertQPS := float64(n) / elapsed.Seconds()

		// Quick recall check.
		queryRng := rand.New(rand.NewSource(99))
		var totalRecall float64
		for q := 0; q < numQueries; q++ {
			idx := queryRng.Intn(n)
			query := vectors[idx]

			type idDist struct {
				id   uint64
				dist float32
			}
			dists := make([]idDist, n)
			for i := 0; i < n; i++ {
				dists[i] = idDist{uint64(i), CosineDistance(query, vectors[i])}
			}
			sort.Slice(dists, func(a, b int) bool { return dists[a].dist < dists[b].dist })
			truth := make(map[uint64]bool, k)
			for i := 0; i < k; i++ {
				truth[dists[i].id] = true
			}

			results := g.SearchWithEf(query, k, 200, nil)
			hits := 0
			for _, r := range results {
				if truth[r.Key] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
		recall := totalRecall / float64(numQueries)

		t.Logf("%-6d %10.0f %10.4f %10v", dim, insertQPS, recall, elapsed.Round(time.Millisecond))
	}
}
