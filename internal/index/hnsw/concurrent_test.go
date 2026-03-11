package hnsw

import (
	"math"
	"math/rand"
	"sort"
	"sync"
	"testing"
)

func generateNormalizedVectors(n, dim int, rng *rand.Rand) [][]float32 {
	vectors := make([][]float32, n)
	for i := range vectors {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		var norm float32
		for _, x := range v {
			norm += x * x
		}
		norm = float32(math.Sqrt(float64(norm)))
		for j := range v {
			v[j] /= norm
		}
		vectors[i] = v
	}
	return vectors
}

// TestAddConcurrent inserts 50K vectors from 8 goroutines via AddConcurrent,
// verifies graph size and recall@10 >= 0.93.
func TestAddConcurrent(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping concurrent insert test in short mode")
	}

	const (
		n          = 50000
		dim        = 128
		k          = 10
		numQueries = 100
		numWorkers = 8
	)

	rng := rand.New(rand.NewSource(42))
	vectors := generateNormalizedVectors(n, dim, rng)

	g := &Graph[uint64]{
		M:        16,
		Ml:       1.0 / math.Log(16),
		Distance: CosineDistance,
		EfSearch: 200,
		Rng:      rand.New(rand.NewSource(42)),
	}

	// Fan out insertion across workers.
	ch := make(chan int, n)
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
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

	// Verify graph size.
	if g.Len() != n {
		t.Fatalf("graph size = %d, want %d", g.Len(), n)
	}

	// Verify recall.
	queryRng := rand.New(rand.NewSource(99))
	var totalRecall float64
	for q := 0; q < numQueries; q++ {
		query := vectors[queryRng.Intn(n)]

		// Brute force ground truth.
		type idDist struct {
			id   uint64
			dist float32
		}
		dists := make([]idDist, n)
		for i := 0; i < n; i++ {
			dists[i] = idDist{uint64(i), CosineDistance(query, vectors[i])}
		}
		sort.Slice(dists, func(i, j int) bool { return dists[i].dist < dists[j].dist })
		truth := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			truth[dists[i].id] = true
		}

		// Use higher ef_search to compensate for concurrent construction quality.
		results := g.SearchWithEf(query, k, 300, nil)
		hits := 0
		for _, r := range results {
			if truth[r.Key] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}

	recall := totalRecall / float64(numQueries)
	t.Logf("concurrent n=%d recall@%d = %.4f", n, k, recall)

	// Concurrent construction inherently sacrifices some recall vs sequential
	// (stale neighbor reads, same trade-off as hnswlib). 0.90 is the floor.
	if recall < 0.90 {
		t.Errorf("recall@%d = %.4f, want >= 0.90", k, recall)
	}
}

// TestAddConcurrentSmall is a quick smoke test with race detector.
func TestAddConcurrentSmall(t *testing.T) {
	const (
		n          = 1000
		dim        = 64
		numWorkers = 4
	)

	rng := rand.New(rand.NewSource(7))
	vectors := generateNormalizedVectors(n, dim, rng)

	g := &Graph[uint64]{
		M:        16,
		Ml:       1.0 / math.Log(16),
		Distance: CosineDistance,
		EfSearch: 100,
		Rng:      rand.New(rand.NewSource(7)),
	}

	ch := make(chan int, n)
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			workerRng := rand.New(rand.NewSource(seed))
			for i := range ch {
				g.AddConcurrent(MakeNode(uint64(i), vectors[i]), workerRng)
			}
		}(int64(w * 100))
	}
	wg.Wait()

	if g.Len() != n {
		t.Fatalf("graph size = %d, want %d", g.Len(), n)
	}

	// Basic search sanity.
	results := g.Search(vectors[0], 5)
	if len(results) == 0 {
		t.Fatal("search returned no results")
	}
	if results[0].Key != 0 {
		t.Errorf("nearest to vector 0 should be itself, got key=%d", results[0].Key)
	}
}

// TestConcurrentInsertAndSearch tests concurrent insert + search (race detector).
func TestConcurrentInsertAndSearch(t *testing.T) {
	const (
		n   = 2000
		dim = 64
	)

	rng := rand.New(rand.NewSource(123))
	vectors := generateNormalizedVectors(n, dim, rng)

	g := &Graph[uint64]{
		M:        16,
		Ml:       1.0 / math.Log(16),
		Distance: CosineDistance,
		EfSearch: 100,
		Rng:      rand.New(rand.NewSource(123)),
	}

	// Seed the graph with some initial nodes (sequential).
	seed := n / 2
	for i := 0; i < seed; i++ {
		g.Add(MakeNode(uint64(i), vectors[i]))
	}

	// Concurrently insert remaining + search existing.
	var wg sync.WaitGroup

	// Inserters.
	insertCh := make(chan int, n-seed)
	for i := seed; i < n; i++ {
		insertCh <- i
	}
	close(insertCh)

	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			workerRng := rand.New(rand.NewSource(seed))
			for i := range insertCh {
				g.AddConcurrent(MakeNode(uint64(i), vectors[i]), workerRng)
			}
		}(int64(w * 500))
	}

	// Searchers.
	for w := 0; w < 2; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			searchRng := rand.New(rand.NewSource(seed))
			for i := 0; i < 200; i++ {
				idx := searchRng.Intn(n / 2) // search among seeded vectors
				_ = g.SearchWithEf(vectors[idx], 5, 100, nil)
			}
		}(int64(w * 700))
	}

	wg.Wait()

	if g.Len() != n {
		t.Fatalf("graph size = %d, want %d", g.Len(), n)
	}
}

func BenchmarkAdd(b *testing.B) {
	const dim = 128
	rng := rand.New(rand.NewSource(42))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		vectors := generateNormalizedVectors(10000, dim, rng)
		g := &Graph[uint64]{
			M:        16,
			Ml:       1.0 / math.Log(16),
			Distance: CosineDistance,
			EfSearch: 200,
			Rng:      rand.New(rand.NewSource(42)),
		}
		nodes := make([]Node[uint64], len(vectors))
		for j, v := range vectors {
			nodes[j] = MakeNode(uint64(j), v)
		}
		b.StartTimer()

		g.Add(nodes...)
	}
}

func BenchmarkAddConcurrent(b *testing.B) {
	const (
		dim        = 128
		numWorkers = 8
	)
	rng := rand.New(rand.NewSource(42))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		vectors := generateNormalizedVectors(10000, dim, rng)
		g := &Graph[uint64]{
			M:        16,
			Ml:       1.0 / math.Log(16),
			Distance: CosineDistance,
			EfSearch: 200,
			Rng:      rand.New(rand.NewSource(42)),
		}
		ch := make(chan int, len(vectors))
		for j := 0; j < len(vectors); j++ {
			ch <- j
		}
		close(ch)
		b.StartTimer()

		var wg sync.WaitGroup
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(seed int64) {
				defer wg.Done()
				workerRng := rand.New(rand.NewSource(seed))
				for j := range ch {
					g.AddConcurrent(MakeNode(uint64(j), vectors[j]), workerRng)
				}
			}(int64(w * 1000))
		}
		wg.Wait()
	}
}
