package index

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

// TestIVFBinaryBasic tests basic functionality
func TestIVFBinaryBasic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping IVF binary test in short mode (slow k-means training)")
	}
	dim := 768
	nlist := 100

	idx, err := NewIVFBinaryIndex(IVFBinaryConfig{
		Dim:    dim,
		Nlist:  nlist,
		Nprobe: 10,
	})
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Generate training data
	numTrain := 10000
	trainVecs := generateRandomVectorsIVF(numTrain, dim)

	// Train
	t.Log("Training index...")
	start := time.Now()
	if err := idx.Train(trainVecs); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}
	t.Logf("Training took: %v", time.Since(start))

	// Add vectors
	numVecs := 50000
	vecs := generateRandomVectorsIVF(numVecs, dim)
	ids := make([]uint64, numVecs)
	for i := range ids {
		ids[i] = uint64(i)
	}

	t.Log("Adding vectors...")
	start = time.Now()
	if err := idx.AddBatch(ids, vecs); err != nil {
		t.Fatalf("Failed to add: %v", err)
	}
	t.Logf("Added %d vectors in %v", numVecs, time.Since(start))

	// Search
	query := generateRandomVectorsIVF(1, dim)
	results, err := idx.Search(query, 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("Expected 10 results, got %d", len(results))
	}

	t.Logf("Top 10 results:")
	for i, r := range results {
		t.Logf("  %d: ID=%d Score=%.4f", i+1, r.ID, r.Score)
	}

	// Stats
	stats := idx.Stats()
	t.Logf("Index stats: %+v", stats)
}

// BenchmarkIVFBinarySearch benchmarks search performance
func BenchmarkIVFBinarySearch(b *testing.B) {
	dims := []int{768}
	sizes := []int{100000, 500000, 1000000}
	nprobes := []int{10, 20, 50}

	for _, dim := range dims {
		for _, size := range sizes {
			for _, nprobe := range nprobes {
				name := fmt.Sprintf("dim%d_size%dk_nprobe%d", dim, size/1000, nprobe)
				b.Run(name, func(b *testing.B) {
					benchmarkIVFBinary(b, dim, size, nprobe)
				})
			}
		}
	}
}

func benchmarkIVFBinary(b *testing.B, dim, numVecs, nprobe int) {
	// Calculate optimal nlist (sqrt(n) is a good heuristic)
	nlist := int(float64(numVecs) * 0.01) // 1% of vectors per cluster
	if nlist < 100 {
		nlist = 100
	}
	if nlist > 1000 {
		nlist = 1000
	}

	idx, err := NewIVFBinaryIndex(IVFBinaryConfig{
		Dim:    dim,
		Nlist:  nlist,
		Nprobe: nprobe,
	})
	if err != nil {
		b.Fatalf("Failed to create index: %v", err)
	}

	// Generate and add vectors
	b.Logf("Generating %d vectors...", numVecs)
	vecs := generateRandomVectorsIVF(numVecs, dim)

	// Train on subset
	trainSize := numVecs / 10
	if trainSize < 10000 {
		trainSize = 10000
	}
	if trainSize > numVecs {
		trainSize = numVecs
	}

	b.Logf("Training with %d vectors, nlist=%d...", trainSize, nlist)
	if err := idx.Train(vecs[:trainSize*dim]); err != nil {
		b.Fatalf("Training failed: %v", err)
	}

	// Add vectors in batches
	ids := make([]uint64, numVecs)
	for i := range ids {
		ids[i] = uint64(i)
	}

	b.Logf("Adding %d vectors...", numVecs)
	start := time.Now()

	batchSize := 10000
	for i := 0; i < numVecs; i += batchSize {
		end := i + batchSize
		if end > numVecs {
			end = numVecs
		}
		if err := idx.AddBatch(ids[i:end], vecs[i*dim:end*dim]); err != nil {
			b.Fatalf("Failed to add batch: %v", err)
		}
	}
	addTime := time.Since(start)
	b.Logf("Added %d vectors in %v (%.0f vec/s)", numVecs, addTime, float64(numVecs)/addTime.Seconds())

	// Report memory usage
	memUsage := idx.MemoryUsage()
	originalSize := int64(numVecs * dim * 4)
	compression := float64(originalSize) / float64(memUsage)
	b.Logf("Memory: %.2f MB (%.1fx compression)", float64(memUsage)/(1024*1024), compression)

	// Generate query vectors
	numQueries := 1000
	queries := generateRandomVectorsIVF(numQueries, dim)

	// Warmup
	for i := 0; i < 100; i++ {
		idx.Search(queries[(i%numQueries)*dim:(i%numQueries+1)*dim], 10)
	}

	// Benchmark
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		queryIdx := i % numQueries
		_, err := idx.Search(queries[queryIdx*dim:(queryIdx+1)*dim], 10)
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}

	b.StopTimer()

	// Report QPS
	elapsed := b.Elapsed()
	qps := float64(b.N) / elapsed.Seconds()
	b.ReportMetric(qps, "qps")
}

// BenchmarkCompareIndexTypes compares different index types
func BenchmarkCompareIndexTypes(b *testing.B) {
	dim := 768
	numVecs := 100000

	vecs := generateRandomVectorsIVF(numVecs, dim)
	ids := make([]uint64, numVecs)
	for i := range ids {
		ids[i] = uint64(i)
	}
	queries := generateRandomVectorsIVF(100, dim)

	b.Run("BinaryIndex", func(b *testing.B) {
		idx := NewBinaryIndex(dim)
		idx.Train(vecs[:10000*dim])
		idx.AddBatch(ids, vecs)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			queryIdx := i % 100
			idx.Search(queries[queryIdx*dim:(queryIdx+1)*dim], 10)
		}
	})

	b.Run("IVFBinary_nprobe10", func(b *testing.B) {
		idx, _ := NewIVFBinaryIndex(IVFBinaryConfig{Dim: dim, Nlist: 100, Nprobe: 10})
		idx.Train(vecs[:10000*dim])
		idx.AddBatch(ids, vecs)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			queryIdx := i % 100
			idx.Search(queries[queryIdx*dim:(queryIdx+1)*dim], 10)
		}
	})

	b.Run("IVFBinary_nprobe20", func(b *testing.B) {
		idx, _ := NewIVFBinaryIndex(IVFBinaryConfig{Dim: dim, Nlist: 100, Nprobe: 20})
		idx.Train(vecs[:10000*dim])
		idx.AddBatch(ids, vecs)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			queryIdx := i % 100
			idx.Search(queries[queryIdx*dim:(queryIdx+1)*dim], 10)
		}
	})
}

// TestIVFBinaryRecall tests recall quality
func TestIVFBinaryRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping recall test in short mode")
	}

	dim := 768
	numVecs := 10000
	numQueries := 100
	k := 10

	vecs := generateRandomVectorsIVF(numVecs, dim)
	ids := make([]uint64, numVecs)
	for i := range ids {
		ids[i] = uint64(i)
	}
	queries := generateRandomVectorsIVF(numQueries, dim)

	// Build IVF-Binary index
	ivfIdx, _ := NewIVFBinaryIndex(IVFBinaryConfig{Dim: dim, Nlist: 100, Nprobe: 20})
	ivfIdx.Train(vecs)
	ivfIdx.AddBatch(ids, vecs)

	// Compute ground truth using brute force
	computeGroundTruth := func(query []float32) []uint64 {
		type scored struct {
			id    uint64
			score float32
		}
		scores := make([]scored, numVecs)
		for i := 0; i < numVecs; i++ {
			vec := vecs[i*dim : (i+1)*dim]
			scores[i] = scored{id: uint64(i), score: cosineSimilarity(query, vec)}
		}
		// Sort by score descending
		for i := 0; i < k; i++ {
			maxIdx := i
			for j := i + 1; j < len(scores); j++ {
				if scores[j].score > scores[maxIdx].score {
					maxIdx = j
				}
			}
			scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
		}
		result := make([]uint64, k)
		for i := 0; i < k; i++ {
			result[i] = scores[i].id
		}
		return result
	}

	// Compute recall
	var totalRecall float64
	for q := 0; q < numQueries; q++ {
		query := queries[q*dim : (q+1)*dim]

		groundTruth := computeGroundTruth(query)
		gtSet := make(map[uint64]bool)
		for _, id := range groundTruth {
			gtSet[id] = true
		}

		results, _ := ivfIdx.Search(query, k)
		hits := 0
		for _, r := range results {
			if gtSet[r.ID] {
				hits++
			}
		}

		totalRecall += float64(hits) / float64(k)
	}

	avgRecall := totalRecall / float64(numQueries)
	t.Logf("Average recall@%d: %.2f%% (nprobe=20, nlist=100)", k, avgRecall*100)

	if avgRecall < 0.5 {
		t.Errorf("Recall too low: %.2f%% (expected > 50%%)", avgRecall*100)
	}
}

// generateRandomVectorsIVF generates random normalized vectors for IVF tests
func generateRandomVectorsIVF(n, dim int) []float32 {
	rand.Seed(time.Now().UnixNano())
	vecs := make([]float32, n*dim)

	numWorkers := runtime.NumCPU()
	chunkSize := (n + numWorkers - 1) / numWorkers

	done := make(chan bool, numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			start := workerID * chunkSize
			end := start + chunkSize
			if end > n {
				end = n
			}

			localRand := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID)))

			for i := start; i < end; i++ {
				var norm float32
				for j := 0; j < dim; j++ {
					v := localRand.Float32()*2 - 1 // [-1, 1]
					vecs[i*dim+j] = v
					norm += v * v
				}
				// Normalize
				norm = float32(1.0 / float64(norm+1e-8))
				for j := 0; j < dim; j++ {
					vecs[i*dim+j] *= norm
				}
			}
			done <- true
		}(w)
	}

	for w := 0; w < numWorkers; w++ {
		<-done
	}

	return vecs
}
