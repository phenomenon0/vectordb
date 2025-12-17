package index

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// =============================================================================
// PQ-ADC Unit Tests
// =============================================================================

func TestPQIndex_Basic(t *testing.T) {
	dim := 128
	numVectors := 1000
	if testing.Short() {
		numVectors = 300 // Minimum for k-means with ksub=256
	}
	numQueries := 10
	k := 10

	// Create index
	idx, err := NewPQIndex(PQIndexConfig{
		Dim:  dim,
		M:    16,  // 16 subvectors
		Ksub: 256, // 8-bit codes
	})
	if err != nil {
		t.Fatalf("Failed to create PQ index: %v", err)
	}

	// Generate random vectors
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}

	// Train
	if err := idx.Train(vectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	// Add vectors
	ids := make([]uint64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = uint64(i)
	}
	if err := idx.AddBatch(ids, vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Verify size
	if idx.Size() != numVectors {
		t.Errorf("Expected size %d, got %d", numVectors, idx.Size())
	}

	// Search
	for q := 0; q < numQueries; q++ {
		query := vectors[q*dim : (q+1)*dim]
		results, err := idx.Search(query, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) != k {
			t.Errorf("Expected %d results, got %d", k, len(results))
		}

		// The query vector itself should be among the top results
		found := false
		for _, r := range results {
			if r.ID == uint64(q) {
				found = true
				break
			}
		}
		if !found {
			t.Logf("Warning: Query vector %d not in top-%d results (this can happen with PQ approximation)", q, k)
		}
	}

	// Print stats
	stats := idx.Stats()
	t.Logf("PQ Index Stats: %+v", stats)
	t.Logf("Compression Ratio: %.1fx", idx.CompressionRatio())
	t.Logf("Memory Usage: %.2f MB", float64(idx.MemoryUsage())/(1024*1024))
}

func TestIVFPQIndex_Basic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping IVF-PQ test in short mode (k-means training is slow)")
	}

	dim := 128
	numVectors := 5000
	numQueries := 10
	k := 10

	// Create index
	idx, err := NewIVFPQIndex(IVFPQConfig{
		Dim:    dim,
		Nlist:  50,  // 50 clusters
		Nprobe: 10,  // Search 10 clusters
		M:      16,  // 16 subvectors
		Ksub:   256, // 8-bit codes
	})
	if err != nil {
		t.Fatalf("Failed to create IVF-PQ index: %v", err)
	}

	// Generate random vectors
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}

	// Train
	if err := idx.Train(vectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	// Add vectors
	ids := make([]uint64, numVectors)
	for i := 0; i < numVectors; i++ {
		ids[i] = uint64(i)
	}
	if err := idx.AddBatch(ids, vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Verify size
	if idx.Size() != numVectors {
		t.Errorf("Expected size %d, got %d", numVectors, idx.Size())
	}

	// Search
	for q := 0; q < numQueries; q++ {
		query := vectors[q*dim : (q+1)*dim]
		results, err := idx.Search(query, k)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) < 1 {
			t.Errorf("Expected at least 1 result, got %d", len(results))
		}
	}

	// Print stats
	stats := idx.Stats()
	t.Logf("IVF-PQ Index Stats: %+v", stats)
}

func TestDistanceTable_Correctness(t *testing.T) {
	dim := 64
	m := 8
	ksub := 256

	// Create and train PQ
	pq, err := NewProductQuantizer(dim, m, ksub)
	if err != nil {
		t.Fatalf("Failed to create PQ: %v", err)
	}

	// Generate training data
	trainingData := make([]float32, 1000*dim)
	for i := range trainingData {
		trainingData[i] = rand.Float32()
	}
	if err := pq.Train(trainingData, 25); err != nil {
		t.Fatalf("Failed to train PQ: %v", err)
	}

	// Generate query and test vector
	query := make([]float32, dim)
	testVec := make([]float32, dim)
	for i := 0; i < dim; i++ {
		query[i] = rand.Float32()
		testVec[i] = rand.Float32()
	}

	// Quantize test vector
	codes, err := pq.Quantize(testVec)
	if err != nil {
		t.Fatalf("Failed to quantize: %v", err)
	}

	// Compute distance using table lookup
	dt, err := pq.ComputeDistanceTable(query)
	if err != nil {
		t.Fatalf("Failed to compute distance table: %v", err)
	}
	adcDist := dt.LookupDistance(codes)

	// Compute distance directly (dequantize and compare)
	reconstructed, err := pq.Dequantize(codes)
	if err != nil {
		t.Fatalf("Failed to dequantize: %v", err)
	}
	var directDist float32
	for i := 0; i < dim; i++ {
		diff := query[i] - reconstructed[i]
		directDist += diff * diff
	}

	// They should be equal (both use same PQ codebooks)
	if abs32(adcDist-directDist) > 1e-5 {
		t.Errorf("ADC distance mismatch: ADC=%.6f, direct=%.6f", adcDist, directDist)
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// =============================================================================
// PQ-ADC Benchmarks - Compare ADC vs Naive
// =============================================================================

// BenchmarkADC_vs_Naive compares ADC table lookup against naive distance computation
func BenchmarkADC_vs_Naive(b *testing.B) {
	dims := []int{128, 384, 768}
	numVectors := 10000

	for _, dim := range dims {
		b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
			m := dim / 8
			if m < 4 {
				m = 4
			}
			ksub := 256

			// Create and train PQ
			pq, err := NewProductQuantizer(dim, m, ksub)
			if err != nil {
				b.Fatalf("Failed to create PQ: %v", err)
			}

			// Generate and train
			trainingData := make([]float32, numVectors*dim)
			for i := range trainingData {
				trainingData[i] = rand.Float32()
			}
			if err := pq.Train(trainingData, 15); err != nil {
				b.Fatalf("Failed to train PQ: %v", err)
			}

			// Quantize all vectors
			codes, err := pq.Quantize(trainingData)
			if err != nil {
				b.Fatalf("Failed to quantize: %v", err)
			}

			// Generate query
			query := make([]float32, dim)
			for i := range query {
				query[i] = rand.Float32()
			}

			// Benchmark ADC (table lookup)
			b.Run("ADC", func(b *testing.B) {
				dt, _ := pq.ComputeDistanceTable(query)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = dt.LookupDistanceBatch(codes, numVectors)
				}
				b.ReportMetric(float64(numVectors*b.N)/b.Elapsed().Seconds()/1e6, "Mvecs/s")
			})

			// Benchmark Naive (reconstruct + compare)
			b.Run("Naive", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < numVectors; j++ {
						vecCodes := codes[j*m : (j+1)*m]
						// Reconstruct each subvector and compute distance
						var dist float32
						for sub := 0; sub < m; sub++ {
							centroid := pq.codebooks[sub][vecCodes[sub]]
							querySubvec := query[sub*pq.dsub : (sub+1)*pq.dsub]
							for d := 0; d < pq.dsub; d++ {
								diff := querySubvec[d] - centroid[d]
								dist += diff * diff
							}
						}
						_ = dist
					}
				}
				b.ReportMetric(float64(numVectors*b.N)/b.Elapsed().Seconds()/1e6, "Mvecs/s")
			})
		})
	}
}

// BenchmarkPQIndex_Search benchmarks the full PQ index search
func BenchmarkPQIndex_Search(b *testing.B) {
	scales := []int{10000, 50000, 100000}
	dim := 384

	for _, numVectors := range scales {
		b.Run(fmt.Sprintf("n=%d", numVectors), func(b *testing.B) {
			idx, err := NewPQIndex(PQIndexConfig{
				Dim:  dim,
				M:    48, // 48 subvectors for 384d
				Ksub: 256,
			})
			if err != nil {
				b.Fatalf("Failed to create index: %v", err)
			}

			// Generate and train
			vectors := make([]float32, numVectors*dim)
			for i := range vectors {
				vectors[i] = rand.Float32()
			}
			if err := idx.Train(vectors); err != nil {
				b.Fatalf("Failed to train: %v", err)
			}

			// Add vectors
			ids := make([]uint64, numVectors)
			for i := range ids {
				ids[i] = uint64(i)
			}
			if err := idx.AddBatch(ids, vectors); err != nil {
				b.Fatalf("Failed to add: %v", err)
			}

			// Generate query
			query := make([]float32, dim)
			for i := range query {
				query[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := idx.Search(query, 10)
				if err != nil {
					b.Fatal(err)
				}
			}
			qps := float64(b.N) / b.Elapsed().Seconds()
			b.ReportMetric(qps, "QPS")
		})
	}
}

// BenchmarkIVFPQIndex_Search benchmarks IVF-PQ index search
func BenchmarkIVFPQIndex_Search(b *testing.B) {
	scales := []int{10000, 50000, 100000}
	dim := 384

	for _, numVectors := range scales {
		b.Run(fmt.Sprintf("n=%d", numVectors), func(b *testing.B) {
			nlist := numVectors / 100
			if nlist < 10 {
				nlist = 10
			}
			if nlist > 1000 {
				nlist = 1000
			}

			idx, err := NewIVFPQIndex(IVFPQConfig{
				Dim:    dim,
				Nlist:  nlist,
				Nprobe: 10,
				M:      48,
				Ksub:   256,
			})
			if err != nil {
				b.Fatalf("Failed to create index: %v", err)
			}

			// Generate and train
			vectors := make([]float32, numVectors*dim)
			for i := range vectors {
				vectors[i] = rand.Float32()
			}
			if err := idx.Train(vectors); err != nil {
				b.Fatalf("Failed to train: %v", err)
			}

			// Add vectors
			ids := make([]uint64, numVectors)
			for i := range ids {
				ids[i] = uint64(i)
			}
			if err := idx.AddBatch(ids, vectors); err != nil {
				b.Fatalf("Failed to add: %v", err)
			}

			// Generate query
			query := make([]float32, dim)
			for i := range query {
				query[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := idx.Search(query, 10)
				if err != nil {
					b.Fatal(err)
				}
			}
			qps := float64(b.N) / b.Elapsed().Seconds()
			b.ReportMetric(qps, "QPS")
		})
	}
}

// BenchmarkDistanceTableComputation measures ADC table precomputation overhead
func BenchmarkDistanceTableComputation(b *testing.B) {
	dims := []int{128, 384, 768}

	for _, dim := range dims {
		b.Run(fmt.Sprintf("dim=%d", dim), func(b *testing.B) {
			m := dim / 8
			ksub := 256

			pq, _ := NewProductQuantizer(dim, m, ksub)

			// Train
			trainingData := make([]float32, 1000*dim)
			for i := range trainingData {
				trainingData[i] = rand.Float32()
			}
			pq.Train(trainingData, 15)

			query := make([]float32, dim)
			for i := range query {
				query[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = pq.ComputeDistanceTable(query)
			}
			// Report time in microseconds
			b.ReportMetric(float64(b.Elapsed().Microseconds())/float64(b.N), "us/table")
		})
	}
}

// =============================================================================
// QPS Comparison Test (Not a benchmark, but prints QPS for comparison)
// =============================================================================

func TestPQADC_QPS_Comparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping QPS comparison in short mode")
	}

	dim := 384
	numVectors := 50000
	numQueries := 100
	k := 10

	t.Logf("=== PQ-ADC QPS Comparison ===")
	t.Logf("Vectors: %d, Dimension: %d, Queries: %d", numVectors, dim, numQueries)

	// Generate data
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			queries[i][j] = rand.Float32()
		}
	}

	// Test PQ Index
	t.Run("PQIndex", func(t *testing.T) {
		idx, _ := NewPQIndex(PQIndexConfig{Dim: dim, M: 48, Ksub: 256})
		idx.Train(vectors)
		ids := make([]uint64, numVectors)
		for i := range ids {
			ids[i] = uint64(i)
		}
		idx.AddBatch(ids, vectors)

		start := time.Now()
		for _, q := range queries {
			idx.Search(q, k)
		}
		elapsed := time.Since(start)
		qps := float64(numQueries) / elapsed.Seconds()
		t.Logf("PQ Index: %.0f QPS (%.2f ms/query)", qps, elapsed.Seconds()*1000/float64(numQueries))
	})

	// Test IVF-PQ Index
	t.Run("IVFPQIndex", func(t *testing.T) {
		idx, _ := NewIVFPQIndex(IVFPQConfig{
			Dim:    dim,
			Nlist:  500,
			Nprobe: 10,
			M:      48,
			Ksub:   256,
		})
		idx.Train(vectors)
		ids := make([]uint64, numVectors)
		for i := range ids {
			ids[i] = uint64(i)
		}
		idx.AddBatch(ids, vectors)

		start := time.Now()
		for _, q := range queries {
			idx.Search(q, k)
		}
		elapsed := time.Since(start)
		qps := float64(numQueries) / elapsed.Seconds()
		t.Logf("IVF-PQ Index: %.0f QPS (%.2f ms/query)", qps, elapsed.Seconds()*1000/float64(numQueries))
	})
}

// =============================================================================
// Memory and Compression Tests
// =============================================================================

func TestPQIndex_Compression(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping compression test in short mode")
	}

	dim := 768
	numVectors := 10000

	idx, _ := NewPQIndex(PQIndexConfig{
		Dim:  dim,
		M:    96, // 96 subvectors = 96 bytes per vector
		Ksub: 256,
	})

	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	idx.Train(vectors)

	ids := make([]uint64, numVectors)
	for i := range ids {
		ids[i] = uint64(i)
	}
	idx.AddBatch(ids, vectors)

	stats := idx.Stats()
	ratio := idx.CompressionRatio()
	memMB := float64(idx.MemoryUsage()) / (1024 * 1024)
	originalMB := float64(numVectors*dim*4) / (1024 * 1024)

	t.Logf("PQ Compression Stats:")
	t.Logf("  Dimension: %d", dim)
	t.Logf("  Subvectors (M): %d", stats["num_subvectors"])
	t.Logf("  Bytes per vector: %d", stats["num_subvectors"])
	t.Logf("  Original size: %.2f MB", originalMB)
	t.Logf("  Compressed size: %.2f MB", memMB)
	t.Logf("  Compression ratio: %.1fx", ratio)

	// With M=96, we get 768*4/96 = 32x compression
	if ratio < 20 {
		t.Errorf("Expected compression ratio > 20x, got %.1fx", ratio)
	}
}
