package index

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// =============================================================================
// PQ4 Unit Tests
// =============================================================================

func TestPQ4Index_Basic(t *testing.T) {
	dim := 128
	numVectors := 1000
	if testing.Short() {
		numVectors = 300 // Minimum for k-means training
	}
	k := 10

	// Create index (m must be even and divide dim)
	idx, err := NewPQ4Index(PQ4IndexConfig{
		Dim: dim,
		M:   32, // 32 subvectors, 4 dims each
	})
	if err != nil {
		t.Fatalf("Failed to create PQ4 index: %v", err)
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
	query := vectors[:dim]
	results, err := idx.Search(query, k)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(results) != k {
		t.Errorf("Expected %d results, got %d", k, len(results))
	}

	// Print stats
	stats := idx.Stats()
	t.Logf("PQ4 Index Stats: %+v", stats)
	t.Logf("Compression Ratio: %.1fx", idx.CompressionRatio())
	t.Logf("Memory Usage: %.2f MB", float64(idx.MemoryUsage())/(1024*1024))
}

func TestPQ4_Compression(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping compression test in short mode")
	}

	dim := 768
	numVectors := 10000

	idx, err := NewPQ4Index(PQ4IndexConfig{
		Dim: dim,
		M:   192, // 192 subvectors = 96 bytes per vector (vs 192 for PQ8)
	})
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

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

	t.Logf("PQ4 Compression Stats:")
	t.Logf("  Dimension: %d", dim)
	t.Logf("  Subvectors (M): %d", stats["num_subvectors"])
	t.Logf("  Bytes per vector: %d", stats["bytes_per_vector"])
	t.Logf("  Original size: %.2f MB", originalMB)
	t.Logf("  Compressed size: %.2f MB", memMB)
	t.Logf("  Compression ratio: %.1fx", ratio)

	// PQ4 with M=192 gives 768*4/96 = 32x compression (same as PQ8 with M=96)
	// But uses half the storage (96 bytes vs 192 bytes for same M)
	if ratio < 25 {
		t.Errorf("Expected compression ratio > 25x, got %.1fx", ratio)
	}
}

func TestPQ4_PackedCodes(t *testing.T) {
	dim := 64
	m := 16

	pq, err := NewPQ4Quantizer(dim, m)
	if err != nil {
		t.Fatalf("Failed to create quantizer: %v", err)
	}

	// Train
	trainingData := make([]float32, 1000*dim)
	for i := range trainingData {
		trainingData[i] = rand.Float32()
	}
	pq.Train(trainingData, 15)

	// Test single vector
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = rand.Float32()
	}

	// Quantize packed
	packed, err := pq.Quantize(vector)
	if err != nil {
		t.Fatalf("Packed quantize failed: %v", err)
	}

	// Quantize unpacked
	unpacked, err := pq.QuantizeUnpacked(vector)
	if err != nil {
		t.Fatalf("Unpacked quantize failed: %v", err)
	}

	// Verify sizes
	expectedPackedSize := m / 2
	if len(packed) != expectedPackedSize {
		t.Errorf("Packed size: expected %d, got %d", expectedPackedSize, len(packed))
	}
	if len(unpacked) != m {
		t.Errorf("Unpacked size: expected %d, got %d", m, len(unpacked))
	}

	// Verify packing matches unpacking
	for j := 0; j < m; j += 2 {
		code1 := int(packed[j/2] & 0x0F)
		code2 := int((packed[j/2] >> 4) & 0x0F)

		if code1 != int(unpacked[j]) {
			t.Errorf("Code mismatch at %d: packed=%d, unpacked=%d", j, code1, unpacked[j])
		}
		if code2 != int(unpacked[j+1]) {
			t.Errorf("Code mismatch at %d: packed=%d, unpacked=%d", j+1, code2, unpacked[j+1])
		}
	}

	// Verify dequantization
	reconstructed, err := pq.Dequantize(packed)
	if err != nil {
		t.Fatalf("Dequantize failed: %v", err)
	}
	if len(reconstructed) != dim {
		t.Errorf("Reconstructed dimension: expected %d, got %d", dim, len(reconstructed))
	}
}

// =============================================================================
// PQ4 Benchmarks
// =============================================================================

// BenchmarkPQ4_vs_PQ8 compares PQ4 and PQ8 performance
func BenchmarkPQ4_vs_PQ8(b *testing.B) {
	dim := 384
	numVectors := 10000

	// Generate vectors
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	query := make([]float32, dim)
	for i := range query {
		query[i] = rand.Float32()
	}
	ids := make([]uint64, numVectors)
	for i := range ids {
		ids[i] = uint64(i)
	}

	b.Run("PQ4_M=96", func(b *testing.B) {
		idx, _ := NewPQ4Index(PQ4IndexConfig{Dim: dim, M: 96})
		idx.Train(vectors)
		idx.AddBatch(ids, vectors)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx.Search(query, 10)
		}
		qps := float64(b.N) / b.Elapsed().Seconds()
		b.ReportMetric(qps, "QPS")
		b.ReportMetric(float64(idx.pq.BytesPerVector()), "bytes/vec")
	})

	b.Run("PQ8_M=48", func(b *testing.B) {
		idx, _ := NewPQIndex(PQIndexConfig{Dim: dim, M: 48, Ksub: 256})
		idx.Train(vectors)
		idx.AddBatch(ids, vectors)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx.Search(query, 10)
		}
		qps := float64(b.N) / b.Elapsed().Seconds()
		b.ReportMetric(qps, "QPS")
		b.ReportMetric(float64(48), "bytes/vec")
	})
}

// BenchmarkPQ4Index_Search benchmarks PQ4 search at various scales
func BenchmarkPQ4Index_Search(b *testing.B) {
	scales := []int{10000, 50000, 100000}
	dim := 384

	for _, numVectors := range scales {
		b.Run(fmt.Sprintf("n=%d", numVectors), func(b *testing.B) {
			idx, _ := NewPQ4Index(PQ4IndexConfig{
				Dim: dim,
				M:   96, // 96 subvectors = 48 bytes per vector
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

			query := make([]float32, dim)
			for i := range query {
				query[i] = rand.Float32()
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
			qps := float64(b.N) / b.Elapsed().Seconds()
			b.ReportMetric(qps, "QPS")
		})
	}
}

// BenchmarkPQ4_DistanceTableLookup benchmarks just the distance table lookups
func BenchmarkPQ4_DistanceTableLookup(b *testing.B) {
	dim := 384
	m := 96
	numVectors := 10000

	pq, _ := NewPQ4Quantizer(dim, m)
	vectors := make([]float32, numVectors*dim)
	for i := range vectors {
		vectors[i] = rand.Float32()
	}
	pq.Train(vectors, 15)

	codes, _ := pq.Quantize(vectors)
	query := make([]float32, dim)
	for i := range query {
		query[i] = rand.Float32()
	}

	dt, _ := pq.ComputeDistanceTable(query)

	b.Run("Packed", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = dt.LookupDistanceBatchPacked(codes, numVectors)
		}
		b.ReportMetric(float64(numVectors*b.N)/b.Elapsed().Seconds()/1e6, "Mvecs/s")
	})
}

// =============================================================================
// QPS Comparison Test
// =============================================================================

func TestPQ4_QPS_Comparison(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping QPS comparison in short mode")
	}

	dim := 384
	numVectors := 50000
	numQueries := 100
	k := 10

	t.Logf("=== PQ4 vs PQ8 QPS Comparison ===")
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
	ids := make([]uint64, numVectors)
	for i := range ids {
		ids[i] = uint64(i)
	}

	// Test PQ4 Index
	t.Run("PQ4Index", func(t *testing.T) {
		idx, _ := NewPQ4Index(PQ4IndexConfig{Dim: dim, M: 96})
		idx.Train(vectors)
		idx.AddBatch(ids, vectors)

		start := time.Now()
		for _, q := range queries {
			idx.Search(q, k)
		}
		elapsed := time.Since(start)
		qps := float64(numQueries) / elapsed.Seconds()
		t.Logf("PQ4 Index (M=96): %.0f QPS (%.2f ms/query), %d bytes/vec, %.1fx compression",
			qps, elapsed.Seconds()*1000/float64(numQueries),
			idx.pq.BytesPerVector(), idx.CompressionRatio())
	})

	// Test PQ8 Index
	t.Run("PQ8Index", func(t *testing.T) {
		idx, _ := NewPQIndex(PQIndexConfig{Dim: dim, M: 48, Ksub: 256})
		idx.Train(vectors)
		idx.AddBatch(ids, vectors)

		start := time.Now()
		for _, q := range queries {
			idx.Search(q, k)
		}
		elapsed := time.Since(start)
		qps := float64(numQueries) / elapsed.Seconds()
		t.Logf("PQ8 Index (M=48): %.0f QPS (%.2f ms/query), %d bytes/vec, %.1fx compression",
			qps, elapsed.Seconds()*1000/float64(numQueries), 48, idx.CompressionRatio())
	})
}
