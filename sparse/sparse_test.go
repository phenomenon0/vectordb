package sparse

import (
	"context"
	"math"
	"testing"
)

func TestNewSparseVector(t *testing.T) {
	tests := []struct {
		name        string
		indices     []uint32
		values      []float32
		dim         int
		expectError bool
	}{
		{
			name:        "valid sparse vector",
			indices:     []uint32{0, 5, 10},
			values:      []float32{1.0, 2.0, 3.0},
			dim:         20,
			expectError: false,
		},
		{
			name:        "empty sparse vector",
			indices:     []uint32{},
			values:      []float32{},
			dim:         10,
			expectError: false,
		},
		{
			name:        "length mismatch",
			indices:     []uint32{0, 1},
			values:      []float32{1.0},
			dim:         10,
			expectError: true,
		},
		{
			name:        "zero value",
			indices:     []uint32{0, 1},
			values:      []float32{1.0, 0.0},
			dim:         10,
			expectError: true,
		},
		{
			name:        "index exceeds dimension",
			indices:     []uint32{0, 11},
			values:      []float32{1.0, 2.0},
			dim:         10,
			expectError: true,
		},
		{
			name:        "unsorted indices",
			indices:     []uint32{10, 5, 0},
			values:      []float32{3.0, 2.0, 1.0},
			dim:         20,
			expectError: false, // Should auto-sort
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sv, err := NewSparseVector(tt.indices, tt.values, tt.dim)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if sv.Dim != tt.dim {
				t.Errorf("dimension mismatch: got %d, want %d", sv.Dim, tt.dim)
			}

			// Check sorted
			for i := 1; i < len(sv.Indices); i++ {
				if sv.Indices[i] <= sv.Indices[i-1] {
					t.Error("indices not sorted")
				}
			}
		})
	}
}

func TestFromDense(t *testing.T) {
	dense := []float32{1.0, 0.0, 0.0, 2.0, 0.0, 3.0}
	sv := FromDense(dense, 0.1)

	expectedNnz := 3
	if sv.Nnz() != expectedNnz {
		t.Errorf("nnz mismatch: got %d, want %d", sv.Nnz(), expectedNnz)
	}

	if sv.Dim != len(dense) {
		t.Errorf("dim mismatch: got %d, want %d", sv.Dim, len(dense))
	}

	// Check indices
	expectedIndices := []uint32{0, 3, 5}
	for i, idx := range sv.Indices {
		if idx != expectedIndices[i] {
			t.Errorf("index[%d] mismatch: got %d, want %d", i, idx, expectedIndices[i])
		}
	}
}

func TestToDense(t *testing.T) {
	indices := []uint32{0, 3, 5}
	values := []float32{1.0, 2.0, 3.0}
	sv, _ := NewSparseVector(indices, values, 10)

	dense := sv.ToDense()

	if len(dense) != 10 {
		t.Errorf("dense length mismatch: got %d, want %d", len(dense), 10)
	}

	expected := []float32{1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0}
	for i, v := range dense {
		if v != expected[i] {
			t.Errorf("value[%d] mismatch: got %f, want %f", i, v, expected[i])
		}
	}
}

func TestSparsity(t *testing.T) {
	indices := []uint32{0, 3, 5}
	values := []float32{1.0, 2.0, 3.0}
	sv, _ := NewSparseVector(indices, values, 10)

	expectedSparsity := 0.7 // 7 zeros out of 10
	sparsity := sv.Sparsity()

	if math.Abs(float64(sparsity-expectedSparsity)) > 0.01 {
		t.Errorf("sparsity mismatch: got %f, want %f", sparsity, expectedSparsity)
	}
}

func TestNormAndNormalize(t *testing.T) {
	indices := []uint32{0, 1}
	values := []float32{3.0, 4.0} // Norm = 5.0
	sv, _ := NewSparseVector(indices, values, 10)

	norm := sv.Norm()
	expectedNorm := float32(5.0)

	if math.Abs(float64(norm-expectedNorm)) > 0.01 {
		t.Errorf("norm mismatch: got %f, want %f", norm, expectedNorm)
	}

	sv.Normalize()
	normAfter := sv.Norm()

	if math.Abs(float64(normAfter-1.0)) > 0.01 {
		t.Errorf("normalized norm should be 1.0, got %f", normAfter)
	}
}

func TestDotProduct(t *testing.T) {
	// a = [1, 0, 2, 0, 3]
	a, _ := NewSparseVector([]uint32{0, 2, 4}, []float32{1.0, 2.0, 3.0}, 5)

	// b = [1, 0, 0, 0, 3]
	b, _ := NewSparseVector([]uint32{0, 4}, []float32{1.0, 3.0}, 5)

	// dot = 1*1 + 2*0 + 3*3 = 10
	dot := DotProduct(a, b)
	expectedDot := float32(10.0)

	if math.Abs(float64(dot-expectedDot)) > 0.01 {
		t.Errorf("dot product mismatch: got %f, want %f", dot, expectedDot)
	}
}

func TestCosineSimilarity(t *testing.T) {
	// a = [1, 0, 0]
	a, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 3)

	// b = [1, 0, 0]
	b, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 3)

	// Same vector, cosine = 1.0
	cosine := CosineSimilarity(a, b)
	if math.Abs(float64(cosine-1.0)) > 0.01 {
		t.Errorf("cosine should be 1.0 for identical vectors, got %f", cosine)
	}

	// c = [0, 1, 0]
	c, _ := NewSparseVector([]uint32{1}, []float32{1.0}, 3)

	// Orthogonal vectors, cosine = 0.0
	cosine = CosineSimilarity(a, c)
	if math.Abs(float64(cosine)) > 0.01 {
		t.Errorf("cosine should be 0.0 for orthogonal vectors, got %f", cosine)
	}
}

func TestAdd(t *testing.T) {
	// a = [1, 0, 2]
	a, _ := NewSparseVector([]uint32{0, 2}, []float32{1.0, 2.0}, 3)

	// b = [0, 3, 0]
	b, _ := NewSparseVector([]uint32{1}, []float32{3.0}, 3)

	// a + b = [1, 3, 2]
	sum, err := Add(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []float32{1.0, 3.0, 2.0}
	dense := sum.ToDense()

	for i, v := range expected {
		if math.Abs(float64(dense[i]-v)) > 0.01 {
			t.Errorf("value[%d] mismatch: got %f, want %f", i, dense[i], v)
		}
	}
}

func TestScale(t *testing.T) {
	sv, _ := NewSparseVector([]uint32{0, 2}, []float32{1.0, 2.0}, 3)
	sv.Scale(2.0)

	expected := []float32{2.0, 4.0}
	for i, v := range sv.Values {
		if math.Abs(float64(v-expected[i])) > 0.01 {
			t.Errorf("value[%d] mismatch after scale: got %f, want %f", i, v, expected[i])
		}
	}
}

func TestInvertedIndex_AddAndSearch(t *testing.T) {
	ctx := context.Background()
	idx := NewInvertedIndex(100)

	// Add documents
	// Doc 0: [1.0 at index 0, 2.0 at index 5]
	doc0, _ := NewSparseVector([]uint32{0, 5}, []float32{1.0, 2.0}, 100)
	err := idx.Add(ctx, 0, doc0)
	if err != nil {
		t.Fatalf("failed to add doc0: %v", err)
	}

	// Doc 1: [1.0 at index 0, 3.0 at index 10]
	doc1, _ := NewSparseVector([]uint32{0, 10}, []float32{1.0, 3.0}, 100)
	err = idx.Add(ctx, 1, doc1)
	if err != nil {
		t.Fatalf("failed to add doc1: %v", err)
	}

	// Doc 2: [2.0 at index 5]
	doc2, _ := NewSparseVector([]uint32{5}, []float32{2.0}, 100)
	err = idx.Add(ctx, 2, doc2)
	if err != nil {
		t.Fatalf("failed to add doc2: %v", err)
	}

	// Query: [1.0 at index 0] - should match doc0 and doc1
	query, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 100)
	results, err := idx.Search(ctx, query, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// Check that doc0 and doc1 are in results
	foundDocs := make(map[uint64]bool)
	for _, r := range results {
		foundDocs[r.DocID] = true
	}

	if !foundDocs[0] || !foundDocs[1] {
		t.Error("expected doc0 and doc1 in results")
	}
}

func TestInvertedIndex_BM25Scoring(t *testing.T) {
	ctx := context.Background()
	idx := NewInvertedIndex(100)

	// Add multiple documents with varying term frequencies
	for i := uint64(0); i < 10; i++ {
		// Each doc has term 0 with frequency 1.0
		doc, _ := NewSparseVector([]uint32{0, uint32(i + 1)}, []float32{1.0, float32(i + 1)}, 100)
		err := idx.Add(ctx, i, doc)
		if err != nil {
			t.Fatalf("failed to add doc %d: %v", i, err)
		}
	}

	// Query for term 0
	query, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 100)
	results, err := idx.Search(ctx, query, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results))
	}

	// Scores should be positive and sorted descending
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Error("results not sorted by score descending")
		}
		if results[i].Score <= 0 {
			t.Error("BM25 score should be positive")
		}
	}
}

func TestInvertedIndex_DotProductSearch(t *testing.T) {
	ctx := context.Background()
	idx := NewInvertedIndex(10)

	// Add docs
	doc0, _ := NewSparseVector([]uint32{0, 1}, []float32{1.0, 2.0}, 10)
	doc1, _ := NewSparseVector([]uint32{0, 2}, []float32{1.0, 3.0}, 10)

	idx.Add(ctx, 0, doc0)
	idx.Add(ctx, 1, doc1)

	// Query: [1.0, 2.0, 0, ...]
	query, _ := NewSparseVector([]uint32{0, 1}, []float32{1.0, 2.0}, 10)

	results, err := idx.SearchDotProduct(ctx, query, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Doc0: dot = 1*1 + 2*2 = 5
	// Doc1: dot = 1*1 + 0*2 = 1
	// Doc0 should rank higher

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results[0].DocID != 0 {
		t.Errorf("doc0 should rank first, got doc%d", results[0].DocID)
	}

	expectedScore := float32(5.0)
	if math.Abs(float64(results[0].Score-expectedScore)) > 0.01 {
		t.Errorf("score mismatch: got %f, want %f", results[0].Score, expectedScore)
	}
}

func TestInvertedIndex_CosineSearch(t *testing.T) {
	ctx := context.Background()
	idx := NewInvertedIndex(10)

	// Add identical normalized vectors
	doc0, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 10)
	doc1, _ := NewSparseVector([]uint32{0}, []float32{2.0}, 10) // Different magnitude

	idx.Add(ctx, 0, doc0)
	idx.Add(ctx, 1, doc1)

	query, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 10)

	results, err := idx.SearchCosine(ctx, query, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	// Both should have cosine = 1.0 (same direction)
	for _, r := range results {
		if math.Abs(float64(r.Score-1.0)) > 0.01 {
			t.Errorf("cosine should be 1.0, got %f for doc%d", r.Score, r.DocID)
		}
	}
}

func TestInvertedIndex_Delete(t *testing.T) {
	ctx := context.Background()
	idx := NewInvertedIndex(10)

	doc0, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 10)
	doc1, _ := NewSparseVector([]uint32{0}, []float32{2.0}, 10)

	idx.Add(ctx, 0, doc0)
	idx.Add(ctx, 1, doc1)

	if idx.Count() != 2 {
		t.Errorf("expected 2 docs, got %d", idx.Count())
	}

	// Delete doc0
	err := idx.Delete(ctx, 0)
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	if idx.Count() != 1 {
		t.Errorf("expected 1 doc after delete, got %d", idx.Count())
	}

	// Search should only return doc1
	query, _ := NewSparseVector([]uint32{0}, []float32{1.0}, 10)
	results, err := idx.Search(ctx, query, 10)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) != 1 || results[0].DocID != 1 {
		t.Error("expected only doc1 in results after delete")
	}
}

func TestInvertedIndex_Stats(t *testing.T) {
	ctx := context.Background()
	idx := NewInvertedIndex(100)

	// Add documents
	for i := uint64(0); i < 5; i++ {
		doc, _ := NewSparseVector([]uint32{0, 1, 2}, []float32{1.0, 2.0, 3.0}, 100)
		idx.Add(ctx, i, doc)
	}

	stats := idx.Stats()

	if stats.TotalDocs != 5 {
		t.Errorf("expected 5 docs, got %d", stats.TotalDocs)
	}

	if stats.TotalTerms != 3 {
		t.Errorf("expected 3 terms, got %d", stats.TotalTerms)
	}

	if stats.MemoryUsage <= 0 {
		t.Error("memory usage should be positive")
	}
}

func BenchmarkSparseVectorCreation(b *testing.B) {
	indices := make([]uint32, 100)
	values := make([]float32, 100)
	for i := range indices {
		indices[i] = uint32(i * 10)
		values[i] = float32(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = NewSparseVector(indices, values, 10000)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	indices := make([]uint32, 100)
	values := make([]float32, 100)
	for i := range indices {
		indices[i] = uint32(i * 10)
		values[i] = float32(i + 1)
	}

	vec, _ := NewSparseVector(indices, values, 10000)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = DotProduct(vec, vec)
	}
}

func BenchmarkInvertedIndexSearch(b *testing.B) {
	ctx := context.Background()
	idx := NewInvertedIndex(10000)

	// Add 1000 documents
	for i := uint64(0); i < 1000; i++ {
		indices := []uint32{uint32(i % 100), uint32((i + 1) % 100), uint32((i + 2) % 100)}
		values := []float32{1.0, 2.0, 3.0}
		doc, _ := NewSparseVector(indices, values, 10000)
		idx.Add(ctx, i, doc)
	}

	query, _ := NewSparseVector([]uint32{0, 1, 2}, []float32{1.0, 1.0, 1.0}, 10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(ctx, query, 10)
	}
}
