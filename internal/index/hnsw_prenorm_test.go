package index

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/phenomenon0/vectordb/internal/index/simd"
)

// TestHNSWPrenormalization validates that pre-normalized vector storage works correctly.
func TestHNSWPrenormalization(t *testing.T) {
	dim := 128
	ctx := context.Background()

	t.Run("StoredVectorsAreUnitNorm", func(t *testing.T) {
		idx, err := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize": true,
		})
		if err != nil {
			t.Fatalf("create: %v", err)
		}

		// Insert a non-unit vector
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(i + 1) // not unit norm
		}
		origNorm := l2norm(vec)
		if math.Abs(float64(origNorm)-1.0) < 0.01 {
			t.Fatal("test vector should not already be unit norm")
		}

		if err := idx.Add(ctx, 1, vec); err != nil {
			t.Fatalf("add: %v", err)
		}

		// Check the stored vector is unit norm
		h := idx.(*HNSWIndex)
		stored := h.vectors[1]
		storedNorm := l2norm(stored)
		if math.Abs(float64(storedNorm)-1.0) > 0.001 {
			t.Errorf("stored vector norm = %.6f, want ~1.0", storedNorm)
		}

		// Original vector should NOT be modified (we copy before normalizing)
		origAfter := l2norm(vec)
		if math.Abs(float64(origAfter)-float64(origNorm)) > 0.001 {
			t.Errorf("original vector was mutated: norm %.6f -> %.6f", origNorm, origAfter)
		}
	})

	t.Run("SearchFindsCorrectNeighbors", func(t *testing.T) {
		idx, err := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize": true,
			"ef_search":    64,
		})
		if err != nil {
			t.Fatalf("create: %v", err)
		}

		rng := rand.New(rand.NewSource(42))

		// Insert 200 random vectors
		vecs := make([][]float32, 200)
		for i := range vecs {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			vecs[i] = v
			if err := idx.Add(ctx, uint64(i), v); err != nil {
				t.Fatalf("add %d: %v", i, err)
			}
		}

		// Search for vec[0] — it should be its own nearest neighbor
		results, err := idx.Search(ctx, vecs[0], 5, DefaultSearchParams{})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(results) == 0 {
			t.Fatal("no results")
		}
		if results[0].ID != 0 {
			t.Errorf("expected first result ID=0, got ID=%d", results[0].ID)
		}

		// Verify distances are non-negative and ordered
		for i := 1; i < len(results); i++ {
			if results[i].Distance < results[i-1].Distance {
				t.Errorf("results not sorted: dist[%d]=%.4f < dist[%d]=%.4f",
					i, results[i].Distance, i-1, results[i-1].Distance)
			}
		}
	})

	t.Run("PrenormDisabledStillWorks", func(t *testing.T) {
		idx, err := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize": false,
		})
		if err != nil {
			t.Fatalf("create: %v", err)
		}

		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(i + 1)
		}
		origNorm := l2norm(vec)

		if err := idx.Add(ctx, 1, vec); err != nil {
			t.Fatalf("add: %v", err)
		}

		// Stored vector should NOT be normalized
		h := idx.(*HNSWIndex)
		stored := h.vectors[1]
		storedNorm := l2norm(stored)
		if math.Abs(float64(storedNorm)-float64(origNorm)) > 0.001 {
			t.Errorf("stored norm = %.6f, want original norm %.6f (prenormalize=false)", storedNorm, origNorm)
		}

		// Search should still work
		results, err := idx.Search(ctx, vec, 1, DefaultSearchParams{})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(results) == 0 || results[0].ID != 1 {
			t.Error("search with prenormalize=false failed to find inserted vector")
		}
	})

	t.Run("BatchAddNormalizesVectors", func(t *testing.T) {
		idx, err := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize": true,
		})
		if err != nil {
			t.Fatalf("create: %v", err)
		}

		vecs := make(map[uint64][]float32)
		for i := 0; i < 50; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32((i+1)*dim + j)
			}
			vecs[uint64(i)] = v
		}

		if err := idx.(*HNSWIndex).BatchAdd(ctx, vecs); err != nil {
			t.Fatalf("batch add: %v", err)
		}

		h := idx.(*HNSWIndex)
		for id := range vecs {
			stored := h.vectors[id]
			norm := l2norm(stored)
			if math.Abs(float64(norm)-1.0) > 0.001 {
				t.Errorf("vector %d norm = %.6f after BatchAdd, want ~1.0", id, norm)
				break
			}
		}
	})

	t.Run("ExportImportPreservesConfig", func(t *testing.T) {
		idx1, err := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize": true,
		})
		if err != nil {
			t.Fatalf("create: %v", err)
		}

		rng := rand.New(rand.NewSource(99))
		for i := 0; i < 20; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			if err := idx1.Add(ctx, uint64(i), v); err != nil {
				t.Fatalf("add: %v", err)
			}
		}

		data, err := idx1.Export()
		if err != nil {
			t.Fatalf("export: %v", err)
		}

		idx2, err := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize": false, // deliberately different
		})
		if err != nil {
			t.Fatalf("create idx2: %v", err)
		}

		if err := idx2.Import(data); err != nil {
			t.Fatalf("import: %v", err)
		}

		// After import, prenormalized should be true (from export data)
		h2 := idx2.(*HNSWIndex)
		if !h2.prenormalized {
			t.Error("import should restore prenormalized=true from exported config")
		}

		// Vectors should still be unit norm after import
		for id := range h2.vectors {
			norm := l2norm(h2.vectors[id])
			if math.Abs(float64(norm)-1.0) > 0.01 {
				t.Errorf("imported vector %d norm = %.6f, want ~1.0", id, norm)
				break
			}
		}

		// Search should work on imported index
		query := make([]float32, dim)
		for i := range query {
			query[i] = rng.Float32()*2 - 1
		}
		results, err := idx2.Search(ctx, query, 5, DefaultSearchParams{})
		if err != nil {
			t.Fatalf("search after import: %v", err)
		}
		if len(results) == 0 {
			t.Error("no results from imported prenormalized index")
		}
	})

	t.Run("PrenormMatchesFullCosine", func(t *testing.T) {
		// Verify that prenormalized distance produces equivalent ranking to full cosine
		rng := rand.New(rand.NewSource(77))

		idxPre, _ := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize":    true,
			"ef_search":       128,
			"ef_construction": 200,
		})
		idxFull, _ := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize":    false,
			"ef_search":       128,
			"ef_construction": 200,
		})

		// Insert same vectors into both
		for i := 0; i < 500; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			idxPre.Add(ctx, uint64(i), v)
			idxFull.Add(ctx, uint64(i), v)
		}

		// Query and compare top-10 results
		query := make([]float32, dim)
		for i := range query {
			query[i] = rng.Float32()*2 - 1
		}

		resPre, _ := idxPre.Search(ctx, query, 10, HNSWSearchParams{EfSearch: 128})
		resFull, _ := idxFull.Search(ctx, query, 10, HNSWSearchParams{EfSearch: 128})

		if len(resPre) == 0 || len(resFull) == 0 {
			t.Fatal("both indexes should return results")
		}

		// Compute overlap: how many of the same IDs appear in both result sets
		fullIDs := make(map[uint64]bool)
		for _, r := range resFull {
			fullIDs[r.ID] = true
		}
		overlap := 0
		for _, r := range resPre {
			if fullIDs[r.ID] {
				overlap++
			}
		}

		// Due to non-deterministic HNSW construction, we can't expect perfect overlap,
		// but with ef_search=128 and 500 vectors, overlap should be high (>= 7/10)
		minOverlap := 7
		if overlap < minOverlap {
			t.Errorf("prenorm vs full cosine overlap = %d/%d, want >= %d",
				overlap, len(resPre), minOverlap)
			t.Logf("prenorm results: %v", idsOf(resPre))
			t.Logf("full results:    %v", idsOf(resFull))
		}
	})

	t.Run("ResurrectionNormalizesVector", func(t *testing.T) {
		idx, err := NewHNSWIndex(dim, map[string]interface{}{
			"prenormalize": true,
		})
		if err != nil {
			t.Fatalf("create: %v", err)
		}

		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = float32(i + 1)
		}

		if err := idx.Add(ctx, 1, vec); err != nil {
			t.Fatalf("add: %v", err)
		}
		if err := idx.Delete(ctx, 1); err != nil {
			t.Fatalf("delete: %v", err)
		}

		// Resurrect with new (non-unit) vector
		newVec := make([]float32, dim)
		for i := range newVec {
			newVec[i] = float32(i*2 + 5)
		}
		if err := idx.Add(ctx, 1, newVec); err != nil {
			t.Fatalf("resurrect: %v", err)
		}

		h := idx.(*HNSWIndex)
		stored := h.vectors[1]
		norm := l2norm(stored)
		if math.Abs(float64(norm)-1.0) > 0.001 {
			t.Errorf("resurrected vector norm = %.6f, want ~1.0", norm)
		}
	})

	t.Run("NormalizedCosineDistanceAccuracy", func(t *testing.T) {
		// Verify NormalizedCosineDistanceF32 matches CosineDistanceF32 on unit vectors
		rng := rand.New(rand.NewSource(123))

		for trial := 0; trial < 100; trial++ {
			a := make([]float32, dim)
			b := make([]float32, dim)
			for i := range a {
				a[i] = rng.Float32()*2 - 1
				b[i] = rng.Float32()*2 - 1
			}
			simd.NormalizeF32(a)
			simd.NormalizeF32(b)

			full := simd.CosineDistanceF32(a, b)
			norm := simd.NormalizedCosineDistanceF32(a, b)

			diff := math.Abs(float64(full) - float64(norm))
			if diff > 0.001 {
				t.Errorf("trial %d: full cosine=%.6f, normalized=%.6f, diff=%.6f",
					trial, full, norm, diff)
			}
		}
	})
}

func l2norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

func idsOf(results []Result) []uint64 {
	ids := make([]uint64, len(results))
	for i, r := range results {
		ids[i] = r.ID
	}
	return ids
}
