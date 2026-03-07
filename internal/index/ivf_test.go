package index

import (
	"context"
	"math"
	"math/rand"
	"testing"
)

func TestIVFBasicOperations(t *testing.T) {
	dim := 128
	nlist := 10
	nprobe := 3

	config := map[string]interface{}{
		"nlist":  nlist,
		"nprobe": nprobe,
		"metric": "cosine",
	}

	idx, err := NewIVFIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create IVF index: %v", err)
	}

	// Add vectors (need at least 10x nlist for auto-training)
	numVectors := nlist * 10
	vectors := make(map[uint64][]float32)

	for i := 0; i < numVectors; i++ {
		id := uint64(i)
		vec := randomVector(dim)
		vectors[id] = vec

		err := idx.Add(context.Background(), id, vec)
		if err != nil {
			t.Fatalf("failed to add vector %d: %v", id, err)
		}
	}

	// Check stats
	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Errorf("expected count %d, got %d", numVectors, stats.Count)
	}

	if stats.Name != "IVF" {
		t.Errorf("expected name IVF, got %s", stats.Name)
	}

	// Check that centroids were trained
	centroidsTrained, ok := stats.Extra["centroids_ready"].(bool)
	if !ok || !centroidsTrained {
		t.Error("centroids should be trained after adding enough vectors")
	}

	// Search
	query := randomVector(dim)
	results, err := idx.Search(context.Background(), query, 10, IVFSearchParams{NProbe: nprobe})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results")
	}

	if len(results) > 10 {
		t.Errorf("expected at most 10 results, got %d", len(results))
	}

	// Verify results are sorted by distance
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Error("results not sorted by distance")
		}
	}

	// Delete a vector
	deleteID := uint64(0)
	err = idx.Delete(context.Background(), deleteID)
	if err != nil {
		t.Fatalf("failed to delete vector: %v", err)
	}

	// Search should not return deleted vector
	results, _ = idx.Search(context.Background(), vectors[deleteID], 10, IVFSearchParams{NProbe: nprobe})
	for _, r := range results {
		if r.ID == deleteID {
			t.Error("deleted vector found in search results")
		}
	}
}

func TestIVFExportImport(t *testing.T) {
	dim := 64
	nlist := 5
	numVectors := 50

	config := map[string]interface{}{
		"nlist":  nlist,
		"nprobe": 2,
		"metric": "euclidean",
	}

	// Create and populate index
	idx1, err := NewIVFIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	vectors := make(map[uint64][]float32)
	for i := 0; i < numVectors; i++ {
		id := uint64(i)
		vec := randomVector(dim)
		vectors[id] = vec
		idx1.Add(context.Background(), id, vec)
	}

	// Export
	data, err := idx1.Export()
	if err != nil {
		t.Fatalf("export failed: %v", err)
	}

	// Import into new index
	idx2, err := NewIVFIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index 2: %v", err)
	}

	err = idx2.Import(data)
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}

	// Compare stats
	stats1 := idx1.Stats()
	stats2 := idx2.Stats()

	if stats1.Count != stats2.Count {
		t.Errorf("count mismatch: %d vs %d", stats1.Count, stats2.Count)
	}

	if stats1.Dim != stats2.Dim {
		t.Errorf("dimension mismatch: %d vs %d", stats1.Dim, stats2.Dim)
	}

	// Compare search results
	query := randomVector(dim)
	results1, _ := idx1.Search(context.Background(), query, 5, IVFSearchParams{NProbe: 2})
	results2, _ := idx2.Search(context.Background(), query, 5, IVFSearchParams{NProbe: 2})

	if len(results1) != len(results2) {
		t.Errorf("result count mismatch: %d vs %d", len(results1), len(results2))
	}

	for i := range results1 {
		if results1[i].ID != results2[i].ID {
			t.Errorf("result %d ID mismatch: %d vs %d", i, results1[i].ID, results2[i].ID)
		}

		// Distances should be very close
		distDiff := math.Abs(float64(results1[i].Distance - results2[i].Distance))
		if distDiff > 1e-5 {
			t.Errorf("result %d distance mismatch: %f vs %f", i, results1[i].Distance, results2[i].Distance)
		}
	}
}

func TestIVFSearchParameters(t *testing.T) {
	dim := 128
	nlist := 20

	config := map[string]interface{}{
		"nlist":  nlist,
		"nprobe": 5,
		"metric": "cosine",
	}

	idx, _ := NewIVFIndex(dim, config)

	// Add vectors
	numVectors := nlist * 10
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	query := randomVector(dim)

	// Test different nprobe values
	for nprobe := 1; nprobe <= nlist; nprobe *= 2 {
		results, err := idx.Search(context.Background(), query, 10, IVFSearchParams{NProbe: nprobe})
		if err != nil {
			t.Fatalf("search with nprobe=%d failed: %v", nprobe, err)
		}

		t.Logf("nprobe=%d: found %d results", nprobe, len(results))

		// Higher nprobe should find at least as many results (up to k)
		// (This is a weak condition but tests parameter handling)
		if len(results) > 10 {
			t.Errorf("nprobe=%d returned too many results: %d", nprobe, len(results))
		}
	}
}

func TestIVFBruteForceSearch(t *testing.T) {
	// Test search before centroids are trained (should fall back to brute force)
	dim := 64
	config := map[string]interface{}{
		"nlist":  10,
		"nprobe": 3,
	}

	idx, _ := NewIVFIndex(dim, config)

	// Add only a few vectors (not enough to trigger training)
	numVectors := 5
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Search should work even without trained centroids
	query := randomVector(dim)
	results, err := idx.Search(context.Background(), query, 3, IVFSearchParams{})
	if err != nil {
		t.Fatalf("brute force search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results from brute force")
	}
}

func TestKMeans(t *testing.T) {
	dim := 10
	k := 3
	numVectors := 100

	// Generate vectors
	vectors := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		vectors[i] = randomVector(dim)
	}

	// Run k-means
	centroids, err := kMeans(vectors, k, dim, 10)
	if err != nil {
		t.Fatalf("k-means failed: %v", err)
	}

	if len(centroids) != k {
		t.Errorf("expected %d centroids, got %d", k, len(centroids))
	}

	// Check centroid dimensions
	for i, centroid := range centroids {
		if len(centroid) != dim {
			t.Errorf("centroid %d has wrong dimension: expected %d, got %d", i, dim, len(centroid))
		}
	}

	// Verify centroids are distinct
	for i := 0; i < k; i++ {
		for j := i + 1; j < k; j++ {
			dist := euclideanDistance(centroids[i], centroids[j])
			if dist < 1e-6 {
				t.Errorf("centroids %d and %d are too similar", i, j)
			}
		}
	}
}

func TestIVFClusterBalance(t *testing.T) {
	dim := 64
	nlist := 10

	config := map[string]interface{}{
		"nlist":  nlist,
		"nprobe": 3,
	}

	idx, _ := NewIVFIndex(dim, config)

	// Add many vectors
	numVectors := nlist * 100
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	stats := idx.Stats()

	// Check cluster balance
	balance, ok := stats.Extra["cluster_balance"].(float64)
	if !ok {
		t.Error("cluster balance not found in stats")
	}

	avgSize, ok := stats.Extra["avg_cluster_size"].(float64)
	if !ok {
		t.Error("avg cluster size not found in stats")
	}

	t.Logf("Cluster balance (std dev): %.2f", balance)
	t.Logf("Average cluster size: %.2f", avgSize)

	// Expect reasonable balance (std dev < 50% of average)
	if balance > avgSize*0.5 {
		t.Logf("Warning: clusters may be imbalanced (std dev %.2f vs avg %.2f)", balance, avgSize)
	}
}

func TestIVFDimensionMismatch(t *testing.T) {
	dim := 128

	idx, _ := NewIVFIndex(dim, map[string]interface{}{"nlist": 10})

	// Try to add vector with wrong dimension
	wrongVec := randomVector(64)
	err := idx.Add(context.Background(), 1, wrongVec)
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}

	// Add correct vector
	correctVec := randomVector(dim)
	idx.Add(context.Background(), 1, correctVec)

	// Try to search with wrong dimension
	wrongQuery := randomVector(64)
	_, err = idx.Search(context.Background(), wrongQuery, 5, IVFSearchParams{})
	if err == nil {
		t.Error("expected error for query dimension mismatch")
	}
}

func BenchmarkIVFAdd(b *testing.B) {
	dim := 128
	config := map[string]interface{}{
		"nlist":  100,
		"nprobe": 10,
	}

	idx, _ := NewIVFIndex(dim, config)

	// Pre-generate vectors
	vectors := make([][]float32, b.N)
	for i := 0; i < b.N; i++ {
		vectors[i] = randomVector(dim)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Add(context.Background(), uint64(i), vectors[i])
	}
}

func BenchmarkIVFSearch(b *testing.B) {
	dim := 128
	nlist := 100
	numVectors := 10000

	config := map[string]interface{}{
		"nlist":  nlist,
		"nprobe": 10,
	}

	idx, _ := NewIVFIndex(dim, config)

	// Populate index
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	query := randomVector(dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(context.Background(), query, 10, IVFSearchParams{NProbe: 10})
	}
}

// Helper function
func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // Range [-1, 1]
	}
	return vec
}
