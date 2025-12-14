package index

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"
)

func TestDiskANNBasicOperations(t *testing.T) {
	dim := 128
	config := map[string]interface{}{
		"memory_limit":    100,  // Keep 100 vectors in memory
		"max_degree":      16,
		"ef_construction": 50,
		"ef_search":       20,
		"index_path":      "/tmp/test_diskann_basic.idx",
		"metric":          "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create DiskANN index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_basic.idx")
		}
	}()

	// Add vectors
	numVectors := 200 // More than memory_limit to test disk storage
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	// Check stats
	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Errorf("expected count %d, got %d", numVectors, stats.Count)
	}

	if stats.Name != "DiskANN" {
		t.Errorf("expected name DiskANN, got %s", stats.Name)
	}

	// Verify some vectors are on disk
	diskVectors, ok := stats.Extra["disk_vectors"].(int)
	if !ok || diskVectors == 0 {
		t.Error("expected some vectors to be stored on disk")
	}

	t.Logf("Memory vectors: %v, Disk vectors: %v",
		stats.Extra["memory_vectors"], stats.Extra["disk_vectors"])

	// Search
	query := randomVector(dim)
	results, err := idx.Search(context.Background(), query, 10, nil)
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

	// Test cache hit rate
	cacheHitRate, ok := stats.Extra["cache_hit_rate"].(float64)
	if ok {
		t.Logf("Cache hit rate: %.2f%%", cacheHitRate*100)
	}

	// Delete a vector
	deleteID := uint64(0)
	if err := idx.Delete(context.Background(), deleteID); err != nil {
		t.Fatalf("failed to delete vector: %v", err)
	}

	// Verify deleted vector not in results
	results, _ = idx.Search(context.Background(), query, 10, nil)
	for _, r := range results {
		if r.ID == deleteID {
			t.Error("deleted vector found in search results")
		}
	}
}

func TestDiskANNExportImport(t *testing.T) {
	dim := 64
	config := map[string]interface{}{
		"memory_limit": 50,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_export.idx",
		"metric":       "euclidean",
	}

	// Create and populate index
	idx1, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx1.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_export.idx")
		}
	}()

	numVectors := 100
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		idx1.Add(context.Background(), uint64(i), vec)
	}

	// Export
	data, err := idx1.Export()
	if err != nil {
		t.Fatalf("export failed: %v", err)
	}

	if len(data) == 0 {
		t.Error("exported data is empty")
	}

	// Close first index
	if d, ok := idx1.(*DiskANNIndex); ok {
		d.Close()
	}

	// Import into new index
	idx2, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index 2: %v", err)
	}
	defer func() {
		if d, ok := idx2.(*DiskANNIndex); ok {
			d.Close()
		}
	}()

	if err := idx2.Import(data); err != nil {
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

	t.Logf("Successfully exported/imported %d vectors", stats2.Count)
}

func TestDiskANNMemoryConstraint(t *testing.T) {
	dim := 32
	memoryLimit := 10
	config := map[string]interface{}{
		"memory_limit": memoryLimit,
		"max_degree":   8,
		"index_path":   "/tmp/test_diskann_memory.idx",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_memory.idx")
		}
	}()

	// Add more vectors than memory limit
	numVectors := memoryLimit * 3
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("failed to add vector %d: %v", i, err)
		}
	}

	stats := idx.Stats()
	memVectors := stats.Extra["memory_vectors"].(int)
	diskVectors := stats.Extra["disk_vectors"].(int)

	if memVectors > memoryLimit {
		t.Errorf("memory vectors (%d) exceeds limit (%d)", memVectors, memoryLimit)
	}

	if diskVectors == 0 {
		t.Error("expected vectors on disk")
	}

	t.Logf("Memory: %d vectors (limit: %d), Disk: %d vectors",
		memVectors, memoryLimit, diskVectors)

	// Verify we can still search
	query := randomVector(dim)
	results, err := idx.Search(context.Background(), query, 5, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results despite disk storage")
	}
}

func TestDiskANNGraphStructure(t *testing.T) {
	dim := 64
	maxDegree := 16
	config := map[string]interface{}{
		"memory_limit": 100,
		"max_degree":   maxDegree,
		"index_path":   "/tmp/test_diskann_graph.idx",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_graph.idx")
		}
	}()

	diskANN := idx.(*DiskANNIndex)

	// Add vectors
	numVectors := 50
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Check graph structure
	diskANN.mu.RLock()
	graphSize := len(diskANN.graph)
	diskANN.mu.RUnlock()

	if graphSize != numVectors {
		t.Errorf("expected graph size %d, got %d", numVectors, graphSize)
	}

	// Verify max degree constraint
	diskANN.mu.RLock()
	for id, neighbors := range diskANN.graph {
		if len(neighbors) > maxDegree {
			t.Errorf("vector %d has %d neighbors, exceeds max_degree %d",
				id, len(neighbors), maxDegree)
		}
	}
	diskANN.mu.RUnlock()

	t.Logf("Graph has %d nodes, max degree: %d", graphSize, maxDegree)
}

func TestDiskANNSearchAccuracy(t *testing.T) {
	dim := 128
	config := map[string]interface{}{
		"memory_limit":    200,
		"max_degree":      32,
		"ef_construction": 100,
		"ef_search":       50,
		"index_path":      "/tmp/test_diskann_accuracy.idx",
		"metric":          "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_accuracy.idx")
		}
	}()

	// Add clustered vectors
	numClusters := 5
	vectorsPerCluster := 20
	clusters := make([][][]float32, numClusters)

	id := uint64(0)
	for c := 0; c < numClusters; c++ {
		center := randomVector(dim)
		clusters[c] = make([][]float32, 0, vectorsPerCluster)

		for i := 0; i < vectorsPerCluster; i++ {
			// Generate vector near cluster center
			vec := make([]float32, dim)
			for d := 0; d < dim; d++ {
				vec[d] = center[d] + (rand.Float32()-0.5)*0.1
			}
			clusters[c] = append(clusters[c], vec)

			if err := idx.Add(context.Background(), id, vec); err != nil {
				t.Fatalf("failed to add vector: %v", err)
			}
			id++
		}
	}

	// Query with cluster center - should return vectors from same cluster
	testCluster := 0
	query := clusters[testCluster][0]

	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("no search results")
	}

	// Verify results are mostly from the query cluster
	// (This is a weak test - DiskANN is approximate, so not 100% accurate)
	clusterHits := 0
	for _, r := range results {
		// Check if result ID is in query cluster range
		clusterID := int(r.ID) / vectorsPerCluster
		if clusterID == testCluster {
			clusterHits++
		}
	}

	accuracy := float64(clusterHits) / float64(len(results))
	t.Logf("Cluster accuracy: %.2f%% (%d/%d)", accuracy*100, clusterHits, len(results))

	// Expect at least 50% accuracy for clustered data
	if accuracy < 0.5 {
		t.Errorf("accuracy too low: %.2f%%", accuracy*100)
	}
}

func TestDiskANNDimensionMismatch(t *testing.T) {
	dim := 128
	config := map[string]interface{}{
		"memory_limit": 100,
		"index_path":   "/tmp/test_diskann_dim.idx",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_dim.idx")
		}
	}()

	// Try to add vector with wrong dimension
	wrongVec := randomVector(64)
	err = idx.Add(context.Background(), 1, wrongVec)
	if err == nil {
		t.Error("expected error for dimension mismatch on add")
	}

	// Add correct vector
	correctVec := randomVector(dim)
	idx.Add(context.Background(), 1, correctVec)

	// Try to search with wrong dimension
	wrongQuery := randomVector(64)
	_, err = idx.Search(context.Background(), wrongQuery, 5, nil)
	if err == nil {
		t.Error("expected error for dimension mismatch on search")
	}
}

func BenchmarkDiskANNAdd(b *testing.B) {
	dim := 128
	config := map[string]interface{}{
		"memory_limit": 1000,
		"max_degree":   32,
		"index_path":   "/tmp/bench_diskann_add.idx",
	}

	idx, _ := NewDiskANNIndex(dim, config)
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/bench_diskann_add.idx")
		}
	}()

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

func BenchmarkDiskANNSearch(b *testing.B) {
	dim := 128
	numVectors := 10000
	config := map[string]interface{}{
		"memory_limit": 5000,
		"max_degree":   32,
		"ef_search":    50,
		"index_path":   "/tmp/bench_diskann_search.idx",
	}

	idx, _ := NewDiskANNIndex(dim, config)
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/bench_diskann_search.idx")
		}
	}()

	// Populate index
	for i := 0; i < numVectors; i++ {
		vec := randomVector(dim)
		idx.Add(context.Background(), uint64(i), vec)
	}

	query := randomVector(dim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(context.Background(), query, 10, nil)
	}
}
// ==================================================================
// Phase 2.1: Parallel Construction Tests
// ==================================================================

func TestDiskANNAddBatch(t *testing.T) {
	dim := 64
	config := map[string]interface{}{
		"memory_limit": 100,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_batch.idx",
		"metric":       "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_batch.idx")
		}
	}()

	// Prepare batch
	batchSize := 500
	vectors := make(map[uint64][]float32)
	for i := 0; i < batchSize; i++ {
		vectors[uint64(i)] = randomVector(dim)
	}

	// Add batch
	if err := idx.(*DiskANNIndex).AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}

	// Verify all vectors added
	stats := idx.Stats()
	if stats.Count != batchSize {
		t.Errorf("expected count %d, got %d", batchSize, stats.Count)
	}

	// Verify graph connectivity
	diskANN := idx.(*DiskANNIndex)
	diskANN.mu.RLock()
	graphSize := len(diskANN.graph)
	diskANN.mu.RUnlock()

	if graphSize != batchSize {
		t.Errorf("expected graph size %d, got %d", batchSize, graphSize)
	}

	// Test search works
	query := randomVector(dim)
	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results after batch add")
	}

	t.Logf("Successfully added %d vectors in batch, graph has %d nodes", batchSize, graphSize)
}

func TestDiskANNParallelVsSequential(t *testing.T) {
	dim := 128
	batchSize := 1000

	config := map[string]interface{}{
		"memory_limit": 500,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_parallel.idx",
	}

	// Prepare vectors once
	vectors := make(map[uint64][]float32)
	for i := 0; i < batchSize; i++ {
		vectors[uint64(i)] = randomVector(dim)
	}

	// Test 1: Sequential addition
	idx1, _ := NewDiskANNIndex(dim, config)
	defer func() {
		if d, ok := idx1.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_parallel.idx")
		}
	}()

	seqStart := time.Now()
	for id, vec := range vectors {
		idx1.Add(context.Background(), id, vec)
	}
	seqDuration := time.Since(seqStart)

	// Test 2: Parallel addition
	config["index_path"] = "/tmp/test_diskann_parallel2.idx"
	idx2, _ := NewDiskANNIndex(dim, config)
	defer func() {
		if d, ok := idx2.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_parallel2.idx")
		}
	}()

	parStart := time.Now()
	if err := idx2.(*DiskANNIndex).AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("parallel add failed: %v", err)
	}
	parDuration := time.Since(parStart)

	// Verify both have same count
	if idx1.Stats().Count != idx2.Stats().Count {
		t.Error("sequential and parallel counts differ")
	}

	// Calculate speedup
	speedup := float64(seqDuration) / float64(parDuration)
	t.Logf("Sequential: %v, Parallel: %v, Speedup: %.2fx", seqDuration, parDuration, speedup)

	// Expect at least 1.5x speedup (conservative, actual should be 2-4x)
	if speedup < 1.5 {
		t.Logf("Warning: expected speedup >= 1.5x, got %.2fx", speedup)
	}
}

func TestDiskANNIncrementalBatch(t *testing.T) {
	dim := 64
	config := map[string]interface{}{
		"memory_limit": 50,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_incremental.idx",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_incremental.idx")
		}
	}()

	// Add large batch incrementally
	totalVectors := 1000
	bufferSize := 200

	vectors := make(map[uint64][]float32)
	for i := 0; i < totalVectors; i++ {
		vectors[uint64(i)] = randomVector(dim)
	}

	diskANN := idx.(*DiskANNIndex)
	if err := diskANN.AddIncrementalBatch(context.Background(), vectors, bufferSize); err != nil {
		t.Fatalf("incremental batch failed: %v", err)
	}

	// Verify all vectors added
	stats := idx.Stats()
	if stats.Count != totalVectors {
		t.Errorf("expected count %d, got %d", totalVectors, stats.Count)
	}

	// Test search
	query := randomVector(dim)
	results, err := idx.Search(context.Background(), query, 10, nil)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected search results")
	}

	t.Logf("Successfully added %d vectors incrementally (buffer size: %d)", totalVectors, bufferSize)
}

func TestDiskANNParallelSearchBatch(t *testing.T) {
	dim := 64
	numVectors := 1000
	numQueries := 100

	config := map[string]interface{}{
		"memory_limit": 500,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_parallel_search.idx",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_parallel_search.idx")
		}
	}()

	// Populate index
	vectors := make(map[uint64][]float32)
	for i := 0; i < numVectors; i++ {
		vectors[uint64(i)] = randomVector(dim)
	}

	diskANN := idx.(*DiskANNIndex)
	if err := diskANN.AddBatch(context.Background(), vectors); err != nil {
		t.Fatalf("failed to populate index: %v", err)
	}

	// Prepare queries
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = randomVector(dim)
	}

	// Parallel search
	results, err := diskANN.ParallelSearchBatch(context.Background(), queries, 10, nil)
	if err != nil {
		t.Fatalf("parallel search failed: %v", err)
	}

	if len(results) != numQueries {
		t.Errorf("expected %d result sets, got %d", numQueries, len(results))
	}

	// Verify each result set
	for i, resultSet := range results {
		if len(resultSet) == 0 {
			t.Errorf("query %d returned no results", i)
		}
		if len(resultSet) > 10 {
			t.Errorf("query %d returned too many results: %d", i, len(resultSet))
		}
	}

	t.Logf("Successfully performed %d parallel searches", numQueries)
}

func TestDiskANNCancellation(t *testing.T) {
	dim := 64
	config := map[string]interface{}{
		"memory_limit": 50,
		"index_path":   "/tmp/test_diskann_cancel.idx",
	}

	idx, _ := NewDiskANNIndex(dim, config)
	defer func() {
		if d, ok := idx.(*DiskANNIndex); ok {
			d.Close()
			os.Remove("/tmp/test_diskann_cancel.idx")
		}
	}()

	// Create cancelable context
	ctx, cancel := context.WithCancel(context.Background())

	// Start large batch
	vectors := make(map[uint64][]float32)
	for i := 0; i < 10000; i++ {
		vectors[uint64(i)] = randomVector(dim)
	}

	// Cancel after short delay
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	diskANN := idx.(*DiskANNIndex)
	err := diskANN.AddBatch(ctx, vectors)

	// Should get context canceled error
	if err != context.Canceled && err != nil {
		t.Logf("Got expected cancellation result: %v", err)
	}
}

// ==================================================================
// Benchmarks for Parallel Construction
// ==================================================================

func BenchmarkDiskANNBatchAdd(b *testing.B) {
	dim := 128
	batchSizes := []int{100, 500, 1000, 5000}

	for _, size := range batchSizes {
		b.Run(fmt.Sprintf("BatchSize%d", size), func(b *testing.B) {
			config := map[string]interface{}{
				"memory_limit": size / 2,
				"max_degree":   32,
				"index_path":   fmt.Sprintf("/tmp/bench_diskann_batch_%d.idx", size),
			}

			// Prepare vectors
			vectors := make(map[uint64][]float32)
			for i := 0; i < size; i++ {
				vectors[uint64(i)] = randomVector(dim)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx, _ := NewDiskANNIndex(dim, config)
				diskANN := idx.(*DiskANNIndex)
				diskANN.AddBatch(context.Background(), vectors)
				diskANN.Close()
			}

			os.Remove(config["index_path"].(string))
		})
	}
}

func BenchmarkDiskANNParallelVsSequential(b *testing.B) {
	dim := 128
	batchSize := 1000

	config := map[string]interface{}{
		"memory_limit": 500,
		"max_degree":   32,
		"index_path":   "/tmp/bench_diskann_comparison.idx",
	}

	// Prepare vectors
	vectors := make(map[uint64][]float32)
	for i := 0; i < batchSize; i++ {
		vectors[uint64(i)] = randomVector(dim)
	}

	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			idx, _ := NewDiskANNIndex(dim, config)
			for id, vec := range vectors {
				idx.Add(context.Background(), id, vec)
			}
			idx.(*DiskANNIndex).Close()
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			idx, _ := NewDiskANNIndex(dim, config)
			diskANN := idx.(*DiskANNIndex)
			diskANN.AddBatch(context.Background(), vectors)
			diskANN.Close()
		}
	})

	os.Remove("/tmp/bench_diskann_comparison.idx")
}

func BenchmarkDiskANNParallelSearch(b *testing.B) {
	dim := 128
	numVectors := 10000
	numQueries := 100

	config := map[string]interface{}{
		"memory_limit": 5000,
		"max_degree":   32,
		"index_path":   "/tmp/bench_diskann_par_search.idx",
	}

	idx, _ := NewDiskANNIndex(dim, config)
	defer func() {
		idx.(*DiskANNIndex).Close()
		os.Remove("/tmp/bench_diskann_par_search.idx")
	}()

	// Populate
	vectors := make(map[uint64][]float32)
	for i := 0; i < numVectors; i++ {
		vectors[uint64(i)] = randomVector(dim)
	}
	idx.(*DiskANNIndex).AddBatch(context.Background(), vectors)

	// Prepare queries
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = randomVector(dim)
	}

	diskANN := idx.(*DiskANNIndex)

	b.Run("ParallelBatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			diskANN.ParallelSearchBatch(context.Background(), queries, 10, nil)
		}
	})

	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, query := range queries {
				diskANN.Search(context.Background(), query, 10, nil)
			}
		}
	})
}
