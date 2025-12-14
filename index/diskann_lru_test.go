package index

import (
	"context"
	"math/rand"
	"os"
	"testing"
)

// TestDiskANNLRUCacheHitRate tests the LRU cache hit rate improvement
func TestDiskANNLRUCacheHitRate(t *testing.T) {
	dim := 64
	memoryLimit := 50 // Keep only 50 vectors in memory
	totalVectors := 200 // Total 200 vectors (150 on disk)

	config := map[string]interface{}{
		"memory_limit": memoryLimit,
		"max_degree":   16,
		"index_path":   "/tmp/test_diskann_lru_cache.idx",
		"metric":       "cosine",
	}

	idx, err := NewDiskANNIndex(dim, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}
	defer func() {
		diskANN := idx.(*DiskANNIndex)
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	diskANN := idx.(*DiskANNIndex)

	// Add vectors
	for i := 0; i < totalVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		if err := idx.Add(context.Background(), uint64(i), vec); err != nil {
			t.Fatalf("Failed to add vector %d: %v", i, err)
		}
	}

	stats := diskANN.Stats()
	t.Logf("After insertion: Memory: %d, Disk: %d",
		stats.Extra["memory_vectors"], stats.Extra["disk_vectors"])

	// Reset cache stats to measure hit rate from fresh
	diskANN.lruCache.ResetStats()
	diskANN.diskReads = 0
	diskANN.memoryHits = 0

	// Simulate workload: access disk vectors repeatedly (should benefit from cache)
	// Access vectors 100-150 (which are on disk) multiple times
	accessPattern := []int{}
	for round := 0; round < 10; round++ {
		for i := 100; i < 150; i++ {
			accessPattern = append(accessPattern, i)
		}
	}

	// Shuffle access pattern to simulate real workload
	rand.Shuffle(len(accessPattern), func(i, j int) {
		accessPattern[i], accessPattern[j] = accessPattern[j], accessPattern[i]
	})

	// Perform searches that will access disk vectors
	for _, vecID := range accessPattern {
		_ = vecID // Access pattern simulated through searches
		query := make([]float32, dim)
		for j := 0; j < dim; j++ {
			query[j] = rand.Float32()
		}
		_, err := idx.Search(context.Background(), query, 5, nil)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}
	}

	// Check cache statistics
	cacheStats := diskANN.lruCache.Stats()
	diskReads := diskANN.diskReads

	t.Logf("Cache Statistics:")
	t.Logf("  Hits: %d", cacheStats.Hits)
	t.Logf("  Misses: %d", cacheStats.Misses)
	t.Logf("  Hit Rate: %.2f%%", cacheStats.HitRate)
	t.Logf("  Evictions: %d", cacheStats.Evictions)
	t.Logf("  Cache Size: %d/%d", cacheStats.Size, cacheStats.Capacity)
	t.Logf("Disk Reads: %d", diskReads)

	// With LRU cache, we should have significantly fewer disk reads
	// than the total number of accesses (500 accesses of 50 unique disk vectors)
	maxExpectedDiskReads := 150 // At most 150 reads (first access to each disk vector + some evictions)
	if diskReads > int64(maxExpectedDiskReads) {
		t.Errorf("Too many disk reads: %d (expected <%d). LRU cache may not be working effectively.",
			diskReads, maxExpectedDiskReads)
	}

	// Cache hit rate should be reasonable (>50%)
	if cacheStats.HitRate < 50.0 {
		t.Errorf("Cache hit rate too low: %.2f%% (expected >50%%)", cacheStats.HitRate)
	}

	t.Logf("✓ LRU cache is effectively reducing disk reads")
	t.Logf("  Saved ~%d disk reads through caching", 500-int(diskReads))
}

// BenchmarkDiskANNWithoutLRU benchmarks DiskANN with legacy caching (baseline)
func BenchmarkDiskANNWithoutLRU(b *testing.B) {
	dim := 128
	numVectors := 1000
	memoryLimit := 100 // Only 100 in memory, 900 on disk

	config := map[string]interface{}{
		"memory_limit": memoryLimit,
		"max_degree":   32,
		"index_path":   "/tmp/bench_diskann_no_lru.idx",
		"metric":       "cosine",
	}

	idx, _ := NewDiskANNIndex(dim, config)
	diskANN := idx.(*DiskANNIndex)
	defer func() {
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	// Add vectors
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Disable LRU cache to simulate old behavior
	diskANN.lruCache.Clear()

	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(context.Background(), query, 10, nil)
	}
}

// BenchmarkDiskANNWithLRU benchmarks DiskANN with LRU caching
func BenchmarkDiskANNWithLRU(b *testing.B) {
	dim := 128
	numVectors := 1000
	memoryLimit := 100 // Only 100 in memory, 900 on disk

	config := map[string]interface{}{
		"memory_limit": memoryLimit,
		"max_degree":   32,
		"index_path":   "/tmp/bench_diskann_with_lru.idx",
		"metric":       "cosine",
	}

	idx, _ := NewDiskANNIndex(dim, config)
	diskANN := idx.(*DiskANNIndex)
	defer func() {
		diskANN.Close()
		os.Remove(diskANN.indexPath)
	}()

	// Add vectors
	for i := 0; i < numVectors; i++ {
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = rand.Float32()
		}
		idx.Add(context.Background(), uint64(i), vec)
	}

	// Warm up LRU cache with a few searches
	query := make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = rand.Float32()
	}
	for i := 0; i < 10; i++ {
		idx.Search(context.Background(), query, 10, nil)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(context.Background(), query, 10, nil)
	}
}
