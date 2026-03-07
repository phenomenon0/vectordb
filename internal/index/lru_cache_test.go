package index

import (
	"sync"
	"testing"
)

func TestLRUCacheBasicOperations(t *testing.T) {
	cache := NewLRUCache(3, 0) // Capacity of 3, no memory limit

	// Test Put and Get
	vec1 := []float32{1.0, 2.0, 3.0}
	vec2 := []float32{4.0, 5.0, 6.0}

	cache.Put(1, vec1)
	cache.Put(2, vec2)

	// Get existing
	retrieved, ok := cache.Get(1)
	if !ok {
		t.Fatal("Expected to find vector 1")
	}
	if len(retrieved.([]float32)) != 3 {
		t.Errorf("Expected vector length 3, got %d", len(retrieved.([]float32)))
	}

	// Get non-existing
	_, ok = cache.Get(999)
	if ok {
		t.Error("Expected miss for non-existing key")
	}

	// Check stats
	stats := cache.Stats()
	if stats.Hits != 1 {
		t.Errorf("Expected 1 hit, got %d", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Errorf("Expected 1 miss, got %d", stats.Misses)
	}
	if stats.Size != 2 {
		t.Errorf("Expected size 2, got %d", stats.Size)
	}
}

func TestLRUCacheCapacityEviction(t *testing.T) {
	cache := NewLRUCache(3, 0) // Capacity of 3

	// Add 4 vectors, should evict the oldest
	for i := 1; i <= 4; i++ {
		vec := []float32{float32(i)}
		cache.Put(uint64(i), vec)
	}

	// Vector 1 should be evicted (oldest)
	_, ok := cache.Get(1)
	if ok {
		t.Error("Expected vector 1 to be evicted")
	}

	// Vectors 2, 3, 4 should exist
	for i := 2; i <= 4; i++ {
		_, ok := cache.Get(uint64(i))
		if !ok {
			t.Errorf("Expected vector %d to exist", i)
		}
	}

	stats := cache.Stats()
	if stats.Size != 3 {
		t.Errorf("Expected size 3, got %d", stats.Size)
	}
	if stats.Evictions != 1 {
		t.Errorf("Expected 1 eviction, got %d", stats.Evictions)
	}
}

func TestLRUCacheLRUOrdering(t *testing.T) {
	cache := NewLRUCache(3, 0)

	// Add 3 vectors
	for i := 1; i <= 3; i++ {
		cache.Put(uint64(i), []float32{float32(i)})
	}

	// Access vector 1 (make it most recently used)
	cache.Get(1)

	// Add vector 4, should evict vector 2 (oldest untouched)
	cache.Put(4, []float32{4.0})

	// Vector 1 should still exist (was accessed)
	_, ok := cache.Get(1)
	if !ok {
		t.Error("Expected vector 1 to still exist (was accessed)")
	}

	// Vector 2 should be evicted
	_, ok = cache.Get(2)
	if ok {
		t.Error("Expected vector 2 to be evicted")
	}

	// Vectors 3 and 4 should exist
	_, ok = cache.Get(3)
	if !ok {
		t.Error("Expected vector 3 to exist")
	}
	_, ok = cache.Get(4)
	if !ok {
		t.Error("Expected vector 4 to exist")
	}
}

func TestLRUCacheMemoryEviction(t *testing.T) {
	// Each float32 is 4 bytes, + 24 bytes slice overhead
	// Vector of length 10 = 40 + 24 = 64 bytes
	maxBytes := int64(200) // Room for ~3 vectors

	cache := NewLRUCache(0, maxBytes) // No capacity limit, memory limit only

	vec := make([]float32, 10) // ~64 bytes

	// Add 4 vectors, should evict based on memory
	for i := 1; i <= 4; i++ {
		cache.Put(uint64(i), vec)
	}

	// Should have evicted at least one vector
	stats := cache.Stats()
	if stats.Evictions < 1 {
		t.Errorf("Expected at least 1 eviction due to memory limit, got %d", stats.Evictions)
	}
	if stats.MemoryUsed > maxBytes {
		t.Errorf("Memory usage %d exceeds limit %d", stats.MemoryUsed, maxBytes)
	}
}

func TestLRUCacheUpdate(t *testing.T) {
	cache := NewLRUCache(10, 0)

	vec1 := []float32{1.0, 2.0, 3.0}
	vec2 := []float32{4.0, 5.0, 6.0, 7.0}

	// Initial put
	cache.Put(1, vec1)
	stats := cache.Stats()
	initialSize := stats.MemoryUsed

	// Update with larger vector
	cache.Put(1, vec2)

	// Should still have only 1 entry
	stats = cache.Stats()
	if stats.Size != 1 {
		t.Errorf("Expected size 1 after update, got %d", stats.Size)
	}

	// Memory should have increased
	if stats.MemoryUsed <= initialSize {
		t.Errorf("Expected memory to increase after update, got %d (was %d)",
			stats.MemoryUsed, initialSize)
	}

	// Retrieved should be the new vector
	retrieved, ok := cache.Get(1)
	if !ok {
		t.Fatal("Expected to find updated vector")
	}
	if len(retrieved.([]float32)) != 4 {
		t.Errorf("Expected updated vector length 4, got %d", len(retrieved.([]float32)))
	}
}

func TestLRUCacheRemove(t *testing.T) {
	cache := NewLRUCache(10, 0)

	cache.Put(1, []float32{1.0})
	cache.Put(2, []float32{2.0})
	cache.Put(3, []float32{3.0})

	// Remove vector 2
	cache.Remove(2)

	// Should not find vector 2
	_, ok := cache.Get(2)
	if ok {
		t.Error("Expected vector 2 to be removed")
	}

	// Others should still exist
	_, ok = cache.Get(1)
	if !ok {
		t.Error("Expected vector 1 to exist")
	}
	_, ok = cache.Get(3)
	if !ok {
		t.Error("Expected vector 3 to exist")
	}

	stats := cache.Stats()
	if stats.Size != 2 {
		t.Errorf("Expected size 2 after removal, got %d", stats.Size)
	}
}

func TestLRUCacheClear(t *testing.T) {
	cache := NewLRUCache(10, 0)

	for i := 1; i <= 5; i++ {
		cache.Put(uint64(i), []float32{float32(i)})
	}

	// Clear all
	cache.Clear()

	stats := cache.Stats()
	if stats.Size != 0 {
		t.Errorf("Expected size 0 after clear, got %d", stats.Size)
	}
	if stats.MemoryUsed != 0 {
		t.Errorf("Expected memory 0 after clear, got %d", stats.MemoryUsed)
	}

	// Should not find any vectors
	for i := 1; i <= 5; i++ {
		_, ok := cache.Get(uint64(i))
		if ok {
			t.Errorf("Expected vector %d to be cleared", i)
		}
	}
}

func TestLRUCacheQuantizedVectors(t *testing.T) {
	cache := NewLRUCache(10, 0)

	// Test with byte slices (quantized vectors)
	quantized1 := []byte{1, 2, 3, 4}
	quantized2 := []byte{5, 6, 7, 8}

	cache.Put(1, quantized1)
	cache.Put(2, quantized2)

	retrieved, ok := cache.Get(1)
	if !ok {
		t.Fatal("Expected to find quantized vector")
	}

	bytes := retrieved.([]byte)
	if len(bytes) != 4 {
		t.Errorf("Expected 4 bytes, got %d", len(bytes))
	}
	if bytes[0] != 1 {
		t.Errorf("Expected first byte 1, got %d", bytes[0])
	}
}

func TestLRUCacheThreadSafety(t *testing.T) {
	cache := NewLRUCache(1000, 0)
	var wg sync.WaitGroup

	// Concurrent writes
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vec := []float32{float32(id)}
			cache.Put(uint64(id), vec)
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			cache.Get(uint64(id))
		}(i)
	}

	wg.Wait()

	// Should not panic and should have entries
	stats := cache.Stats()
	if stats.Size == 0 {
		t.Error("Expected cache to have entries after concurrent operations")
	}
}

func TestLRUCacheStats(t *testing.T) {
	cache := NewLRUCache(10, 1000)

	// Perform operations
	cache.Put(1, []float32{1.0, 2.0})
	cache.Put(2, []float32{3.0, 4.0})

	cache.Get(1) // Hit
	cache.Get(2) // Hit
	cache.Get(3) // Miss
	cache.Get(4) // Miss

	stats := cache.Stats()

	if stats.Hits != 2 {
		t.Errorf("Expected 2 hits, got %d", stats.Hits)
	}
	if stats.Misses != 2 {
		t.Errorf("Expected 2 misses, got %d", stats.Misses)
	}
	if stats.HitRate != 50.0 {
		t.Errorf("Expected 50%% hit rate, got %.2f%%", stats.HitRate)
	}
	if stats.Capacity != 10 {
		t.Errorf("Expected capacity 10, got %d", stats.Capacity)
	}
	if stats.MemoryLimit != 1000 {
		t.Errorf("Expected memory limit 1000, got %d", stats.MemoryLimit)
	}

	// Reset stats
	cache.ResetStats()
	stats = cache.Stats()

	if stats.Hits != 0 || stats.Misses != 0 {
		t.Error("Expected stats to be reset")
	}
	if stats.Size != 2 {
		t.Error("Expected size to remain after stats reset")
	}
}

func TestLRUCacheForVectors(t *testing.T) {
	cache := NewLRUCacheForVectors(5, 0)

	vec1 := []float32{1.0, 2.0, 3.0}
	vec2 := []float32{4.0, 5.0, 6.0}

	// Type-safe operations
	cache.Put(1, vec1)
	cache.Put(2, vec2)

	retrieved, ok := cache.Get(1)
	if !ok {
		t.Fatal("Expected to find vector 1")
	}
	if len(retrieved) != 3 {
		t.Errorf("Expected vector length 3, got %d", len(retrieved))
	}
	if retrieved[0] != 1.0 {
		t.Errorf("Expected first element 1.0, got %f", retrieved[0])
	}

	// Test stats
	stats := cache.Stats()
	if stats.Hits != 1 {
		t.Errorf("Expected 1 hit, got %d", stats.Hits)
	}

	// Test clear
	cache.Clear()
	stats = cache.Stats()
	if stats.Size != 0 {
		t.Errorf("Expected size 0 after clear, got %d", stats.Size)
	}
}

func TestLRUCacheSizeCalculation(t *testing.T) {
	cache := NewLRUCache(10, 0)

	// Test float32 vector size
	vec32 := make([]float32, 100)
	expectedSize := int64(100*4 + 24) // 100 floats * 4 bytes + slice overhead

	cache.Put(1, vec32)
	stats := cache.Stats()

	if stats.MemoryUsed != expectedSize {
		t.Errorf("Expected memory %d bytes, got %d", expectedSize, stats.MemoryUsed)
	}

	// Test byte vector size
	vecBytes := make([]byte, 100)
	expectedSizeBytes := int64(100 + 24) // 100 bytes + slice overhead

	cache.Put(2, vecBytes)
	stats = cache.Stats()

	totalExpected := expectedSize + expectedSizeBytes
	if stats.MemoryUsed != totalExpected {
		t.Errorf("Expected total memory %d bytes, got %d", totalExpected, stats.MemoryUsed)
	}
}

func BenchmarkLRUCacheGet(b *testing.B) {
	cache := NewLRUCache(10000, 0)

	// Pre-populate cache
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		cache.Put(uint64(i), vec)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Get(uint64(i % 1000))
	}
}

func BenchmarkLRUCachePut(b *testing.B) {
	cache := NewLRUCache(10000, 0)
	vec := make([]float32, 128)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Put(uint64(i), vec)
	}
}

func BenchmarkLRUCacheConcurrent(b *testing.B) {
	cache := NewLRUCache(10000, 0)

	// Pre-populate
	for i := 0; i < 1000; i++ {
		vec := make([]float32, 128)
		cache.Put(uint64(i), vec)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%2 == 0 {
				cache.Get(uint64(i % 1000))
			} else {
				vec := make([]float32, 128)
				cache.Put(uint64(i), vec)
			}
			i++
		}
	})
}
