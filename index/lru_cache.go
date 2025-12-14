package index

import (
	"container/list"
	"sync"
)

// LRUCache is a thread-safe Least Recently Used cache for vectors
// Uses a doubly-linked list for O(1) access reordering and eviction
type LRUCache struct {
	capacity    int                       // Max number of entries
	maxBytes    int64                     // Max memory usage in bytes
	currentSize int64                     // Current memory usage
	cache       map[uint64]*list.Element  // Fast lookup
	lruList     *list.List                // Eviction order (back = LRU)
	mu          sync.RWMutex              // Thread safety
	hits        uint64                    // Cache hits
	misses      uint64                    // Cache misses
	evictions   uint64                    // Evictions performed
}

// lruEntry represents a cache entry with size tracking
type lruEntry struct {
	id     uint64
	vector interface{} // []float32 or []byte (quantized)
	size   int64       // Memory footprint in bytes
}

// NewLRUCache creates a new LRU cache with capacity and memory limits
// capacity: max number of entries (0 = unlimited, use maxBytes only)
// maxBytes: max memory usage in bytes (0 = unlimited, use capacity only)
func NewLRUCache(capacity int, maxBytes int64) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		maxBytes: maxBytes,
		cache:    make(map[uint64]*list.Element),
		lruList:  list.New(),
	}
}

// Get retrieves a vector from cache and marks it as recently used
// Returns (vector, true) if found, (nil, false) if not found
func (c *LRUCache) Get(id uint64) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.cache[id]
	if !ok {
		c.misses++
		return nil, false
	}

	// Move to front (most recently used)
	c.lruList.MoveToFront(elem)
	c.hits++

	entry := elem.Value.(*lruEntry)
	return entry.vector, true
}

// Put adds or updates a vector in the cache
// If the cache is full, evicts the least recently used entry
func (c *LRUCache) Put(id uint64, vector interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Calculate entry size
	size := c.calculateSize(vector)

	// Check if already exists (update)
	if elem, ok := c.cache[id]; ok {
		entry := elem.Value.(*lruEntry)
		oldSize := entry.size

		// Update entry
		entry.vector = vector
		entry.size = size
		c.currentSize = c.currentSize - oldSize + size

		// Move to front
		c.lruList.MoveToFront(elem)
		return
	}

	// Evict entries if necessary
	c.evictIfNeeded(size)

	// Add new entry
	entry := &lruEntry{
		id:     id,
		vector: vector,
		size:   size,
	}

	elem := c.lruList.PushFront(entry)
	c.cache[id] = elem
	c.currentSize += size
}

// Remove removes an entry from the cache
func (c *LRUCache) Remove(id uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.cache[id]; ok {
		c.removeElement(elem)
	}
}

// Clear removes all entries from the cache
func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache = make(map[uint64]*list.Element)
	c.lruList = list.New()
	c.currentSize = 0
}

// Stats returns cache statistics
func (c *LRUCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(c.hits) / float64(total) * 100
	}

	return CacheStats{
		Hits:        c.hits,
		Misses:      c.misses,
		Evictions:   c.evictions,
		Size:        len(c.cache),
		Capacity:    c.capacity,
		MemoryUsed:  c.currentSize,
		MemoryLimit: c.maxBytes,
		HitRate:     hitRate,
	}
}

// ResetStats resets cache statistics (hits, misses, evictions)
func (c *LRUCache) ResetStats() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.hits = 0
	c.misses = 0
	c.evictions = 0
}

// evictIfNeeded evicts entries until there's space for a new entry of given size
func (c *LRUCache) evictIfNeeded(newSize int64) {
	// Evict by capacity
	if c.capacity > 0 {
		for len(c.cache) >= c.capacity {
			c.evictOldest()
		}
	}

	// Evict by memory size
	if c.maxBytes > 0 {
		for c.currentSize+newSize > c.maxBytes && c.lruList.Len() > 0 {
			c.evictOldest()
		}
	}
}

// evictOldest removes the least recently used entry
func (c *LRUCache) evictOldest() {
	elem := c.lruList.Back()
	if elem != nil {
		c.removeElement(elem)
		c.evictions++
	}
}

// removeElement removes a list element from cache
func (c *LRUCache) removeElement(elem *list.Element) {
	entry := elem.Value.(*lruEntry)
	delete(c.cache, entry.id)
	c.lruList.Remove(elem)
	c.currentSize -= entry.size
}

// calculateSize computes the memory footprint of a vector
func (c *LRUCache) calculateSize(vector interface{}) int64 {
	switch v := vector.(type) {
	case []float32:
		// 4 bytes per float32 + slice overhead
		return int64(len(v)*4 + 24)
	case []byte:
		// 1 byte per byte + slice overhead
		return int64(len(v) + 24)
	default:
		// Unknown type, estimate conservatively
		return 1024
	}
}

// CacheStats holds cache performance statistics
type CacheStats struct {
	Hits        uint64  // Number of cache hits
	Misses      uint64  // Number of cache misses
	Evictions   uint64  // Number of evictions
	Size        int     // Current number of entries
	Capacity    int     // Max number of entries
	MemoryUsed  int64   // Current memory usage (bytes)
	MemoryLimit int64   // Max memory usage (bytes)
	HitRate     float64 // Hit rate percentage
}

// LRUCacheForVectors is a specialized LRU cache for float32 vectors
// Provides type-safe access for the common case
type LRUCacheForVectors struct {
	cache *LRUCache
}

// NewLRUCacheForVectors creates a vector-specific LRU cache
func NewLRUCacheForVectors(capacity int, maxBytes int64) *LRUCacheForVectors {
	return &LRUCacheForVectors{
		cache: NewLRUCache(capacity, maxBytes),
	}
}

// Get retrieves a float32 vector from cache
func (c *LRUCacheForVectors) Get(id uint64) ([]float32, bool) {
	vec, ok := c.cache.Get(id)
	if !ok {
		return nil, false
	}
	return vec.([]float32), true
}

// Put adds a float32 vector to cache
func (c *LRUCacheForVectors) Put(id uint64, vector []float32) {
	c.cache.Put(id, vector)
}

// Remove removes a vector from cache
func (c *LRUCacheForVectors) Remove(id uint64) {
	c.cache.Remove(id)
}

// Clear removes all vectors from cache
func (c *LRUCacheForVectors) Clear() {
	c.cache.Clear()
}

// Stats returns cache statistics
func (c *LRUCacheForVectors) Stats() CacheStats {
	return c.cache.Stats()
}

// ResetStats resets cache statistics
func (c *LRUCacheForVectors) ResetStats() {
	c.cache.ResetStats()
}
