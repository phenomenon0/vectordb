package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// ===========================================================================================
// QUERY RESULT CACHING
// LRU cache with TTL for 10-100x speedup on repeated queries
// ===========================================================================================

// CacheEntry represents a cached query result
type CacheEntry struct {
	Key        string
	Results    []map[string]any
	CreatedAt  time.Time
	ExpiresAt  time.Time
	AccessedAt time.Time
	HitCount   int64
}

// IsExpired checks if the cache entry has expired
func (ce *CacheEntry) IsExpired() bool {
	return time.Now().After(ce.ExpiresAt)
}

// QueryCache implements an LRU cache with TTL for query results
type QueryCache struct {
	mu sync.RWMutex

	// Cache storage: key -> entry
	cache map[string]*CacheEntry

	// LRU tracking (doubly-linked list)
	head *cacheNode
	tail *cacheNode

	// Node lookup for O(1) access
	nodes map[string]*cacheNode

	// Configuration
	maxSize int
	ttl     time.Duration

	// Statistics
	stats *CacheStats
}

// cacheNode is a node in the LRU doubly-linked list
type cacheNode struct {
	key  string
	prev *cacheNode
	next *cacheNode
}

// NewQueryCache creates a new query cache
func NewQueryCache(maxSize int, ttl time.Duration) *QueryCache {
	return &QueryCache{
		cache:   make(map[string]*CacheEntry),
		nodes:   make(map[string]*cacheNode),
		maxSize: maxSize,
		ttl:     ttl,
		stats:   NewCacheStats(),
	}
}

// Get retrieves a cached query result
func (qc *QueryCache) Get(queryVec []float32, topK int, filter map[string]string, mode string) []map[string]any {
	key := qc.generateKey(queryVec, topK, filter, mode)

	qc.mu.Lock()
	defer qc.mu.Unlock()

	entry, ok := qc.cache[key]
	if !ok {
		qc.stats.RecordMiss()
		return nil
	}

	// Check if expired
	if entry.IsExpired() {
		qc.removeEntry(key)
		qc.stats.RecordExpiration()
		return nil
	}

	// Cache hit!
	entry.AccessedAt = time.Now()
	entry.HitCount++
	qc.stats.RecordHit()

	// Move to front (most recently used)
	qc.moveToFront(key)

	return entry.Results
}

// Put stores a query result in the cache
func (qc *QueryCache) Put(
	queryVec []float32,
	topK int,
	filter map[string]string,
	mode string,
	results []map[string]any,
) {
	key := qc.generateKey(queryVec, topK, filter, mode)

	qc.mu.Lock()
	defer qc.mu.Unlock()

	// Check if key already exists
	if _, ok := qc.cache[key]; ok {
		// Update existing entry
		entry := qc.cache[key]
		entry.Results = results
		entry.ExpiresAt = time.Now().Add(qc.ttl)
		qc.moveToFront(key)
		return
	}

	// Create new entry
	entry := &CacheEntry{
		Key:        key,
		Results:    results,
		CreatedAt:  time.Now(),
		ExpiresAt:  time.Now().Add(qc.ttl),
		AccessedAt: time.Now(),
		HitCount:   0,
	}

	qc.cache[key] = entry
	qc.addToFront(key)

	qc.stats.RecordPut(len(results))

	// Evict if over capacity
	if len(qc.cache) > qc.maxSize {
		qc.evictLRU()
	}
}

// InvalidateAll clears the entire cache (call on writes)
func (qc *QueryCache) InvalidateAll() {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	qc.cache = make(map[string]*CacheEntry)
	qc.nodes = make(map[string]*cacheNode)
	qc.head = nil
	qc.tail = nil

	qc.stats.RecordInvalidation(0) // Full invalidation
}

// InvalidateCollection invalidates all cache entries for a specific collection
func (qc *QueryCache) InvalidateCollection(collection string) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	invalidated := 0
	for key, entry := range qc.cache {
		// Check if this entry is for the given collection
		// (This is a simple string match - in production, you'd parse the key)
		if contains(entry.Results, collection) {
			qc.removeEntry(key)
			invalidated++
		}
	}

	qc.stats.RecordInvalidation(invalidated)
}

// EvictExpired removes all expired entries
func (qc *QueryCache) EvictExpired() int {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	evicted := 0
	for key, entry := range qc.cache {
		if entry.IsExpired() {
			qc.removeEntry(key)
			evicted++
		}
	}

	return evicted
}

// GetSize returns the current cache size
func (qc *QueryCache) GetSize() int {
	qc.mu.RLock()
	defer qc.mu.RUnlock()
	return len(qc.cache)
}

// GetStats returns cache statistics
func (qc *QueryCache) GetStats() map[string]any {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	stats := qc.stats.GetStats()
	stats["current_size"] = len(qc.cache)
	stats["max_size"] = qc.maxSize
	stats["ttl_seconds"] = qc.ttl.Seconds()

	// Top 10 most accessed entries
	topEntries := qc.getTopEntries(10)
	stats["top_entries"] = topEntries

	return stats
}

// ===========================================================================================
// LRU IMPLEMENTATION
// ===========================================================================================

// moveToFront moves a node to the front of the LRU list
func (qc *QueryCache) moveToFront(key string) {
	node, ok := qc.nodes[key]
	if !ok {
		return
	}

	// Already at front?
	if node == qc.head {
		return
	}

	// Remove from current position
	qc.removeNode(node)

	// Add to front
	qc.addNodeToFront(node)
}

// addToFront adds a new key to the front of the LRU list
func (qc *QueryCache) addToFront(key string) {
	node := &cacheNode{key: key}
	qc.nodes[key] = node
	qc.addNodeToFront(node)
}

// addNodeToFront adds a node to the front of the list
func (qc *QueryCache) addNodeToFront(node *cacheNode) {
	if qc.head == nil {
		// Empty list
		qc.head = node
		qc.tail = node
		node.prev = nil
		node.next = nil
	} else {
		// Add to front
		node.next = qc.head
		node.prev = nil
		qc.head.prev = node
		qc.head = node
	}
}

// removeNode removes a node from the list
func (qc *QueryCache) removeNode(node *cacheNode) {
	if node.prev != nil {
		node.prev.next = node.next
	} else {
		// Removing head
		qc.head = node.next
	}

	if node.next != nil {
		node.next.prev = node.prev
	} else {
		// Removing tail
		qc.tail = node.prev
	}
}

// evictLRU evicts the least recently used entry
func (qc *QueryCache) evictLRU() {
	if qc.tail == nil {
		return
	}

	// Remove tail (least recently used)
	key := qc.tail.key
	qc.removeEntry(key)
	qc.stats.RecordEviction()
}

// removeEntry removes an entry from the cache
func (qc *QueryCache) removeEntry(key string) {
	// Remove from cache
	delete(qc.cache, key)

	// Remove from LRU list
	if node, ok := qc.nodes[key]; ok {
		qc.removeNode(node)
		delete(qc.nodes, key)
	}
}

// ===========================================================================================
// KEY GENERATION
// ===========================================================================================

// generateKey generates a cache key from query parameters
func (qc *QueryCache) generateKey(
	queryVec []float32,
	topK int,
	filter map[string]string,
	mode string,
) string {
	// Create a deterministic key from query parameters
	// Use hash of vector + parameters to keep key size manageable

	h := sha256.New()

	// Hash query vector (sample first 10 dims to keep key size small)
	sampleSize := 10
	if len(queryVec) < sampleSize {
		sampleSize = len(queryVec)
	}
	for i := 0; i < sampleSize; i++ {
		h.Write([]byte(fmt.Sprintf("%.4f", queryVec[i])))
	}

	// Hash parameters
	h.Write([]byte(fmt.Sprintf("%d", topK)))
	h.Write([]byte(mode))

	// Hash filter (sorted for determinism)
	if len(filter) > 0 {
		filterJSON, _ := json.Marshal(filter)
		h.Write(filterJSON)
	}

	return fmt.Sprintf("%x", h.Sum(nil))
}

// ===========================================================================================
// CACHE STATISTICS
// ===========================================================================================

// CacheStats tracks cache performance statistics
type CacheStats struct {
	mu sync.RWMutex

	Hits           int64
	Misses         int64
	Puts           int64
	Evictions      int64
	Expirations    int64
	Invalidations  int64
	TotalResultsCached int64
}

// NewCacheStats creates a new cache statistics tracker
func NewCacheStats() *CacheStats {
	return &CacheStats{}
}

// RecordHit records a cache hit
func (cs *CacheStats) RecordHit() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.Hits++
}

// RecordMiss records a cache miss
func (cs *CacheStats) RecordMiss() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.Misses++
}

// RecordPut records a cache put operation
func (cs *CacheStats) RecordPut(resultCount int) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.Puts++
	cs.TotalResultsCached += int64(resultCount)
}

// RecordEviction records a cache eviction
func (cs *CacheStats) RecordEviction() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.Evictions++
}

// RecordExpiration records a cache expiration
func (cs *CacheStats) RecordExpiration() {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.Expirations++
}

// RecordInvalidation records cache invalidations
func (cs *CacheStats) RecordInvalidation(count int) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.Invalidations += int64(count)
}

// GetStats returns current statistics
func (cs *CacheStats) GetStats() map[string]any {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	total := cs.Hits + cs.Misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(cs.Hits) / float64(total)
	}

	avgResultsPerEntry := float64(0)
	if cs.Puts > 0 {
		avgResultsPerEntry = float64(cs.TotalResultsCached) / float64(cs.Puts)
	}

	return map[string]any{
		"hits":                  cs.Hits,
		"misses":                cs.Misses,
		"puts":                  cs.Puts,
		"evictions":             cs.Evictions,
		"expirations":           cs.Expirations,
		"invalidations":         cs.Invalidations,
		"hit_rate":              hitRate,
		"total_requests":        total,
		"total_results_cached":  cs.TotalResultsCached,
		"avg_results_per_entry": avgResultsPerEntry,
	}
}

// ===========================================================================================
// HELPER FUNCTIONS
// ===========================================================================================

// getTopEntries returns the top N most accessed cache entries
func (qc *QueryCache) getTopEntries(n int) []map[string]any {
	entries := make([]*CacheEntry, 0, len(qc.cache))
	for _, entry := range qc.cache {
		entries = append(entries, entry)
	}

	// Sort by hit count (descending)
	for i := 0; i < len(entries)-1; i++ {
		for j := i + 1; j < len(entries); j++ {
			if entries[j].HitCount > entries[i].HitCount {
				entries[i], entries[j] = entries[j], entries[i]
			}
		}
	}

	// Take top N
	if len(entries) > n {
		entries = entries[:n]
	}

	result := make([]map[string]any, len(entries))
	for i, entry := range entries {
		result[i] = map[string]any{
			"key":         entry.Key[:16] + "...", // Truncated key
			"hit_count":   entry.HitCount,
			"created_at":  entry.CreatedAt.Unix(),
			"accessed_at": entry.AccessedAt.Unix(),
			"age_seconds": time.Since(entry.CreatedAt).Seconds(),
		}
	}

	return result
}

// contains checks if results contain a specific collection
// (Simple implementation - in production, parse the key properly)
func contains(results []map[string]any, collection string) bool {
	for _, result := range results {
		if coll, ok := result["collection"].(string); ok {
			if coll == collection {
				return true
			}
		}
	}
	return false
}

// ===========================================================================================
// CACHE WARMUP
// ===========================================================================================

// CacheWarmer pre-populates the cache with common queries
type CacheWarmer struct {
	cache  *QueryCache
	warmQueries []WarmQuery
}

// WarmQuery represents a query to pre-populate in cache
type WarmQuery struct {
	QueryVec []float32
	TopK     int
	Filter   map[string]string
	Mode     string
}

// NewCacheWarmer creates a new cache warmer
func NewCacheWarmer(cache *QueryCache) *CacheWarmer {
	return &CacheWarmer{
		cache:       cache,
		warmQueries: make([]WarmQuery, 0),
	}
}

// AddWarmQuery adds a query to warm up on startup
func (cw *CacheWarmer) AddWarmQuery(query WarmQuery) {
	cw.warmQueries = append(cw.warmQueries, query)
}

// WarmUp executes all warm queries and populates the cache
func (cw *CacheWarmer) WarmUp(searcher func(WarmQuery) []map[string]any) {
	for _, query := range cw.warmQueries {
		results := searcher(query)
		cw.cache.Put(query.QueryVec, query.TopK, query.Filter, query.Mode, results)
	}
}

// ===========================================================================================
// USAGE EXAMPLE
// ===========================================================================================

/*
Example usage:

// Create cache (max 10000 entries, 5 minute TTL)
cache := NewQueryCache(10000, 5*time.Minute)

// Query 1 (cache miss)
queryVec := []float32{0.1, 0.2, 0.3, ...}
results := cache.Get(queryVec, 10, filter, "hybrid")
if results == nil {
    // Cache miss - perform actual search
    results = vectorStore.Search(queryVec, 10, filter, "hybrid")

    // Store in cache
    cache.Put(queryVec, 10, filter, "hybrid", results)
}

// Query 2 (same query - cache hit!)
results = cache.Get(queryVec, 10, filter, "hybrid")
// Instant return from cache (0.1ms instead of 50ms)

// On write operations, invalidate cache
vectorStore.Insert(doc, vec, meta)
cache.InvalidateAll()  // or cache.InvalidateCollection(collection)

// Periodic cleanup
go func() {
    ticker := time.NewTicker(1 * time.Minute)
    for range ticker.C {
        evicted := cache.EvictExpired()
        fmt.Printf("Evicted %d expired entries\n", evicted)
    }
}()

// Performance impact:
// Popular query asked 1000 times:
// Without cache: 1000 × 50ms = 50 seconds
// With cache: 1 × 50ms + 999 × 0.1ms = ~150ms
// Speedup: 333x!
*/
