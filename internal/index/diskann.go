package index

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/phenomenon0/vectordb/internal/filter"
)

// DiskANNIndex implements a memory-mapped disk-backed vector index
// Designed for 100M+ vectors where memory is constrained
// Uses hybrid memory/disk architecture: shallow HNSW in memory + disk-backed deeper levels
type DiskANNIndex struct {
	mu sync.RWMutex

	dim     int
	count   int
	deleted map[uint64]bool

	// Memory-resident structures (LRU cache)
	lruCache      *LRUCache            // LRU cache for hot vectors
	memoryVectors map[uint64][]float32 // Legacy: kept for backward compatibility
	memoryLimit   int                  // Max vectors to keep in memory

	// Memory-mapped file
	mmapFile   *os.File
	mmapData   []byte
	mmapOffset int64

	// Index structure
	graphStore     GraphStore // Neighbor graph (abstracted for memory/disk backends)
	maxDegree      int       // Maximum edges per node
	efConstruction int                 // Expansion factor during construction
	efSearch       int                 // Expansion factor during search

	// Metadata storage (for filtered search)
	metadata   map[uint64]map[string]interface{}
	payloadIdx *PayloadIndex

	// Metrics
	metric       string
	diskReads    int64
	memoryHits   int64
	cacheHitRate float64

	// File path for persistence
	indexPath string

	// Optional quantization (for both memory and disk vectors)
	quantizer       Quantizer         // Quantizer for compression
	quantizedMemory map[uint64][]byte // Quantized memory-resident vectors
	diskOffsetIndex map[uint64]int64  // ID -> disk offset for quantized data

	// Mmap optimization settings
	mmapReadOnly    bool  // Use read-only mmap for searches (reduces memory)
	prefetchEnabled bool  // Enable prefetching for graph traversal
	compactOnClose  bool  // Compact deleted vectors on close
	maxMmapSize     int64 // Maximum mmap file size (default: 100GB)

	// Offset index for unquantized vectors (faster lookups, O(1) instead of O(n))
	unquantizedOffsetIndex map[uint64]int64

	// Raw config map for deferred lookups (validation limits, etc.)
	rawConfig map[string]interface{}
}

// DiskANNConfig holds configuration for DiskANN index
type DiskANNConfig struct {
	MemoryLimit     int    // Max vectors to keep in memory (default: 10000)
	MaxDegree       int    // Max edges per node (default: 32)
	EfConstruction  int    // Build-time search depth (default: 100)
	EfSearch        int    // Query-time search depth (default: 50)
	IndexPath       string // Path to index file
	Metric          string // "cosine" or "euclidean"
	PrefetchEnabled bool   // Enable mmap prefetching (default: true)
	CompactOnClose  bool   // Compact deleted vectors on close (default: false)
}

// NewDiskANNIndex creates a new DiskANN index
func NewDiskANNIndex(dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dim)
	}

	// Parse config
	memoryLimit := GetConfigInt(config, "memory_limit", 10000)
	maxDegree := GetConfigInt(config, "max_degree", 32)
	efConstruction := GetConfigInt(config, "ef_construction", 100)
	efSearch := GetConfigInt(config, "ef_search", 50)
	indexPath := GetConfigString(config, "index_path", "/tmp/diskann.idx")
	metric := GetConfigString(config, "metric", "cosine")

	// Validate index path to prevent path traversal attacks
	indexPath = filepath.Clean(indexPath)
	if strings.Contains(indexPath, "..") {
		return nil, fmt.Errorf("invalid index path: path traversal not allowed")
	}
	// Ensure path is under allowed directories (either /tmp or current working dir)
	if !strings.HasPrefix(indexPath, "/tmp/") && !strings.HasPrefix(indexPath, "./") && !filepath.IsAbs(indexPath) {
		// For relative paths without ./, prepend ./
		indexPath = "./" + indexPath
	}

	// Calculate LRU cache capacity: should be larger than memoryLimit to accommodate
	// working set during graph searches. Use 3x memory limit as a reasonable default.
	// This allows caching recently accessed disk vectors without thrashing.
	cacheCapacity := memoryLimit * 3
	bytesPerVector := int64(dim*4 + 100) // Float32 + metadata overhead
	maxBytes := int64(cacheCapacity) * bytesPerVector

	// Parse additional mmap optimization config
	prefetchEnabled := GetConfigBool(config, "prefetch_enabled", true)
	compactOnClose := GetConfigBool(config, "compact_on_close", false)
	// Default max mmap size: 100GB - prevents unbounded file growth
	maxMmapSize := GetConfigInt64(config, "max_mmap_size", 100*1024*1024*1024)
	graphStorage := GetConfigString(config, "graph_storage", "memory")

	// Create graph store based on config
	var graphStore GraphStore
	switch graphStorage {
	case "disk":
		graphPath := indexPath + ".graph"
		var err error
		gs, err := NewMmapGraphStore(graphPath, maxDegree)
		if err != nil {
			return nil, fmt.Errorf("failed to create mmap graph store: %w", err)
		}
		graphStore = gs
	default:
		graphStore = NewMemoryGraphStore()
	}

	idx := &DiskANNIndex{
		dim:                    dim,
		lruCache:               NewLRUCache(cacheCapacity, maxBytes),
		memoryVectors:          make(map[uint64][]float32),
		memoryLimit:            memoryLimit,
		deleted:                make(map[uint64]bool),
		graphStore:             graphStore,
		metadata:               make(map[uint64]map[string]interface{}),
		payloadIdx:             NewPayloadIndex(),
		maxDegree:              maxDegree,
		efConstruction:         efConstruction,
		efSearch:               efSearch,
		metric:                 metric,
		indexPath:              indexPath,
		prefetchEnabled:        prefetchEnabled,
		compactOnClose:         compactOnClose,
		maxMmapSize:            maxMmapSize,
		unquantizedOffsetIndex: make(map[uint64]int64),
		rawConfig:              config,
	}

	// Check for quantization config
	if quantConfig, ok := config["quantization"].(map[string]interface{}); ok {
		quantType := GetConfigString(quantConfig, "type", "")

		switch quantType {
		case "float16":
			idx.quantizer = NewFloat16Quantizer(dim)
			idx.quantizedMemory = make(map[uint64][]byte)
			idx.diskOffsetIndex = make(map[uint64]int64)
		case "uint8":
			idx.quantizer = NewUint8Quantizer(dim)
			idx.quantizedMemory = make(map[uint64][]byte)
			idx.diskOffsetIndex = make(map[uint64]int64)
		case "pq":
			m := GetConfigInt(quantConfig, "m", 8)
			ksub := GetConfigInt(quantConfig, "ksub", 256)
			pq, err := NewProductQuantizer(dim, m, ksub)
			if err != nil {
				return nil, fmt.Errorf("failed to create product quantizer: %w", err)
			}
			idx.quantizer = pq
			idx.quantizedMemory = make(map[uint64][]byte)
			idx.diskOffsetIndex = make(map[uint64]int64)
		}
	}

	// Initialize memory-mapped file
	if err := idx.initMmap(); err != nil {
		return nil, fmt.Errorf("failed to initialize mmap: %w", err)
	}

	return idx, nil
}

func init() {
	// Register DiskANN index type
	Register("diskann", NewDiskANNIndex)
}

// Name returns the index type name
func (d *DiskANNIndex) Name() string {
	return "DiskANN"
}

// initMmap initializes the memory-mapped file
func (d *DiskANNIndex) initMmap() error {
	// Create or open index file
	file, err := os.OpenFile(d.indexPath, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return fmt.Errorf("failed to open index file: %w", err)
	}

	d.mmapFile = file

	// Check existing file size
	stat, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat file: %w", err)
	}

	initialSize := stat.Size()
	if initialSize < 64*1024*1024 {
		// Initialize with minimum size (will grow as needed)
		initialSize = 64 * 1024 * 1024 // 64 MB
		if err := d.mmapFile.Truncate(initialSize); err != nil {
			return fmt.Errorf("failed to truncate file: %w", err)
		}
	}

	// Memory map the file
	data, err := mmapCreate(int(d.mmapFile.Fd()), int(initialSize))
	if err != nil {
		return err
	}

	d.mmapData = data
	d.mmapOffset = 0

	return nil
}

// Add adds a vector to the DiskANN index
func (d *DiskANNIndex) Add(ctx context.Context, id uint64, vector []float32) error {
	if len(vector) != d.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", d.dim, len(vector))
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Make a copy to avoid external mutations
	vecCopy := make([]float32, d.dim)
	copy(vecCopy, vector)

	// Decide: memory or disk storage
	if len(d.memoryVectors)+len(d.quantizedMemory) < d.memoryLimit {
		// Store in memory (hot)
		if d.quantizer != nil {
			// Quantize and store compressed data
			quantized, err := d.quantizer.Quantize(vecCopy)
			if err != nil {
				return fmt.Errorf("failed to quantize vector: %w", err)
			}
			d.quantizedMemory[id] = quantized
		} else {
			// Store unquantized
			d.memoryVectors[id] = vecCopy
		}
	} else {
		// Store on disk (cold)
		if err := d.writeToDisk(id, vecCopy); err != nil {
			return fmt.Errorf("failed to write to disk: %w", err)
		}
	}

	// Build graph edges
	if err := d.buildEdges(id, vector); err != nil {
		return fmt.Errorf("failed to build edges: %w", err)
	}

	delete(d.deleted, id)
	d.count++

	return nil
}

// SetMetadata sets or updates the metadata for a vector.
// This is used for filtered search. Metadata can be set after vector insertion.
func (d *DiskANNIndex) SetMetadata(id uint64, metadata map[string]interface{}) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Check if vector exists
	if _, existsMem := d.memoryVectors[id]; !existsMem {
		if _, existsQuant := d.quantizedMemory[id]; !existsQuant {
			if _, existsDisk := d.diskOffsetIndex[id]; !existsDisk && d.quantizer != nil {
				return fmt.Errorf("vector with ID %d does not exist", id)
			}
			// For unquantized disk vectors, check if ID is in graph
			if !d.graphStore.HasNode(id) {
				return fmt.Errorf("vector with ID %d does not exist", id)
			}
		}
	}

	// Store metadata (make a copy to avoid external mutations)
	if metadata != nil {
		metaCopy := make(map[string]interface{}, len(metadata))
		for k, v := range metadata {
			metaCopy[k] = v
		}
		// Remove old payload index entries before overwriting
		if old := d.metadata[id]; old != nil {
			d.payloadIdx.Remove(id, old)
		}
		d.metadata[id] = metaCopy
		d.payloadIdx.Index(id, metaCopy)
	} else {
		// Allow nil to clear metadata
		if old := d.metadata[id]; old != nil {
			d.payloadIdx.Remove(id, old)
		}
		delete(d.metadata, id)
	}

	return nil
}

// writeToDisk writes a vector to the memory-mapped file
func (d *DiskANNIndex) writeToDisk(id uint64, vector []float32) error {
	var dataToWrite []byte
	var needed int

	if d.quantizer != nil {
		// Quantize vector
		quantized, err := d.quantizer.Quantize(vector)
		if err != nil {
			return fmt.Errorf("failed to quantize vector: %w", err)
		}

		// Calculate space needed: 8 bytes (id) + 4 bytes (length) + quantized data
		dataToWrite = quantized
		needed = 8 + 4 + len(quantized)
	} else {
		// Calculate space needed: 8 bytes (id) + 4*dim bytes (vector)
		needed = 8 + d.dim*4
	}

	// Check if we need to grow the mmap
	if int(d.mmapOffset)+needed > len(d.mmapData) {
		if err := d.growMmap(); err != nil {
			return err
		}
	}

	// Record offset for this ID (for both quantized and unquantized)
	if d.quantizer != nil {
		d.diskOffsetIndex[id] = d.mmapOffset
	} else {
		d.unquantizedOffsetIndex[id] = d.mmapOffset
	}

	// Write ID
	binary.LittleEndian.PutUint64(d.mmapData[d.mmapOffset:], id)
	d.mmapOffset += 8

	if d.quantizer != nil {
		// Write quantized data length
		binary.LittleEndian.PutUint32(d.mmapData[d.mmapOffset:], uint32(len(dataToWrite)))
		d.mmapOffset += 4

		// Write quantized data
		copy(d.mmapData[d.mmapOffset:], dataToWrite)
		d.mmapOffset += int64(len(dataToWrite))
	} else {
		// Write vector (unquantized)
		for _, v := range vector {
			binary.LittleEndian.PutUint32(
				d.mmapData[d.mmapOffset:],
				math.Float32bits(v),
			)
			d.mmapOffset += 4
		}
	}

	return nil
}

// growMmap doubles the size of the memory-mapped file
func (d *DiskANNIndex) growMmap() error {
	// Unmap current mapping
	if err := mmapUnmap(d.mmapData); err != nil {
		return fmt.Errorf("failed to unmap: %w", err)
	}

	// Double the file size
	newSize := int64(len(d.mmapData) * 2)

	// Check against maximum allowed mmap size to prevent unbounded growth
	if d.maxMmapSize > 0 && newSize > d.maxMmapSize {
		return fmt.Errorf("mmap size limit exceeded: requested %d bytes, max allowed %d bytes", newSize, d.maxMmapSize)
	}

	if err := d.mmapFile.Truncate(newSize); err != nil {
		return fmt.Errorf("failed to grow file: %w", err)
	}

	// Remap with new size
	data, err := mmapCreate(int(d.mmapFile.Fd()), int(newSize))
	if err != nil {
		return err
	}

	d.mmapData = data
	return nil
}

// buildEdges constructs graph edges using greedy search
func (d *DiskANNIndex) buildEdges(newID uint64, newVec []float32) error {
	// Find nearest neighbors
	candidates := d.greedySearch(newVec, d.efConstruction)

	// Connect to top-k neighbors
	edges := make([]uint64, 0, d.maxDegree)
	for _, c := range candidates {
		if len(edges) >= d.maxDegree {
			break
		}
		if c.ID != newID && !d.deleted[c.ID] {
			edges = append(edges, c.ID)
		}
	}

	d.graphStore.SetNeighbors(newID, edges)

	// Add reverse edges (bidirectional)
	for _, neighborID := range edges {
		if neighbors := d.graphStore.GetNeighbors(neighborID); neighbors != nil {
			// Add new edge if not at max degree
			if len(neighbors) < d.maxDegree {
				d.graphStore.SetNeighbors(neighborID, append(neighbors, newID))
			} else {
				// Replace farthest neighbor
				neighborVec, err := d.getVector(neighborID)
				if err != nil {
					continue
				}
				farthestIdx := d.findFarthest(neighborVec, neighbors)
				neighbors[farthestIdx] = newID
			}
		}
	}

	return nil
}

// greedySearch performs greedy graph search
func (d *DiskANNIndex) greedySearch(query []float32, ef int) []Result {
	if d.graphStore.Len() == 0 {
		return []Result{}
	}

	// Start from entry point
	entryID, found := FirstNodeID(d.graphStore, d.deleted)
	if !found {
		return []Result{}
	}

	visited := make(map[uint64]bool)
	candidates := make([]Result, 0, ef)

	// Get entry vector
	entryVec, err := d.getVector(entryID)
	if err != nil {
		return []Result{}
	}

	entryDist := d.distance(query, entryVec)
	candidates = append(candidates, Result{ID: entryID, Distance: entryDist})
	visited[entryID] = true

	// Greedy expansion
	for i := 0; i < ef; i++ {
		if i >= len(candidates) {
			break
		}

		current := candidates[i]
		neighbors := d.graphStore.GetNeighbors(current.ID)

		for _, nID := range neighbors {
			if visited[nID] || d.deleted[nID] {
				continue
			}

			nVec, err := d.getVector(nID)
			if err != nil {
				continue
			}

			dist := d.distance(query, nVec)
			candidates = append(candidates, Result{ID: nID, Distance: dist})
			visited[nID] = true
		}

		// Sort by distance (keep best candidates)
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})

		if len(candidates) > ef*2 {
			candidates = candidates[:ef*2]
		}
	}

	// Return top ef results
	if len(candidates) > ef {
		candidates = candidates[:ef]
	}

	return candidates
}

// getVector retrieves a vector from memory or disk
func (d *DiskANNIndex) getVector(id uint64) ([]float32, error) {
	// Try memory-resident vectors first (unquantized)
	if vec, ok := d.memoryVectors[id]; ok {
		d.memoryHits++
		return vec, nil
	}

	// Try quantized memory
	if quantized, ok := d.quantizedMemory[id]; ok {
		d.memoryHits++
		vec, err := d.quantizer.Dequantize(quantized)
		if err != nil {
			return nil, fmt.Errorf("failed to dequantize memory vector %d: %w", id, err)
		}
		return vec, nil
	}

	// Try LRU cache (caches disk reads)
	if cached, ok := d.lruCache.Get(id); ok {
		d.memoryHits++

		// Handle both float32 and quantized ([]byte) vectors
		switch v := cached.(type) {
		case []float32:
			return v, nil
		case []byte:
			// Dequantize if needed
			if d.quantizer != nil {
				vec, err := d.quantizer.Dequantize(v)
				if err != nil {
					return nil, fmt.Errorf("failed to dequantize cached vector %d: %w", id, err)
				}
				return vec, nil
			}
			// If we have []byte but no quantizer, remove from cache and continue to disk read
			d.lruCache.Remove(id)
		}
	}

	// Read from disk (cache miss)
	d.diskReads++
	vec, err := d.readFromDisk(id)
	if err != nil {
		return nil, err
	}

	// Add to LRU cache for future disk reads
	// This caches disk-read vectors to reduce future disk I/O
	if d.quantizer != nil {
		// Store quantized in cache to save memory
		quantized, qerr := d.quantizer.Quantize(vec)
		if qerr == nil {
			d.lruCache.Put(id, quantized)
		}
	} else {
		d.lruCache.Put(id, vec)
	}

	return vec, nil
}

// readFromDisk reads a vector from the memory-mapped file
func (d *DiskANNIndex) readFromDisk(id uint64) ([]float32, error) {
	if d.quantizer != nil {
		// Use offset index for quantized data (variable length records)
		if offset, ok := d.diskOffsetIndex[id]; ok {
			return d.readQuantizedFromOffset(id, offset)
		}
		return nil, fmt.Errorf("vector %d not found in offset index", id)
	}

	// Unquantized: use offset index for O(1) lookup
	if offset, ok := d.unquantizedOffsetIndex[id]; ok {
		return d.readUnquantizedFromOffset(id, offset)
	}

	// Fallback to linear scan for backward compatibility (imported indices)
	return d.readFromDiskLinearScan(id)
}

// readQuantizedFromOffset reads a quantized vector from a known offset
func (d *DiskANNIndex) readQuantizedFromOffset(id uint64, offset int64) ([]float32, error) {
	// Bounds check
	if offset < 0 || offset+12 > int64(len(d.mmapData)) {
		return nil, fmt.Errorf("offset %d out of bounds for vector %d", offset, id)
	}

	// Read ID (verification)
	storedID := binary.LittleEndian.Uint64(d.mmapData[offset:])
	if storedID != id {
		return nil, fmt.Errorf("offset index corruption: expected ID %d, got %d", id, storedID)
	}

	// Read quantized data length
	dataLen := binary.LittleEndian.Uint32(d.mmapData[offset+8:])

	// Bounds check for data
	if offset+12+int64(dataLen) > int64(len(d.mmapData)) {
		return nil, fmt.Errorf("quantized data extends beyond mmap for vector %d", id)
	}

	// Read quantized data directly from mmap (zero-copy reference)
	quantized := d.mmapData[offset+12 : offset+12+int64(dataLen)]

	// Prefetch next likely access if enabled
	if d.prefetchEnabled && offset+12+int64(dataLen)+4096 < int64(len(d.mmapData)) {
		_ = d.mmapData[offset+12+int64(dataLen)+4096] // Touch next page
	}

	// Dequantize
	vec, err := d.quantizer.Dequantize(quantized)
	if err != nil {
		return nil, fmt.Errorf("failed to dequantize disk vector %d: %w", id, err)
	}
	return vec, nil
}

// readUnquantizedFromOffset reads an unquantized vector from a known offset
func (d *DiskANNIndex) readUnquantizedFromOffset(id uint64, offset int64) ([]float32, error) {
	recordSize := int64(8 + d.dim*4)

	// Bounds check
	if offset < 0 || offset+recordSize > int64(len(d.mmapData)) {
		return nil, fmt.Errorf("offset %d out of bounds for vector %d", offset, id)
	}

	// Read ID (verification)
	storedID := binary.LittleEndian.Uint64(d.mmapData[offset:])
	if storedID != id {
		return nil, fmt.Errorf("offset index corruption: expected ID %d, got %d", id, storedID)
	}

	// Read vector directly from mmap
	vec := make([]float32, d.dim)
	for i := 0; i < d.dim; i++ {
		bits := binary.LittleEndian.Uint32(d.mmapData[offset+8+int64(i*4):])
		vec[i] = math.Float32frombits(bits)
	}

	// Prefetch next likely access if enabled
	if d.prefetchEnabled && offset+recordSize+4096 < int64(len(d.mmapData)) {
		_ = d.mmapData[offset+recordSize+4096] // Touch next page
	}

	return vec, nil
}

// readFromDiskLinearScan performs a linear scan for backward compatibility.
// Returns the vector and the discovered offset. The caller is responsible for
// caching the offset — this function does NOT mutate shared state, avoiding
// the data race that occurs when called under RLock (search path).
func (d *DiskANNIndex) readFromDiskLinearScan(id uint64) ([]float32, error) {
	offset := int64(0)
	recordSize := int64(8 + d.dim*4)

	for offset < d.mmapOffset {
		// Read ID
		storedID := binary.LittleEndian.Uint64(d.mmapData[offset:])

		if storedID == id {
			// Read vector
			vec := make([]float32, d.dim)
			for i := 0; i < d.dim; i++ {
				bits := binary.LittleEndian.Uint32(d.mmapData[offset+8+int64(i*4):])
				vec[i] = math.Float32frombits(bits)
			}

			// Cache the discovered offset for future O(1) lookups.
			// This requires a write lock — upgrade from the caller's RLock.
			// Use a goroutine to avoid blocking the search path.
			go func() {
				d.mu.Lock()
				d.unquantizedOffsetIndex[id] = offset
				d.mu.Unlock()
			}()

			return vec, nil
		}

		// Skip to next record using recordSize
		offset += recordSize
	}

	return nil, fmt.Errorf("vector %d not found on disk", id)
}

// Search performs approximate nearest neighbor search
func (d *DiskANNIndex) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	if len(query) != d.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", d.dim, len(query))
	}

	d.mu.RLock()
	defer d.mu.RUnlock()

	// Extract filter from params if provided
	var filterFunc filter.Filter
	// DiskANN doesn't have specific search params yet, but check anyway for future compatibility
	switch p := params.(type) {
	case HNSWSearchParams:
		filterFunc = p.Filter
	case *HNSWSearchParams:
		if p != nil {
			filterFunc = p.Filter
		}
	case IVFSearchParams:
		filterFunc = p.Filter
	case *IVFSearchParams:
		if p != nil {
			filterFunc = p.Filter
		}
	}

	// Try payload index for O(1) lookups on indexed fields
	var payloadCandidates map[uint64]struct{}
	if filterFunc != nil {
		if bitmap, ok := d.payloadIdx.QueryBitmap(filterFunc); ok {
			selectivity := float64(len(bitmap)) / float64(d.count)
			if selectivity < 0.15 {
				payloadCandidates = bitmap
			}
		}
	}

	// Use greedy search with efSearch expansion
	candidates := d.greedySearch(query, d.efSearch)

	// Filter deleted and apply metadata filter, return top-k
	results := make([]Result, 0, k)
	for _, c := range candidates {
		if d.deleted[c.ID] {
			continue
		}

		// Apply metadata filter if provided
		if filterFunc != nil {
			if payloadCandidates != nil {
				if _, match := payloadCandidates[c.ID]; !match {
					continue
				}
			} else {
				meta := d.metadata[c.ID]
				if meta == nil || !filterFunc.Evaluate(meta) {
					continue
				}
			}
		}

		c.Score = 1.0 / (1.0 + c.Distance)
		c.Metadata = d.metadata[c.ID]
		results = append(results, c)
		if len(results) >= k {
			break
		}
	}

	// Update cache hit rate
	if d.diskReads+d.memoryHits > 0 {
		d.cacheHitRate = float64(d.memoryHits) / float64(d.diskReads+d.memoryHits)
	}

	return results, nil
}

// Delete marks a vector as deleted
func (d *DiskANNIndex) Delete(ctx context.Context, id uint64) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.deleted[id] = true

	// Remove from memory if present
	delete(d.memoryVectors, id)
	delete(d.quantizedMemory, id)

	// Remove from LRU cache
	d.lruCache.Remove(id)

	// Note: We keep offset indices for potential undelete/recovery
	// Actual disk space is reclaimed during compaction
	// delete(d.diskOffsetIndex, id)
	// delete(d.unquantizedOffsetIndex, id)

	return nil
}

// Stats returns index statistics
func (d *DiskANNIndex) Stats() IndexStats {
	d.mu.RLock()
	defer d.mu.RUnlock()

	// Get LRU cache statistics
	cacheStats := d.lruCache.Stats()

	// Count hot memory-resident vectors (from legacy maps, not cache)
	// These are the vectors that fit within memoryLimit
	hotMemoryVectors := len(d.memoryVectors) + len(d.quantizedMemory)

	// Calculate memory usage for hot vectors
	var hotMemoryBytes int64
	if d.quantizer != nil {
		for _, data := range d.quantizedMemory {
			hotMemoryBytes += int64(8 + len(data))
		}
	} else {
		hotMemoryBytes = int64(len(d.memoryVectors) * (8 + d.dim*4))
	}

	// Total memory includes hot vectors + LRU cache
	totalMemoryBytes := hotMemoryBytes + cacheStats.MemoryUsed

	diskBytes := int(d.mmapOffset)
	graphBytes := int64(d.graphStore.Len() * 8 * d.maxDegree)

	// Calculate cache hit rate from LRU cache stats
	cacheHitRate := cacheStats.HitRate

	diskVectors := d.count - hotMemoryVectors

	// Count indexed vectors (those with offset index entries)
	indexedDiskVectors := len(d.diskOffsetIndex) + len(d.unquantizedOffsetIndex)

	extra := map[string]interface{}{
		"memory_vectors":       hotMemoryVectors, // Only hot vectors within memoryLimit
		"disk_vectors":         diskVectors,
		"indexed_disk_vectors": indexedDiskVectors, // Vectors with O(1) lookup
		"memory_limit":         d.memoryLimit,
		"max_degree":           d.maxDegree,
		"ef_construction":      d.efConstruction,
		"ef_search":            d.efSearch,
		"disk_reads":           d.diskReads,
		"memory_hits":          d.memoryHits,
		"cache_hit_rate":       cacheHitRate,
		"cache_hits":           cacheStats.Hits,
		"cache_misses":         cacheStats.Misses,
		"cache_evictions":      cacheStats.Evictions,
		"cache_size":           cacheStats.Size,
		"cache_capacity":       cacheStats.Capacity,
		"mmap_size_mb":         len(d.mmapData) / (1024 * 1024),
		"mmap_used_mb":         d.mmapOffset / (1024 * 1024),
		"metric":               d.metric,
		"prefetch_enabled":     d.prefetchEnabled,
		"deleted_count":        len(d.deleted),
	}

	// Add quantization info if enabled
	if d.quantizer != nil {
		extra["quantization"] = d.quantizer.Type()
		extra["quantized_bytes_per_vector"] = d.quantizer.BytesPerVector()

		// Calculate compression ratio
		originalSize := d.dim * 4
		compressedSize := d.quantizer.BytesPerVector()
		extra["compression_ratio"] = float64(originalSize) / float64(compressedSize)

		// Note about hybrid storage
		extra["note"] = "both memory and disk vectors are quantized"
	}

	return IndexStats{
		Name:       "DiskANN",
		Dim:        d.dim,
		Count:      d.count,
		Deleted:    len(d.deleted),
		Active:     d.count - len(d.deleted),
		MemoryUsed: totalMemoryBytes + graphBytes,
		DiskUsed:   int64(diskBytes),
		Extra:      extra,
	}
}

const diskANNExportMagic = "DANN2"

type diskANNVectorRecord struct {
	ID       uint64
	InMemory bool
	Vector   []float32
}

type diskANNExportPayload struct {
	Dim            int
	Count          int
	MemoryLimit    int
	MaxDegree      int
	EfConstruction int
	EfSearch       int
	Metric         string
	Prefetch       bool
	CompactOnClose bool
	MaxMmapSize    int64

	Quantization QuantizationType
	Uint8Min     []float32
	Uint8Max     []float32
	PQCodebooks  [][][]float32
	PQM          int
	PQKSub       int

	Graph    map[uint64][]uint64
	Deleted  map[uint64]bool
	Metadata map[uint64][]byte
	Vectors  []diskANNVectorRecord
}

func (d *DiskANNIndex) exportPayloadLocked() (*diskANNExportPayload, error) {
	records, err := d.exportVectorRecordsLocked()
	if err != nil {
		return nil, err
	}

	payload := &diskANNExportPayload{
		Dim:            d.dim,
		Count:          d.count,
		MemoryLimit:    d.memoryLimit,
		MaxDegree:      d.maxDegree,
		EfConstruction: d.efConstruction,
		EfSearch:       d.efSearch,
		Metric:         d.metric,
		Prefetch:       d.prefetchEnabled,
		CompactOnClose: d.compactOnClose,
		MaxMmapSize:    d.maxMmapSize,
		Quantization:   QuantizationNone,
		Graph:          d.graphStore.Clone(),
		Deleted:        cloneDiskANNDeleted(d.deleted),
		Metadata:       make(map[uint64][]byte, len(d.metadata)),
		Vectors:        records,
	}

	for id, meta := range d.metadata {
		data, err := json.Marshal(meta)
		if err != nil {
			return nil, fmt.Errorf("marshal metadata for %d: %w", id, err)
		}
		payload.Metadata[id] = data
	}

	switch q := d.quantizer.(type) {
	case *Float16Quantizer:
		payload.Quantization = q.Type()
	case *Uint8Quantizer:
		payload.Quantization = q.Type()
		min, max := q.GetCodebook()
		payload.Uint8Min = append([]float32(nil), min...)
		payload.Uint8Max = append([]float32(nil), max...)
	case *ProductQuantizer:
		payload.Quantization = q.Type()
		payload.PQCodebooks = clonePQCodebooks(q.GetCodebooks())
		payload.PQM = q.m
		payload.PQKSub = q.ksub
	}

	return payload, nil
}

func (d *DiskANNIndex) exportVectorRecordsLocked() ([]diskANNVectorRecord, error) {
	idSet := make(map[uint64]struct{}, len(d.memoryVectors)+len(d.quantizedMemory)+len(d.diskOffsetIndex)+len(d.unquantizedOffsetIndex))
	for id := range d.memoryVectors {
		idSet[id] = struct{}{}
	}
	for id := range d.quantizedMemory {
		idSet[id] = struct{}{}
	}
	for id := range d.diskOffsetIndex {
		idSet[id] = struct{}{}
	}
	for id := range d.unquantizedOffsetIndex {
		idSet[id] = struct{}{}
	}

	ids := make([]uint64, 0, len(idSet))
	for id := range idSet {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

	records := make([]diskANNVectorRecord, 0, len(ids))
	for _, id := range ids {
		vec, err := d.vectorForExportLocked(id)
		if err != nil {
			return nil, fmt.Errorf("export vector %d: %w", id, err)
		}
		records = append(records, diskANNVectorRecord{
			ID:       id,
			InMemory: d.vectorStoredInMemoryLocked(id),
			Vector:   vec,
		})
	}

	return records, nil
}

func (d *DiskANNIndex) vectorStoredInMemoryLocked(id uint64) bool {
	if _, ok := d.memoryVectors[id]; ok {
		return true
	}
	_, ok := d.quantizedMemory[id]
	return ok
}

func (d *DiskANNIndex) vectorForExportLocked(id uint64) ([]float32, error) {
	if vec, ok := d.memoryVectors[id]; ok {
		cp := make([]float32, len(vec))
		copy(cp, vec)
		return cp, nil
	}

	if quantized, ok := d.quantizedMemory[id]; ok {
		vec, err := d.quantizer.Dequantize(quantized)
		if err != nil {
			return nil, fmt.Errorf("dequantize memory vector: %w", err)
		}
		return append([]float32(nil), vec...), nil
	}

	vec, err := d.readFromDisk(id)
	if err != nil {
		return nil, err
	}
	return append([]float32(nil), vec...), nil
}

func cloneDiskANNGraph(src map[uint64][]uint64) map[uint64][]uint64 {
	dst := make(map[uint64][]uint64, len(src))
	for id, neighbors := range src {
		dst[id] = append([]uint64(nil), neighbors...)
	}
	return dst
}

func cloneDiskANNDeleted(src map[uint64]bool) map[uint64]bool {
	dst := make(map[uint64]bool, len(src))
	for id, deleted := range src {
		dst[id] = deleted
	}
	return dst
}

func clonePQCodebooks(src [][][]float32) [][][]float32 {
	if src == nil {
		return nil
	}
	dst := make([][][]float32, len(src))
	for i := range src {
		dst[i] = make([][]float32, len(src[i]))
		for j := range src[i] {
			dst[i][j] = append([]float32(nil), src[i][j]...)
		}
	}
	return dst
}

// Export serializes the DiskANN index.
// Uses a write lock because the linear-scan fallback in readFromDisk
// mutates unquantizedOffsetIndex (FIX #6: was RLock, causing data race).
func (d *DiskANNIndex) Export() ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Flush mmap to ensure disk consistency
	if err := mmapSync(d.mmapData); err != nil {
		return nil, fmt.Errorf("failed to sync mmap: %w", err)
	}

	payload, err := d.exportPayloadLocked()
	if err != nil {
		return nil, err
	}

	var encoded bytes.Buffer
	encoded.WriteString(diskANNExportMagic)
	if err := gob.NewEncoder(&encoded).Encode(payload); err != nil {
		return nil, fmt.Errorf("encode diskann export: %w", err)
	}

	return encoded.Bytes(), nil
}

// Import deserializes the DiskANN index
func (d *DiskANNIndex) Import(data []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if bytes.HasPrefix(data, []byte(diskANNExportMagic)) {
		var payload diskANNExportPayload
		if err := gob.NewDecoder(bytes.NewReader(data[len(diskANNExportMagic):])).Decode(&payload); err != nil {
			return fmt.Errorf("decode diskann export: %w", err)
		}
		return d.importPayloadLocked(&payload)
	}

	return d.importLegacyLocked(data)
}

func (d *DiskANNIndex) importPayloadLocked(payload *diskANNExportPayload) error {
	currentIndexPath := d.indexPath
	currentReadOnly := d.mmapReadOnly

	d.dim = payload.Dim
	d.count = payload.Count
	d.memoryLimit = payload.MemoryLimit
	d.maxDegree = payload.MaxDegree
	d.efConstruction = payload.EfConstruction
	d.efSearch = payload.EfSearch
	d.metric = payload.Metric
	d.prefetchEnabled = payload.Prefetch
	d.compactOnClose = payload.CompactOnClose
	d.maxMmapSize = payload.MaxMmapSize
	d.indexPath = currentIndexPath
	d.mmapReadOnly = currentReadOnly

	if err := d.restoreQuantizerLocked(payload); err != nil {
		return err
	}
	if err := d.resetStorageForImportLocked(); err != nil {
		return err
	}

	d.graphStore.ReplaceAll(cloneDiskANNGraph(payload.Graph))
	d.deleted = cloneDiskANNDeleted(payload.Deleted)
	d.metadata = make(map[uint64]map[string]interface{}, len(payload.Metadata))
	for id, raw := range payload.Metadata {
		var meta map[string]interface{}
		if err := json.Unmarshal(raw, &meta); err != nil {
			return fmt.Errorf("unmarshal metadata for %d: %w", id, err)
		}
		d.metadata[id] = meta
	}

	for _, record := range payload.Vectors {
		vecCopy := append([]float32(nil), record.Vector...)
		if record.InMemory {
			if d.quantizer != nil {
				quantized, err := d.quantizer.Quantize(vecCopy)
				if err != nil {
					return fmt.Errorf("restore in-memory vector %d: %w", record.ID, err)
				}
				d.quantizedMemory[record.ID] = quantized
			} else {
				d.memoryVectors[record.ID] = vecCopy
			}
			continue
		}

		if err := d.writeToDisk(record.ID, vecCopy); err != nil {
			return fmt.Errorf("restore disk vector %d: %w", record.ID, err)
		}
	}

	// Rebuild payload index from restored metadata
	d.payloadIdx = NewPayloadIndex()
	for id, meta := range d.metadata {
		if meta != nil {
			d.payloadIdx.Index(id, meta)
		}
	}

	return nil
}

func (d *DiskANNIndex) restoreQuantizerLocked(payload *diskANNExportPayload) error {
	switch payload.Quantization {
	case QuantizationNone:
		d.quantizer = nil
		d.quantizedMemory = nil
		d.diskOffsetIndex = nil
	case QuantizationFloat16:
		d.quantizer = NewFloat16Quantizer(payload.Dim)
		d.quantizedMemory = make(map[uint64][]byte)
		d.diskOffsetIndex = make(map[uint64]int64)
	case QuantizationUint8:
		q := NewUint8Quantizer(payload.Dim)
		if err := q.SetCodebook(append([]float32(nil), payload.Uint8Min...), append([]float32(nil), payload.Uint8Max...)); err != nil {
			return fmt.Errorf("restore uint8 quantizer: %w", err)
		}
		d.quantizer = q
		d.quantizedMemory = make(map[uint64][]byte)
		d.diskOffsetIndex = make(map[uint64]int64)
	case QuantizationProduct:
		q, err := NewProductQuantizer(payload.Dim, payload.PQM, payload.PQKSub)
		if err != nil {
			return fmt.Errorf("create pq quantizer: %w", err)
		}
		if err := q.SetCodebooks(clonePQCodebooks(payload.PQCodebooks)); err != nil {
			return fmt.Errorf("restore pq quantizer: %w", err)
		}
		d.quantizer = q
		d.quantizedMemory = make(map[uint64][]byte)
		d.diskOffsetIndex = make(map[uint64]int64)
	default:
		return fmt.Errorf("unsupported quantization type %d", payload.Quantization)
	}

	return nil
}

func (d *DiskANNIndex) resetStorageForImportLocked() error {
	if d.mmapData != nil {
		if err := mmapUnmap(d.mmapData); err != nil {
			return fmt.Errorf("failed to unmap existing mmap: %w", err)
		}
		d.mmapData = nil
	}
	if d.mmapFile != nil {
		if err := d.mmapFile.Close(); err != nil {
			return fmt.Errorf("failed to close existing mmap file: %w", err)
		}
		d.mmapFile = nil
	}
	if err := os.Remove(d.indexPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to reset diskann file: %w", err)
	}

	cacheCapacity := d.memoryLimit * 3
	if cacheCapacity < 1 {
		cacheCapacity = 1
	}
	bytesPerVector := int64(d.dim*4 + 100)
	maxBytes := int64(cacheCapacity) * bytesPerVector

	d.lruCache = NewLRUCache(cacheCapacity, maxBytes)
	d.memoryVectors = make(map[uint64][]float32)
	d.graphStore = NewMemoryGraphStore()
	d.deleted = make(map[uint64]bool)
	d.metadata = make(map[uint64]map[string]interface{})
	d.unquantizedOffsetIndex = make(map[uint64]int64)
	if d.quantizer != nil {
		d.quantizedMemory = make(map[uint64][]byte)
		d.diskOffsetIndex = make(map[uint64]int64)
	} else {
		d.quantizedMemory = nil
		d.diskOffsetIndex = nil
	}
	d.diskReads = 0
	d.memoryHits = 0
	d.cacheHitRate = 0
	d.mmapOffset = 0

	return d.initMmap()
}

func (d *DiskANNIndex) importLegacyLocked(data []byte) error {

	dataLen := len(data)
	offset := 0

	// Helper to safely read uint32 with bounds check
	readUint32 := func() (uint32, error) {
		if offset+4 > dataLen {
			return 0, fmt.Errorf("data truncated at offset %d (need 4 bytes, have %d)", offset, dataLen-offset)
		}
		v := binary.LittleEndian.Uint32(data[offset:])
		offset += 4
		return v, nil
	}

	// Helper to safely read uint64 with bounds check
	readUint64 := func() (uint64, error) {
		if offset+8 > dataLen {
			return 0, fmt.Errorf("data truncated at offset %d (need 8 bytes, have %d)", offset, dataLen-offset)
		}
		v := binary.LittleEndian.Uint64(data[offset:])
		offset += 8
		return v, nil
	}

	// Helper to safely read bytes with bounds check
	readBytes := func(n int) ([]byte, error) {
		if n < 0 || offset+n > dataLen {
			return nil, fmt.Errorf("data truncated at offset %d (need %d bytes, have %d)", offset, n, dataLen-offset)
		}
		b := data[offset : offset+n]
		offset += n
		return b, nil
	}

	var err error
	var v32 uint32

	// Header
	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading dim: %w", err)
	}
	d.dim = int(v32)

	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading count: %w", err)
	}
	d.count = int(v32)

	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading memoryLimit: %w", err)
	}
	d.memoryLimit = int(v32)

	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading maxDegree: %w", err)
	}
	d.maxDegree = int(v32)

	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading efConstruction: %w", err)
	}
	d.efConstruction = int(v32)

	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading efSearch: %w", err)
	}
	d.efSearch = int(v32)

	// Metric
	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading metric length: %w", err)
	}
	metricBytes, err := readBytes(int(v32))
	if err != nil {
		return fmt.Errorf("reading metric: %w", err)
	}
	d.metric = string(metricBytes)

	// Index path
	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading path length: %w", err)
	}
	pathBytes, err := readBytes(int(v32))
	if err != nil {
		return fmt.Errorf("reading path: %w", err)
	}
	_ = pathBytes

	if err := d.resetStorageForImportLocked(); err != nil {
		return fmt.Errorf("failed to reinit mmap: %w", err)
	}

	// Graph structure
	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading graph size: %w", err)
	}
	graphSize := int(v32)

	// Sanity check to prevent excessive allocation.
	// Configurable via max_graph_size in index config (default 1 billion).
	maxGraphSize := GetConfigInt(d.rawConfig, "max_graph_size", 1_000_000_000)
	if graphSize < 0 || graphSize > maxGraphSize {
		return fmt.Errorf("invalid graph size: %d (max %d)", graphSize, maxGraphSize)
	}

	graphMap := make(map[uint64][]uint64, graphSize)
	for i := 0; i < graphSize; i++ {
		id, err := readUint64()
		if err != nil {
			return fmt.Errorf("reading graph node %d id: %w", i, err)
		}

		nc, err := readUint32()
		if err != nil {
			return fmt.Errorf("reading graph node %d neighbor count: %w", i, err)
		}
		neighborsCount := int(nc)

		// Sanity check neighbor count.
		// Configurable via max_neighbors in index config (default 100,000).
		maxNeighbors := GetConfigInt(d.rawConfig, "max_neighbors", 100_000)
		if neighborsCount < 0 || neighborsCount > maxNeighbors {
			return fmt.Errorf("invalid neighbor count for node %d: %d (max %d)", i, neighborsCount, maxNeighbors)
		}

		neighbors := make([]uint64, neighborsCount)
		for j := 0; j < neighborsCount; j++ {
			neighbors[j], err = readUint64()
			if err != nil {
				return fmt.Errorf("reading graph node %d neighbor %d: %w", i, j, err)
			}
		}
		graphMap[id] = neighbors
	}
	d.graphStore.ReplaceAll(graphMap)

	// Deleted set
	if v32, err = readUint32(); err != nil {
		return fmt.Errorf("reading deleted size: %w", err)
	}
	deletedSize := int(v32)

	// Sanity check
	if deletedSize < 0 || deletedSize > maxGraphSize {
		return fmt.Errorf("invalid deleted size: %d", deletedSize)
	}

	d.deleted = make(map[uint64]bool, deletedSize)
	for i := 0; i < deletedSize; i++ {
		id, err := readUint64()
		if err != nil {
			return fmt.Errorf("reading deleted id %d: %w", i, err)
		}
		d.deleted[id] = true
	}

	// Eagerly rebuild unquantizedOffsetIndex by scanning the mmap file once.
	// This eliminates O(n_vectors) linear scan fallback on cache miss.
	if d.mmapData != nil && d.mmapOffset > 0 && d.quantizer == nil {
		recordSize := int64(8 + d.dim*4)
		scanOffset := int64(0)
		for scanOffset+recordSize <= d.mmapOffset {
			storedID := binary.LittleEndian.Uint64(d.mmapData[scanOffset:])
			d.unquantizedOffsetIndex[storedID] = scanOffset
			scanOffset += recordSize
		}
	} else if d.mmapData != nil && d.mmapOffset > 0 && d.quantizer != nil {
		// Rebuild diskOffsetIndex for quantized data
		scanOffset := int64(0)
		for scanOffset+12 <= d.mmapOffset { // 8 (id) + 4 (length) minimum
			storedID := binary.LittleEndian.Uint64(d.mmapData[scanOffset:])
			d.diskOffsetIndex[storedID] = scanOffset
			dataLen := int64(binary.LittleEndian.Uint32(d.mmapData[scanOffset+8:]))
			scanOffset += 8 + 4 + dataLen
		}
	}

	return nil
}

// CompactSimple is a simple wrapper around the full Compact method
// that uses a background context and ignores the stats.
// Use the full Compact(ctx context.Context) method in diskann_compaction.go
// for more control and statistics.
func (d *DiskANNIndex) CompactSimple() error {
	_, err := d.Compact(context.Background())
	return err
}

// CompactStats returns statistics about potential space savings from compaction
func (d *DiskANNIndex) CompactStats() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	deletedCount := len(d.deleted)
	activeCount := d.count - deletedCount

	recordSize := int64(8 + d.dim*4)
	if d.quantizer != nil {
		recordSize = int64(8 + 4 + d.quantizer.BytesPerVector())
	}

	currentSize := d.mmapOffset
	estimatedCompactedSize := int64(activeCount) * recordSize
	savingsBytes := currentSize - estimatedCompactedSize
	savingsPercent := float64(0)
	if currentSize > 0 {
		savingsPercent = float64(savingsBytes) / float64(currentSize) * 100
	}

	return map[string]interface{}{
		"deleted_count":             deletedCount,
		"active_count":              activeCount,
		"current_size_bytes":        currentSize,
		"estimated_compact_bytes":   estimatedCompactedSize,
		"potential_savings_bytes":   savingsBytes,
		"potential_savings_percent": savingsPercent,
		"compaction_recommended":    savingsPercent > 20, // Recommend if >20% savings
	}
}

// Close cleanly unmaps and closes the index file
func (d *DiskANNIndex) Close() error {
	// Optionally compact before close (do this without holding the main lock
	// since Compact() acquires its own lock)
	if d.compactOnClose {
		d.mu.Lock()
		hasDeleted := len(d.deleted) > 0
		d.mu.Unlock()

		if hasDeleted {
			if _, err := d.Compact(context.Background()); err != nil {
				// Log error but continue with close
				fmt.Printf("compact on close failed: %v\n", err)
			}
		}
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.mmapData != nil {
		if err := mmapUnmap(d.mmapData); err != nil {
			return fmt.Errorf("failed to unmap: %w", err)
		}
		d.mmapData = nil
	}

	if d.mmapFile != nil {
		if err := d.mmapFile.Close(); err != nil {
			return fmt.Errorf("failed to close file: %w", err)
		}
		d.mmapFile = nil
	}

	// Close mmap graph store if applicable
	if gs, ok := d.graphStore.(*MmapGraphStore); ok {
		if err := gs.Close(); err != nil {
			return fmt.Errorf("failed to close mmap graph: %w", err)
		}
	}

	return nil
}

// Helper methods

func (d *DiskANNIndex) distance(a, b []float32) float32 {
	if d.metric == "euclidean" {
		return euclideanDistance(a, b)
	}
	return cosineDistance(a, b)
}

func (d *DiskANNIndex) findFarthest(vec []float32, ids []uint64) int {
	maxDist := float32(-1)
	maxIdx := 0

	for i, id := range ids {
		neighborVec, err := d.getVector(id)
		if err != nil {
			continue
		}
		dist := d.distance(vec, neighborVec)
		if dist > maxDist {
			maxDist = dist
			maxIdx = i
		}
	}

	return maxIdx
}
