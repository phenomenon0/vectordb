package index

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/coder/hnsw"
	"github.com/phenomenon0/vectordb/internal/filter"
	"github.com/phenomenon0/vectordb/internal/index/simd"
)

// simdCosineDistance wraps simd.CosineDistanceF32 as an hnsw.DistanceFunc.
// Uses AVX2+FMA assembly on amd64 for single-pass dot+norms computation,
// replacing the default vek32.CosineSimilarity which does separate passes.
var simdCosineDistance hnsw.DistanceFunc = simd.CosineDistanceF32

// simdNormalizedCosineDistance uses 1-dot(a,b) on pre-normalized vectors.
// 42% faster than full cosine (skips norm computation): 23ns vs 50ns at 768d.
var simdNormalizedCosineDistance hnsw.DistanceFunc = simd.NormalizedCosineDistanceF32

// HNSWIndex wraps the github.com/coder/hnsw implementation
// to conform to the Index interface.
//
// HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor
// search algorithm that provides:
// - O(log n) search complexity
// - High recall (>95% with proper parameters)
// - Incremental construction (no retraining needed)
//
// Thread-safety: Safe for concurrent reads, writes are serialized by mutex.
type HNSWIndex struct {
	mu    sync.RWMutex
	graph *hnsw.Graph[uint64]
	dim   int
	count int

	// Mapping from external ID to internal index
	idToIdx map[uint64]int
	// Store vectors for distance calculation and export
	vectors map[uint64][]float32

	// Tombstone tracking
	deleted map[uint64]bool

	// Metadata storage (for filtered search)
	metadata map[uint64]map[string]interface{}

	// Payload index for O(1) equality / O(log n) range lookups on metadata
	payloadIdx *PayloadIndex

	// Configuration
	m              int     // Connections per node (default: 16)
	ml             float64 // Level multiplier (default: 0.25)
	efSearch       int     // Search beam width (default: 64)
	efConstruction int     // Construction beam width (default: 200)
	prenormalized  bool    // If true, vectors are L2-normalized and cosine uses 1-dot(a,b)

	// Optional quantization (for vector storage, graph remains full precision)
	quantizer     Quantizer         // Quantizer for compressing stored vectors
	quantizedData map[uint64][]byte // ID -> quantized vector storage
}

// NewHNSWIndex creates a new HNSW index.
//
// Configuration parameters (via config map):
//   - m: Connections per node (default: 16, recommended: 5-48)
//   - ml: Level multiplier (default: 0.25)
//   - ef_construction: Build quality (default: 200, recommended: 100-400)
//   - ef_search: Search beam width (default: 64, higher = better recall)
//   - prenormalize: Store unit-norm vectors, use 1-dot(a,b) for cosine (default: true)
//
// Example:
//
//	idx, err := NewHNSWIndex(384, map[string]interface{}{
//	    "m": 16,
//	    "ef_search": 128,
//	})
func NewHNSWIndex(dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dim)
	}

	// Extract configuration
	m := GetConfigInt(config, "m", 16)
	ml := GetConfigFloat(config, "ml", 0.25)
	efSearch := GetConfigInt(config, "ef_search", 200)
	efConstruction := GetConfigInt(config, "ef_construction", 200)

	// Pre-normalization: store unit-norm vectors and use 1-dot(a,b) instead of
	// full cosine distance. 42% faster at 768d (23ns vs 50ns) because norm
	// computation is eliminated. Default: true for cosine similarity.
	prenormalize := GetConfigBool(config, "prenormalize", true)

	// Validate configuration
	if m < 2 || m > 100 {
		return nil, fmt.Errorf("m must be in [2, 100], got %d", m)
	}
	if efSearch < 1 {
		return nil, fmt.Errorf("ef_search must be positive, got %d", efSearch)
	}

	// Create HNSW graph
	g := hnsw.NewGraph[uint64]()
	if prenormalize {
		g.Distance = simdNormalizedCosineDistance
	} else {
		g.Distance = simdCosineDistance
	}
	g.M = m
	g.Ml = ml
	// Use ef_construction during graph building for higher quality
	g.EfSearch = efConstruction

	hnswIdx := &HNSWIndex{
		graph:          g,
		dim:            dim,
		idToIdx:        make(map[uint64]int),
		vectors:        make(map[uint64][]float32),
		deleted:        make(map[uint64]bool),
		metadata:       make(map[uint64]map[string]interface{}),
		payloadIdx:     NewPayloadIndex(),
		m:              m,
		ml:             ml,
		efSearch:       efSearch,
		efConstruction: efConstruction,
		prenormalized:  prenormalize,
	}

	// Check for quantization config
	if quantConfig, ok := config["quantization"].(map[string]interface{}); ok {
		quantType := GetConfigString(quantConfig, "type", "")

		switch quantType {
		case "float16":
			hnswIdx.quantizer = NewFloat16Quantizer(dim)
			hnswIdx.quantizedData = make(map[uint64][]byte)
		case "uint8":
			hnswIdx.quantizer = NewUint8Quantizer(dim)
			hnswIdx.quantizedData = make(map[uint64][]byte)
		case "pq":
			m := GetConfigInt(quantConfig, "m", 8)
			ksub := GetConfigInt(quantConfig, "ksub", 256)
			pq, err := NewProductQuantizer(dim, m, ksub)
			if err != nil {
				return nil, fmt.Errorf("failed to create product quantizer: %w", err)
			}
			hnswIdx.quantizer = pq
			hnswIdx.quantizedData = make(map[uint64][]byte)
		}
	}

	return hnswIdx, nil
}

// Name returns "HNSW"
func (h *HNSWIndex) Name() string {
	return "HNSW"
}

// distanceFunc returns the appropriate distance function for this index.
func (h *HNSWIndex) distanceFunc() hnsw.DistanceFunc {
	if h.prenormalized {
		return simdNormalizedCosineDistance
	}
	return simdCosineDistance
}

// Add inserts a vector into the HNSW index.
//
// Thread-safety: Writes are serialized
func (h *HNSWIndex) Add(ctx context.Context, id uint64, vector []float32) error {
	if len(vector) != h.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", h.dim, len(vector))
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if already exists and not deleted (tombstoned)
	if _, exists := h.idToIdx[id]; exists {
		if !h.deleted[id] {
			return fmt.Errorf("vector with ID %d already exists", id)
		}
		// Vector was tombstoned - allow resurrection with new data
		// Remove tombstone marker
		delete(h.deleted, id)

		// Make a copy to avoid external mutations
		vecCopy := make([]float32, h.dim)
		copy(vecCopy, vector)

		if h.prenormalized {
			simd.NormalizeF32(vecCopy)
		}

		// Update the vector in place in the graph (more efficient than Delete+Add)
		h.graph.Update(id, vecCopy)

		if err := h.storeVectorLocked(id, vecCopy); err != nil {
			return err
		}

		return nil
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Make a copy to avoid external mutations
	vecCopy := make([]float32, h.dim)
	copy(vecCopy, vector)

	if h.prenormalized {
		simd.NormalizeF32(vecCopy)
	}

	// Add to HNSW graph (graph always uses full precision for accuracy)
	h.graph.Add(hnsw.MakeNode(id, vecCopy))

	if err := h.storeVectorLocked(id, vecCopy); err != nil {
		return err
	}

	// Update mappings
	h.idToIdx[id] = h.count
	h.count++

	// NOTE: Lock yielding removed - it caused race conditions.
	// For bulk insert performance, use explicit batching with AddBatch()
	// or construct the index in single-threaded build mode.

	return nil
}

// SetMetadata sets or updates the metadata for a vector.
// This is used for filtered search. Metadata can be set after vector insertion.
func (h *HNSWIndex) SetMetadata(id uint64, metadata map[string]interface{}) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if vector exists
	if _, exists := h.idToIdx[id]; !exists {
		return fmt.Errorf("vector with ID %d does not exist", id)
	}

	// Update payload index (remove old entry first if exists)
	if old, ok := h.metadata[id]; ok {
		h.payloadIdx.Remove(id, old)
	}

	// Store metadata (make a copy to avoid external mutations)
	if metadata != nil {
		metaCopy := make(map[string]interface{}, len(metadata))
		for k, v := range metadata {
			metaCopy[k] = v
		}
		h.metadata[id] = metaCopy
		h.payloadIdx.Index(id, metaCopy)
	} else {
		// Allow nil to clear metadata
		delete(h.metadata, id)
	}

	return nil
}

// BatchAdd inserts multiple vectors efficiently with parallel graph insertion.
// This is significantly faster than calling Add in a loop for large batches.
//
// Graph insertion is parallelized across GOMAXPROCS workers (capped at 8)
// using AddConcurrent for fine-grained locking within the graph.
func (h *HNSWIndex) BatchAdd(ctx context.Context, vectors map[uint64][]float32) error {
	// Pre-validation
	for id, vec := range vectors {
		if len(vec) != h.dim {
			return fmt.Errorf("vector %d dimension mismatch: expected %d, got %d", id, h.dim, len(vec))
		}
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// 1. Sequential: validate duplicates, handle resurrections, collect new nodes.
	newNodes := make([]hnsw.Node[uint64], 0, len(vectors))
	for id, vec := range vectors {
		if _, exists := h.idToIdx[id]; exists {
			if !h.deleted[id] {
				return fmt.Errorf("vector with ID %d already exists", id)
			}
			// Tombstoned — resurrect via Update
			delete(h.deleted, id)
			vecCopy := make([]float32, h.dim)
			copy(vecCopy, vec)
			if h.prenormalized {
				simd.NormalizeF32(vecCopy)
			}
			h.graph.Update(id, vecCopy)
			if err := h.storeVectorLocked(id, vecCopy); err != nil {
				return fmt.Errorf("failed to store resurrected vector %d: %w", id, err)
			}
			continue
		}

		vecCopy := make([]float32, h.dim)
		copy(vecCopy, vec)
		if h.prenormalized {
			simd.NormalizeF32(vecCopy)
		}
		newNodes = append(newNodes, hnsw.MakeNode(id, vecCopy))
	}

	if len(newNodes) == 0 {
		return h.maybeTrainQuantizerLocked()
	}

	// 2. Sequential: pre-register metadata (idToIdx, vectors) so lookups work.
	for _, node := range newNodes {
		if err := h.storeVectorLocked(node.Key, node.Value); err != nil {
			return fmt.Errorf("failed to store vector %d: %w", node.Key, err)
		}
		h.idToIdx[node.Key] = h.count
		h.count++
	}

	// 3. Parallel graph insertion via AddConcurrent.
	if err := h.parallelGraphInsert(ctx, newNodes); err != nil {
		return err
	}

	return h.maybeTrainQuantizerLocked()
}

// BatchAddNoCopy inserts multiple vectors without copying them.
// The caller must guarantee ownership transfer — the slices must not be
// modified after this call. Used by binary import where vectors are freshly decoded.
//
// Graph insertion is parallelized across GOMAXPROCS workers (capped at 8)
// using AddConcurrent for fine-grained locking within the graph.
func (h *HNSWIndex) BatchAddNoCopy(ctx context.Context, vectors map[uint64][]float32) error {
	for id, vec := range vectors {
		if len(vec) != h.dim {
			return fmt.Errorf("vector %d dimension mismatch: expected %d, got %d", id, h.dim, len(vec))
		}
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// 1. Sequential: validate duplicates, handle resurrections, collect new IDs.
	newNodes := make([]hnsw.Node[uint64], 0, len(vectors))
	for id, vec := range vectors {
		if h.prenormalized {
			simd.NormalizeF32(vec)
		}
		if _, exists := h.idToIdx[id]; exists {
			if !h.deleted[id] {
				return fmt.Errorf("vector with ID %d already exists", id)
			}
			// Tombstoned — resurrect via Update
			delete(h.deleted, id)
			h.graph.Update(id, vec)
			if err := h.storeVectorLocked(id, vec); err != nil {
				return fmt.Errorf("failed to store resurrected vector %d: %w", id, err)
			}
			continue
		}
		newNodes = append(newNodes, hnsw.MakeNode(id, vec))
	}

	if len(newNodes) == 0 {
		return h.maybeTrainQuantizerLocked()
	}

	// 2. Sequential: pre-register metadata (idToIdx, vectors) so lookups work.
	for _, node := range newNodes {
		if err := h.storeVectorLocked(node.Key, node.Value); err != nil {
			return fmt.Errorf("failed to store vector %d: %w", node.Key, err)
		}
		h.idToIdx[node.Key] = h.count
		h.count++
	}

	// 3. Parallel graph insertion via AddConcurrent.
	if err := h.parallelGraphInsert(ctx, newNodes); err != nil {
		return err
	}

	return h.maybeTrainQuantizerLocked()
}

// parallelGraphInsert fans out graph insertion across multiple goroutines.
// Must be called with h.mu held. Nodes must already be registered in idToIdx/vectors.
func (h *HNSWIndex) parallelGraphInsert(ctx context.Context, nodes []hnsw.Node[uint64]) error {
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers > 8 {
		numWorkers = 8
	}
	if numWorkers > len(nodes) {
		numWorkers = len(nodes)
	}

	// For small batches, insert sequentially to avoid goroutine overhead.
	if len(nodes) < 100 || numWorkers <= 1 {
		h.graph.Add(nodes...)
		return nil
	}

	ch := make(chan hnsw.Node[uint64], len(nodes))
	for _, n := range nodes {
		ch <- n
	}
	close(ch)

	var wg sync.WaitGroup
	errCh := make(chan error, numWorkers)

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(rand.Intn(1<<30))))
			for node := range ch {
				if ctx.Err() != nil {
					errCh <- ctx.Err()
					return
				}
				g := h.graph
				g.AddConcurrent(node, rng)
			}
		}()
	}

	wg.Wait()
	close(errCh)

	// Return first error if any.
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return nil
}

// Search finds k nearest neighbors using HNSW.
//
// When a filter is provided, uses in-graph filtered search (SOTA approach)
// which filters DURING graph traversal rather than after. This provides
// 5-20x speedup for selective filters.
//
// Thread-safety: Safe for concurrent reads
func (h *HNSWIndex) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	if len(query) != h.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", h.dim, len(query))
	}

	if k <= 0 {
		return nil, fmt.Errorf("k must be positive, got %d", k)
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	// Check context cancellation
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	// Normalize query if using pre-normalized vectors
	searchQuery := query
	if h.prenormalized {
		searchQuery = make([]float32, len(query))
		copy(searchQuery, query)
		simd.NormalizeF32(searchQuery)
	}

	// Extract ef_search parameter and filter if provided
	efSearch := h.efSearch
	var f filter.Filter
	switch p := params.(type) {
	case HNSWSearchParams:
		if p.EfSearch > 0 {
			efSearch = p.EfSearch
		}
		f = p.Filter
	case *HNSWSearchParams:
		if p != nil {
			if p.EfSearch > 0 {
				efSearch = p.EfSearch
			}
			f = p.Filter
		}
	}

	var nodes []hnsw.Node[uint64]

	if f != nil {
		// FILTERED SEARCH: Use in-graph filtering with adaptive over-fetch
		// Filter is applied DURING graph traversal for 5-20x speedup

		// Try payload index for O(1) bitmap lookup
		filterFn := func(id uint64) bool {
			if h.deleted[id] {
				return false
			}
			meta := h.metadata[id]
			if meta == nil {
				return false
			}
			return f.Evaluate(meta)
		}

		if candidates, ok := h.payloadIdx.QueryBitmap(f); ok {
			selectivity := float64(len(candidates)) / float64(h.count)
			if selectivity < 0.15 {
				filterFn = func(id uint64) bool {
					if h.deleted[id] {
						return false
					}
					_, match := candidates[id]
					return match
				}
			}
		}

		// Adaptive over-fetch: if filter is selective, we may not get enough results
		// Start with 2x, increase to 4x, 8x if still not enough
		multiplier := 2
		for attempt := 0; attempt < 3; attempt++ {
			fetchK := k * multiplier
			if fetchK > h.count {
				fetchK = h.count
			}
			nodes = h.graph.SearchWithEf(searchQuery, fetchK, efSearch, filterFn)
			if len(nodes) >= k {
				break
			}
			multiplier *= 2
		}
	} else {
		// UNFILTERED SEARCH: Standard HNSW search
		// Account for deletions by fetching extra candidates
		fetchK := k
		if h.count > 0 {
			deletionRatio := float64(len(h.deleted)) / float64(h.count)
			if deletionRatio > 0.1 {
				fetchK = int(float64(k) / (1.0 - deletionRatio))
				if fetchK > h.count {
					fetchK = h.count
				}
				if fetchK < k {
					fetchK = k // Never fetch less than requested
				}
			}
		}
		nodes = h.graph.SearchWithEf(searchQuery, fetchK, efSearch, nil)
	}

	// Collect results (filter already applied for filtered search)
	candidateVectors := make([][]float32, 0, len(nodes))
	candidateIDs := make([]uint64, 0, len(nodes))
	candidateMetadata := make([]map[string]interface{}, 0, len(nodes))

	for _, node := range nodes {
		// For unfiltered search, still need to skip deleted vectors
		if f == nil && h.deleted[node.Key] {
			continue
		}

		candidateVectors = append(candidateVectors, node.Value)
		candidateIDs = append(candidateIDs, node.Key)
		candidateMetadata = append(candidateMetadata, h.metadata[node.Key])
	}

	if len(candidateVectors) == 0 {
		return []Result{}, nil
	}

	// Compute distances (use GPU if beneficial)
	var distances []float32
	var err error

	if ShouldUseGPU(1, len(candidateVectors)) {
		distances, err = GPUSingleDistance(query, candidateVectors, "cosine")
		if err != nil {
			// Fall back to CPU
			distances = h.computeDistancesCPU(searchQuery, candidateVectors)
		}
	} else {
		// CPU computation for small candidate sets
		distances = h.computeDistancesCPU(searchQuery, candidateVectors)
	}

	// Build results
	results := make([]Result, len(distances))
	for i, dist := range distances {
		results[i] = Result{
			ID:       candidateIDs[i],
			Distance: dist,
			Score:    1.0 / (1.0 + dist),
			Metadata: candidateMetadata[i],
		}
	}

	// Limit to k results
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// computeDistancesCPU computes distances to candidate vectors on CPU
func (h *HNSWIndex) computeDistancesCPU(query []float32, candidates [][]float32) []float32 {
	distances := make([]float32, len(candidates))
	for i, vec := range candidates {
		distances[i] = h.graph.Distance(query, vec)
	}
	return distances
}

// Delete marks a vector as deleted (tombstone).
//
// Thread-safety: Writes are serialized
func (h *HNSWIndex) Delete(ctx context.Context, id uint64) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if exists
	if _, exists := h.idToIdx[id]; !exists {
		return fmt.Errorf("vector with ID %d not found", id)
	}

	// Check context cancellation
	if ctx.Err() != nil {
		return ctx.Err()
	}

	// Mark as deleted (tombstone)
	h.deleted[id] = true

	return nil
}

// Stats returns index statistics
func (h *HNSWIndex) Stats() IndexStats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	deletedCount := len(h.deleted)
	activeCount := h.count - deletedCount

	// Estimate memory usage
	// - Graph nodes: ~(M * 2 * 8 bytes per connection) * count
	//   (Note: graph stores full precision vectors regardless of quantization)
	// - Vectors: stored data (quantized or unquantized)
	// - Mappings: count * (8 + 8) bytes
	graphMem := int64(h.m * 2 * 8 * h.count)

	vectorMem := int64(len(h.vectors) * (8 + h.dim*4))
	for _, data := range h.quantizedData {
		vectorMem += int64(8 + len(data))
	}

	mappingMem := int64(h.count * 16)
	totalMem := graphMem + vectorMem + mappingMem

	extra := map[string]interface{}{
		"m":               h.m,
		"ml":              h.ml,
		"ef_search":       h.efSearch,
		"ef_construction": h.efConstruction,
		"prenormalized":   h.prenormalized,
	}

	// Add quantization info if enabled
	if h.quantizer != nil {
		extra["quantization"] = h.quantizer.Type()
		extra["quantized_bytes_per_vector"] = h.quantizer.BytesPerVector()

		// Calculate compression ratio (for stored vectors only, not graph)
		originalSize := h.dim * 4
		compressedSize := h.quantizer.BytesPerVector()
		extra["compression_ratio"] = float64(originalSize) / float64(compressedSize)

		// Note: graph still uses full precision internally
		extra["note"] = "graph uses full precision, only stored vectors are quantized"
	}

	return IndexStats{
		Name:       "HNSW",
		Dim:        h.dim,
		Count:      h.count,
		Deleted:    deletedCount,
		Active:     activeCount,
		MemoryUsed: totalMem,
		DiskUsed:   0, // In-memory index
		Extra:      extra,
	}
}

// Export serializes the HNSW index to JSON.
//
// Format (JSON):
//
//	{
//	  "version": 2,
//	  "dim": 384,
//	  "config": {"m": 16, "ml": 0.25, "ef_search": 64, "prenormalize": true},
//	  "vectors": [[id, [v1, v2, ...]], ...],
//	  "deleted": [id1, id2, ...]
//	}
//
// Thread-safety: Safe for concurrent reads (snapshot semantics)
func (h *HNSWIndex) Export() ([]byte, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	// Define export types
	type vectorEntry struct {
		ID        uint64                 `json:"id"`
		Vector    []float32              `json:"vector,omitempty"`
		Quantized []byte                 `json:"quantized,omitempty"`
		Metadata  map[string]interface{} `json:"metadata,omitempty"`
		Deleted   bool                   `json:"deleted,omitempty"`
	}

	type exportFormat struct {
		Version   int                    `json:"version"`
		Dim       int                    `json:"dim"`
		Config    map[string]interface{} `json:"config"`
		Quantizer *quantizerState        `json:"quantizer,omitempty"`
		Vectors   []vectorEntry          `json:"vectors"`
		Deleted   []uint64               `json:"deleted,omitempty"`
	}

	quantizerState, err := exportQuantizerState(h.quantizer)
	if err != nil {
		return nil, err
	}

	ids := h.sortedIDsLocked()
	vectors := make([]vectorEntry, 0, len(ids))
	for _, id := range ids {
		entry := vectorEntry{
			ID:      id,
			Deleted: h.deleted[id],
		}
		if vec, ok := h.vectors[id]; ok {
			entry.Vector = append([]float32(nil), vec...)
		}
		if quantized, ok := h.quantizedData[id]; ok {
			entry.Quantized = append([]byte(nil), quantized...)
		}
		if metadata, ok := h.metadata[id]; ok {
			entry.Metadata = copyMetadataMap(metadata)
		}
		vectors = append(vectors, entry)
	}

	// Collect deleted IDs
	deleted := make([]uint64, 0, len(h.deleted))
	for id := range h.deleted {
		deleted = append(deleted, id)
	}

	data := exportFormat{
		Version: 2,
		Dim:     h.dim,
		Config: map[string]interface{}{
			"m":               h.m,
			"ml":              h.ml,
			"ef_search":       h.efSearch,
			"ef_construction": h.efConstruction,
			"prenormalize":    h.prenormalized,
		},
		Quantizer: quantizerState,
		Vectors:   vectors,
		Deleted:   deleted,
	}

	return json.Marshal(data)
}

// Import loads a previously exported HNSW index.
//
// Thread-safety: NOT safe for concurrent access (replaces entire index)
func (h *HNSWIndex) Import(data []byte) error {
	type vectorEntry struct {
		ID        uint64                 `json:"id"`
		Vector    []float32              `json:"vector,omitempty"`
		Quantized []byte                 `json:"quantized,omitempty"`
		Metadata  map[string]interface{} `json:"metadata,omitempty"`
		Deleted   bool                   `json:"deleted,omitempty"`
	}

	type importFormat struct {
		Version   int                    `json:"version"`
		Dim       int                    `json:"dim"`
		Config    map[string]interface{} `json:"config"`
		Quantizer *quantizerState        `json:"quantizer,omitempty"`
		Vectors   []vectorEntry          `json:"vectors"`
		Deleted   []uint64               `json:"deleted,omitempty"`
	}

	var imp importFormat
	if err := json.Unmarshal(data, &imp); err != nil {
		return fmt.Errorf("failed to unmarshal HNSW index: %w", err)
	}

	if imp.Version != 1 && imp.Version != 2 {
		return fmt.Errorf("unsupported HNSW export version: %d", imp.Version)
	}

	if imp.Dim != h.dim {
		return fmt.Errorf("dimension mismatch: index is %d, import data is %d", h.dim, imp.Dim)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	quantizer, err := importQuantizerState(imp.Dim, imp.Quantizer)
	if err != nil {
		return err
	}

	// Clear existing data and restore config from imported state.
	// FIX #4: Copy imported config to both graph AND receiver fields.
	// Previously only graph fields were set; h.m, h.ml, h.efSearch, h.efConstruction
	// remained stale, causing Stats(), Compact(), and search to use wrong values.
	importedM := GetConfigInt(imp.Config, "m", 16)
	importedMl := GetConfigFloat(imp.Config, "ml", 0.25)
	importedEfConstruction := GetConfigInt(imp.Config, "ef_construction", 200)
	importedEfSearch := GetConfigInt(imp.Config, "ef_search", 64)
	importedPrenormalize := GetConfigBool(imp.Config, "prenormalize", false)

	h.m = importedM
	h.ml = importedMl
	h.efSearch = importedEfSearch
	h.efConstruction = importedEfConstruction
	h.prenormalized = importedPrenormalize

	h.graph = hnsw.NewGraph[uint64]()
	h.graph.Distance = h.distanceFunc()
	h.graph.M = importedM
	h.graph.Ml = importedMl
	h.graph.EfSearch = importedEfConstruction

	h.idToIdx = make(map[uint64]int)
	h.vectors = make(map[uint64][]float32)
	h.quantizedData = make(map[uint64][]byte)
	h.deleted = make(map[uint64]bool)
	h.metadata = make(map[uint64]map[string]interface{})
	h.payloadIdx = NewPayloadIndex()
	h.quantizer = quantizer
	h.count = 0

	// Rebuild index — vectors in export are already normalized if prenormalized was true
	for _, entry := range imp.Vectors {
		rawVector := entry.Vector
		if len(rawVector) == 0 && len(entry.Quantized) > 0 {
			if h.quantizer == nil {
				return fmt.Errorf("quantized vector present for ID %d without quantizer state", entry.ID)
			}
			var err error
			rawVector, err = h.quantizer.Dequantize(entry.Quantized)
			if err != nil {
				return fmt.Errorf("failed to dequantize vector %d: %w", entry.ID, err)
			}
		}
		if len(rawVector) == 0 {
			return fmt.Errorf("missing vector payload for ID %d", entry.ID)
		}

		vectorCopy := append([]float32(nil), rawVector...)
		h.graph.Add(hnsw.MakeNode(entry.ID, vectorCopy))

		if len(entry.Quantized) > 0 {
			h.quantizedData[entry.ID] = append([]byte(nil), entry.Quantized...)
		} else {
			h.vectors[entry.ID] = vectorCopy
		}
		if entry.Metadata != nil {
			meta := copyMetadataMap(entry.Metadata)
			h.metadata[entry.ID] = meta
			h.payloadIdx.Index(entry.ID, meta)
		}

		// Update mappings
		h.idToIdx[entry.ID] = h.count
		h.count++

		if imp.Version == 2 && entry.Deleted {
			h.deleted[entry.ID] = true
		}
	}

	if imp.Version == 1 {
		for _, id := range imp.Deleted {
			h.deleted[id] = true
		}
	}

	return nil
}

// Compact rebuilds the HNSW graph excluding deleted vectors.
// This is useful after many deletions to reclaim memory and improve performance.
//
// Returns the number of vectors removed.
func (h *HNSWIndex) Compact() (int, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	deletedCount := len(h.deleted)
	if deletedCount == 0 {
		return 0, nil // Nothing to compact
	}

	// Create new graph
	newGraph := hnsw.NewGraph[uint64]()
	newGraph.Distance = h.distanceFunc()
	newGraph.M = h.m
	newGraph.Ml = h.ml
	newGraph.EfSearch = h.efConstruction

	newIdToIdx := make(map[uint64]int)
	newVectors := make(map[uint64][]float32)
	newQuantized := make(map[uint64][]byte)
	newMetadata := make(map[uint64]map[string]interface{})
	newPayloadIdx := NewPayloadIndex()
	newCount := 0

	for _, id := range h.sortedIDsLocked() {
		if h.deleted[id] {
			continue // Skip deleted
		}

		vec, err := h.loadVectorLocked(id)
		if err != nil {
			return 0, err
		}
		newGraph.Add(hnsw.MakeNode(id, vec))
		if raw, ok := h.vectors[id]; ok {
			newVectors[id] = append([]float32(nil), raw...)
		}
		if quantized, ok := h.quantizedData[id]; ok {
			newQuantized[id] = append([]byte(nil), quantized...)
		}
		if metadata, ok := h.metadata[id]; ok {
			meta := copyMetadataMap(metadata)
			newMetadata[id] = meta
			newPayloadIdx.Index(id, meta)
		}
		newIdToIdx[id] = newCount
		newCount++
	}

	// Replace old data
	h.graph = newGraph
	h.idToIdx = newIdToIdx
	h.vectors = newVectors
	h.quantizedData = newQuantized
	h.metadata = newMetadata
	h.payloadIdx = newPayloadIdx
	h.deleted = make(map[uint64]bool)
	h.count = newCount

	return deletedCount, nil
}

func (h *HNSWIndex) storeVectorLocked(id uint64, vec []float32) error {
	if h.quantizer == nil {
		h.vectors[id] = vec
		return nil
	}

	if quantizerNeedsTraining(h.quantizer) {
		h.vectors[id] = vec
		delete(h.quantizedData, id)
		return h.maybeTrainQuantizerLocked()
	}

	quantized, err := h.quantizer.Quantize(vec)
	if err != nil {
		return fmt.Errorf("failed to quantize vector: %w", err)
	}
	if h.quantizedData == nil {
		h.quantizedData = make(map[uint64][]byte)
	}
	h.quantizedData[id] = quantized
	delete(h.vectors, id)
	return nil
}

func (h *HNSWIndex) maybeTrainQuantizerLocked() error {
	if h.quantizer == nil || !quantizerNeedsTraining(h.quantizer) {
		return nil
	}
	if len(h.vectors) < quantizerMinTrainingVectors(h.quantizer) {
		return nil
	}

	if err := trainQuantizerWithDefaults(h.quantizer, h.flattenVectorsLocked()); err != nil {
		return fmt.Errorf("failed to train quantizer: %w", err)
	}
	if h.quantizedData == nil {
		h.quantizedData = make(map[uint64][]byte, len(h.vectors))
	}
	for _, id := range h.sortedVectorIDsLocked() {
		quantized, err := h.quantizer.Quantize(h.vectors[id])
		if err != nil {
			return fmt.Errorf("failed to quantize vector %d: %w", id, err)
		}
		h.quantizedData[id] = quantized
		delete(h.vectors, id)
	}

	return nil
}

func (h *HNSWIndex) flattenVectorsLocked() []float32 {
	ids := h.sortedVectorIDsLocked()
	flattened := make([]float32, 0, len(ids)*h.dim)
	for _, id := range ids {
		flattened = append(flattened, h.vectors[id]...)
	}
	return flattened
}

func (h *HNSWIndex) sortedVectorIDsLocked() []uint64 {
	ids := make([]uint64, 0, len(h.vectors))
	for id := range h.vectors {
		ids = append(ids, id)
	}
	return sortUint64s(ids)
}

func (h *HNSWIndex) sortedIDsLocked() []uint64 {
	ids := make([]uint64, 0, len(h.idToIdx))
	for id := range h.idToIdx {
		ids = append(ids, id)
	}
	return sortUint64s(ids)
}

func (h *HNSWIndex) loadVectorLocked(id uint64) ([]float32, error) {
	if vec, ok := h.vectors[id]; ok {
		return append([]float32(nil), vec...), nil
	}
	if quantized, ok := h.quantizedData[id]; ok {
		if h.quantizer == nil {
			return nil, fmt.Errorf("vector %d is quantized but quantizer is missing", id)
		}
		vec, err := h.quantizer.Dequantize(quantized)
		if err != nil {
			return nil, fmt.Errorf("failed to dequantize vector %d: %w", id, err)
		}
		return vec, nil
	}
	return nil, fmt.Errorf("vector %d not found", id)
}

func copyMetadataMap(src map[string]interface{}) map[string]interface{} {
	if src == nil {
		return nil
	}
	dst := make(map[string]interface{}, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func sortUint64s(ids []uint64) []uint64 {
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return ids
}

// init registers HNSW index type with the global factory and
// registers our SIMD cosine distance for graph serialization.
func init() {
	Register("hnsw", func(dim int, config map[string]interface{}) (Index, error) {
		return NewHNSWIndex(dim, config)
	})
	// Override the "cosine" distance function so encode/decode round-trips
	// correctly with our SIMD implementation.
	hnsw.RegisterDistanceFunc("cosine", simdCosineDistance)
	// Register pre-normalized cosine for graphs that use prenormalization.
	hnsw.RegisterDistanceFunc("cosine_prenorm", simdNormalizedCosineDistance)
}
