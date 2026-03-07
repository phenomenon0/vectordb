package index

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/phenomenon0/vectordb/internal/filter"
)

// IVFIndex implements Inverted File index for large-scale vector search
// Uses clustering to partition vectors into buckets for efficient search
type IVFIndex struct {
	mu sync.RWMutex

	dim       int                        // Vector dimension
	nlist     int                        // Number of clusters (centroids)
	nprobe    int                        // Number of clusters to search
	centroids [][]float32                // Cluster centers [nlist][dim]
	postings  map[int][]uint64           // Inverted lists: cluster_id -> vector IDs
	vectors   map[uint64][]float32       // ID -> vector storage (unquantized)
	deleted   map[uint64]bool            // Tombstone deletions
	clusterID map[uint64]int             // ID -> assigned cluster
	count     int                        // Total vectors added

	// Metadata storage (for filtered search)
	metadata map[uint64]map[string]interface{}

	metric string // "cosine" or "euclidean"

	// Statistics
	clusterSizes []int // Size of each cluster

	// Optional quantization
	quantizer     Quantizer          // Quantizer for compression
	quantizedData map[uint64][]byte  // ID -> quantized vector storage
}

// NewIVFIndex creates a new IVF index
func NewIVFIndex(dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dim)
	}

	// Parse configuration
	nlist := GetConfigInt(config, "nlist", 100)     // Default: 100 clusters
	nprobe := GetConfigInt(config, "nprobe", 10)    // Default: search 10 clusters
	metric := GetConfigString(config, "metric", "cosine")

	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive, got %d", nlist)
	}

	if nprobe <= 0 || nprobe > nlist {
		return nil, fmt.Errorf("nprobe must be in range [1, nlist], got %d", nprobe)
	}

	ivf := &IVFIndex{
		dim:          dim,
		nlist:        nlist,
		nprobe:       nprobe,
		centroids:    nil, // Will be computed during training
		postings:     make(map[int][]uint64),
		vectors:      make(map[uint64][]float32),
		deleted:      make(map[uint64]bool),
		clusterID:    make(map[uint64]int),
		metadata:     make(map[uint64]map[string]interface{}),
		metric:       metric,
		clusterSizes: make([]int, nlist),
	}

	// Check for quantization config
	if quantConfig, ok := config["quantization"].(map[string]interface{}); ok {
		quantType := GetConfigString(quantConfig, "type", "")

		switch quantType {
		case "float16":
			ivf.quantizer = NewFloat16Quantizer(dim)
			ivf.quantizedData = make(map[uint64][]byte)
		case "uint8":
			ivf.quantizer = NewUint8Quantizer(dim)
			ivf.quantizedData = make(map[uint64][]byte)
		case "pq":
			m := GetConfigInt(quantConfig, "m", 8)
			ksub := GetConfigInt(quantConfig, "ksub", 256)
			pq, err := NewProductQuantizer(dim, m, ksub)
			if err != nil {
				return nil, fmt.Errorf("failed to create product quantizer: %w", err)
			}
			ivf.quantizer = pq
			ivf.quantizedData = make(map[uint64][]byte)
		}
	}

	return ivf, nil
}

func init() {
	// Register IVF index type
	Register("ivf", NewIVFIndex)
}

// Name returns the index type name
func (ivf *IVFIndex) Name() string {
	return "IVF"
}

// Add adds a vector to the IVF index
func (ivf *IVFIndex) Add(ctx context.Context, id uint64, vector []float32) error {
	if len(vector) != ivf.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", ivf.dim, len(vector))
	}

	ivf.mu.Lock()
	defer ivf.mu.Unlock()

	// Make vector copy for processing
	vecCopy := make([]float32, ivf.dim)
	copy(vecCopy, vector)

	// If centroids not trained yet, collect vectors for training
	if ivf.centroids == nil {
		// Store unquantized for training
		ivf.vectors[id] = vecCopy

		// Auto-train when we have enough vectors (at least 10x nlist)
		if len(ivf.vectors) >= ivf.nlist*10 {
			if err := ivf.trainCentroids(); err != nil {
				return fmt.Errorf("failed to train centroids: %w", err)
			}
			// After training, quantize existing vectors if quantizer is configured
			if ivf.quantizer != nil {
				if err := ivf.quantizeExistingVectors(); err != nil {
					return fmt.Errorf("failed to quantize existing vectors: %w", err)
				}
			}
		}
	} else {
		// Centroids trained - store based on quantization config
		if ivf.quantizer != nil {
			// Quantize and store compressed data
			quantized, err := ivf.quantizer.Quantize(vecCopy)
			if err != nil {
				return fmt.Errorf("failed to quantize vector: %w", err)
			}
			ivf.quantizedData[id] = quantized
		} else {
			// Store unquantized
			ivf.vectors[id] = vecCopy
		}
	}

	// Assign to nearest cluster if centroids exist
	if ivf.centroids != nil {
		clusterIdx := ivf.findNearestCluster(vecCopy)
		ivf.clusterID[id] = clusterIdx
		ivf.postings[clusterIdx] = append(ivf.postings[clusterIdx], id)
		ivf.clusterSizes[clusterIdx]++
	}

	delete(ivf.deleted, id)
	ivf.count++

	return nil
}

// SetMetadata sets or updates the metadata for a vector.
// This is used for filtered search. Metadata can be set after vector insertion.
func (ivf *IVFIndex) SetMetadata(id uint64, metadata map[string]interface{}) error {
	ivf.mu.Lock()
	defer ivf.mu.Unlock()

	// Check if vector exists
	if _, exists := ivf.vectors[id]; !exists {
		if _, existsQuant := ivf.quantizedData[id]; !existsQuant {
			return fmt.Errorf("vector with ID %d does not exist", id)
		}
	}

	// Store metadata (make a copy to avoid external mutations)
	if metadata != nil {
		metaCopy := make(map[string]interface{}, len(metadata))
		for k, v := range metadata {
			metaCopy[k] = v
		}
		ivf.metadata[id] = metaCopy
	} else {
		// Allow nil to clear metadata
		delete(ivf.metadata, id)
	}

	return nil
}

// Search performs vector search using IVF
func (ivf *IVFIndex) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	if len(query) != ivf.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", ivf.dim, len(query))
	}

	ivf.mu.RLock()
	defer ivf.mu.RUnlock()

	// If no centroids trained yet, fall back to brute force
	if ivf.centroids == nil {
		return ivf.bruteForceSearch(query, k)
	}

	// Get nprobe and filter from params if provided
	nprobe := ivf.nprobe
	var filter filter.Filter
	if p, ok := params.(IVFSearchParams); ok {
		if p.NProbe > 0 {
			nprobe = p.NProbe
			if nprobe > ivf.nlist {
				nprobe = ivf.nlist
			}
		}
		filter = p.Filter
	}

	// Find nprobe nearest clusters
	nearestClusters := ivf.findNearestClusters(query, nprobe)

	// Collect candidate vectors from these clusters
	candidates := make(map[uint64]bool)
	for _, clusterIdx := range nearestClusters {
		if ids, ok := ivf.postings[clusterIdx]; ok {
			for _, id := range ids {
				if !ivf.deleted[id] {
					candidates[id] = true
				}
			}
		}
	}

	if len(candidates) == 0 {
		return []Result{}, nil
	}

	// Collect candidate vectors, IDs, and metadata that pass the filter
	candidateVectors := make([][]float32, 0, len(candidates))
	candidateIDs := make([]uint64, 0, len(candidates))
	candidateMetadata := make([]map[string]interface{}, 0, len(candidates))

	if ivf.quantizer != nil {
		// Dequantize vectors for search
		for id := range candidates {
			// Apply metadata filter if provided
			if filter != nil {
				meta := ivf.metadata[id]
				if meta == nil || !filter.Evaluate(meta) {
					continue // Skip vectors that don't match filter
				}
			}

			quantized, ok := ivf.quantizedData[id]
			if !ok {
				continue
			}

			vec, err := ivf.quantizer.Dequantize(quantized)
			if err != nil {
				return nil, fmt.Errorf("failed to dequantize vector %d: %w", id, err)
			}

			candidateVectors = append(candidateVectors, vec)
			candidateIDs = append(candidateIDs, id)
			candidateMetadata = append(candidateMetadata, ivf.metadata[id])
		}
	} else {
		// Use unquantized vectors
		for id := range candidates {
			// Apply metadata filter if provided
			if filter != nil {
				meta := ivf.metadata[id]
				if meta == nil || !filter.Evaluate(meta) {
					continue // Skip vectors that don't match filter
				}
			}

			candidateVectors = append(candidateVectors, ivf.vectors[id])
			candidateIDs = append(candidateIDs, id)
			candidateMetadata = append(candidateMetadata, ivf.metadata[id])
		}
	}

	var distances []float32
	var err error

	// Use GPU if beneficial (many candidates)
	if ShouldUseGPU(1, len(candidateVectors)) {
		distances, err = GPUSingleDistance(query, candidateVectors, ivf.metric)
		if err != nil {
			// Fall back to CPU if GPU fails
			distances = ivf.computeCandidateDistancesCPU(query, candidateVectors)
		}
	} else {
		// CPU computation for small candidate set
		distances = ivf.computeCandidateDistancesCPU(query, candidateVectors)
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

	// Sort by distance (ascending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// Return top-k
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// Delete marks a vector as deleted
func (ivf *IVFIndex) Delete(ctx context.Context, id uint64) error {
	ivf.mu.Lock()
	defer ivf.mu.Unlock()

	if _, exists := ivf.vectors[id]; !exists {
		return fmt.Errorf("vector %d not found", id)
	}

	ivf.deleted[id] = true
	return nil
}

// Stats returns index statistics
func (ivf *IVFIndex) Stats() IndexStats {
	ivf.mu.RLock()
	defer ivf.mu.RUnlock()

	deletedCount := len(ivf.deleted)

	// Calculate cluster balance (std dev of cluster sizes)
	avgSize := float64(ivf.count) / float64(ivf.nlist)
	variance := 0.0
	for _, size := range ivf.clusterSizes {
		diff := float64(size) - avgSize
		variance += diff * diff
	}
	stdDev := math.Sqrt(variance / float64(ivf.nlist))

	extra := map[string]interface{}{
		"nlist":            ivf.nlist,
		"nprobe":           ivf.nprobe,
		"metric":           ivf.metric,
		"centroids_ready":  ivf.centroids != nil,
		"cluster_balance":  stdDev,
		"avg_cluster_size": avgSize,
	}

	// Add quantization info if enabled
	if ivf.quantizer != nil {
		extra["quantization"] = ivf.quantizer.Type()
		extra["quantized_bytes_per_vector"] = ivf.quantizer.BytesPerVector()

		// Calculate compression ratio
		originalSize := ivf.dim * 4 // float32 = 4 bytes/value
		compressedSize := ivf.quantizer.BytesPerVector()
		extra["compression_ratio"] = float64(originalSize) / float64(compressedSize)
	}

	return IndexStats{
		Name:       "IVF",
		Dim:        ivf.dim,
		Count:      ivf.count,
		Deleted:    deletedCount,
		Active:     ivf.count - deletedCount,
		MemoryUsed: int64(ivf.estimateMemory()),
		DiskUsed:   0,
		Extra:      extra,
	}
}

// Export serializes the IVF index
func (ivf *IVFIndex) Export() ([]byte, error) {
	ivf.mu.RLock()
	defer ivf.mu.RUnlock()

	// Export format:
	// [4 bytes: dim] [4 bytes: nlist] [4 bytes: nprobe] [4 bytes: count]
	// [4 bytes: metric length] [metric string]
	// [1 byte: centroids trained?]
	// If trained: [centroids data]
	// [4 bytes: num vectors]
	// For each vector: [8 bytes: id] [4 bytes: cluster] [vector data] [1 byte: deleted]

	buf := make([]byte, 0, ivf.estimateMemory())

	// Header
	writeUint32 := func(v uint32) {
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, v)
		buf = append(buf, b...)
	}

	writeUint32(uint32(ivf.dim))
	writeUint32(uint32(ivf.nlist))
	writeUint32(uint32(ivf.nprobe))
	writeUint32(uint32(ivf.count))

	// Metric
	metricBytes := []byte(ivf.metric)
	writeUint32(uint32(len(metricBytes)))
	buf = append(buf, metricBytes...)

	// Centroids
	if ivf.centroids != nil {
		buf = append(buf, 1) // Trained flag
		for _, centroid := range ivf.centroids {
			for _, v := range centroid {
				b := make([]byte, 4)
				binary.LittleEndian.PutUint32(b, math.Float32bits(v))
				buf = append(buf, b...)
			}
		}
	} else {
		buf = append(buf, 0) // Not trained
	}

	// Vectors
	writeUint32(uint32(len(ivf.vectors)))
	for id, vec := range ivf.vectors {
		// ID
		idBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(idBytes, id)
		buf = append(buf, idBytes...)

		// Cluster ID
		clusterIdx := -1
		if cid, ok := ivf.clusterID[id]; ok {
			clusterIdx = cid
		}
		clusterBytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(clusterBytes, uint32(clusterIdx))
		buf = append(buf, clusterBytes...)

		// Vector data
		for _, v := range vec {
			vBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(vBytes, math.Float32bits(v))
			buf = append(buf, vBytes...)
		}

		// Deleted flag
		if ivf.deleted[id] {
			buf = append(buf, 1)
		} else {
			buf = append(buf, 0)
		}
	}

	return buf, nil
}

// Import deserializes the IVF index
func (ivf *IVFIndex) Import(data []byte) error {
	if len(data) < 20 {
		return fmt.Errorf("invalid IVF index data: too short")
	}

	ivf.mu.Lock()
	defer ivf.mu.Unlock()

	offset := 0

	readUint32 := func() uint32 {
		v := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4
		return v
	}

	// Header
	ivf.dim = int(readUint32())
	ivf.nlist = int(readUint32())
	ivf.nprobe = int(readUint32())
	ivf.count = int(readUint32())

	// Metric
	metricLen := int(readUint32())
	ivf.metric = string(data[offset : offset+metricLen])
	offset += metricLen

	// Centroids
	trained := data[offset] == 1
	offset++

	if trained {
		ivf.centroids = make([][]float32, ivf.nlist)
		for i := 0; i < ivf.nlist; i++ {
			centroid := make([]float32, ivf.dim)
			for j := 0; j < ivf.dim; j++ {
				bits := binary.LittleEndian.Uint32(data[offset : offset+4])
				centroid[j] = math.Float32frombits(bits)
				offset += 4
			}
			ivf.centroids[i] = centroid
		}
	}

	// Initialize maps
	numVectors := int(readUint32())
	ivf.vectors = make(map[uint64][]float32, numVectors)
	ivf.postings = make(map[int][]uint64)
	ivf.clusterID = make(map[uint64]int, numVectors)
	ivf.deleted = make(map[uint64]bool)
	ivf.clusterSizes = make([]int, ivf.nlist)

	// Vectors
	for i := 0; i < numVectors; i++ {
		// ID
		id := binary.LittleEndian.Uint64(data[offset : offset+8])
		offset += 8

		// Cluster ID
		clusterIdx := int(int32(readUint32())) // Signed for -1
		if clusterIdx >= 0 {
			ivf.clusterID[id] = clusterIdx
			ivf.postings[clusterIdx] = append(ivf.postings[clusterIdx], id)
			ivf.clusterSizes[clusterIdx]++
		}

		// Vector data
		vec := make([]float32, ivf.dim)
		for j := 0; j < ivf.dim; j++ {
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			vec[j] = math.Float32frombits(bits)
			offset += 4
		}
		ivf.vectors[id] = vec

		// Deleted flag
		if data[offset] == 1 {
			ivf.deleted[id] = true
		}
		offset++
	}

	return nil
}

// Helper methods

// trainCentroids performs k-means clustering to find centroids
func (ivf *IVFIndex) trainCentroids() error {
	if len(ivf.vectors) < ivf.nlist {
		return fmt.Errorf("not enough vectors to train: need %d, have %d", ivf.nlist, len(ivf.vectors))
	}

	// Collect all vectors into a slice
	vectors := make([][]float32, 0, len(ivf.vectors))
	for _, vec := range ivf.vectors {
		vectors = append(vectors, vec)
	}

	// Run k-means clustering
	centroids, err := kMeans(vectors, ivf.nlist, ivf.dim, 20) // 20 iterations
	if err != nil {
		return err
	}

	ivf.centroids = centroids

	// Reassign all existing vectors to clusters
	ivf.postings = make(map[int][]uint64)
	ivf.clusterSizes = make([]int, ivf.nlist)

	for id, vec := range ivf.vectors {
		clusterIdx := ivf.findNearestCluster(vec)
		ivf.clusterID[id] = clusterIdx
		ivf.postings[clusterIdx] = append(ivf.postings[clusterIdx], id)
		ivf.clusterSizes[clusterIdx]++
	}

	return nil
}

// quantizeExistingVectors converts all stored vectors to quantized format
func (ivf *IVFIndex) quantizeExistingVectors() error {
	if ivf.quantizer == nil {
		return nil
	}

	// Quantize all existing vectors
	for id, vec := range ivf.vectors {
		quantized, err := ivf.quantizer.Quantize(vec)
		if err != nil {
			return fmt.Errorf("failed to quantize vector %d: %w", id, err)
		}
		ivf.quantizedData[id] = quantized
	}

	// Clear unquantized vectors to save memory
	ivf.vectors = make(map[uint64][]float32)

	return nil
}

// findNearestCluster finds the nearest centroid to a vector
func (ivf *IVFIndex) findNearestCluster(vec []float32) int {
	if ivf.centroids == nil {
		return 0
	}

	minDist := float32(math.Inf(1))
	minIdx := 0

	for i, centroid := range ivf.centroids {
		dist := ivf.computeDistance(vec, centroid)
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}

	return minIdx
}

// findNearestClusters finds the nprobe nearest centroids to a query
func (ivf *IVFIndex) findNearestClusters(query []float32, nprobe int) []int {
	if ivf.centroids == nil {
		return []int{0}
	}

	var distValues []float32
	var err error

	// Use GPU if beneficial (many centroids)
	if ShouldUseGPU(1, len(ivf.centroids)) {
		distValues, err = GPUSingleDistance(query, ivf.centroids, ivf.metric)
		if err != nil {
			// Fall back to CPU if GPU fails
			distValues = ivf.computeCentroidDistancesCPU(query)
		}
	} else {
		// CPU computation for small number of centroids
		distValues = ivf.computeCentroidDistancesCPU(query)
	}

	// Build cluster distance pairs
	type clusterDist struct {
		idx  int
		dist float32
	}

	distances := make([]clusterDist, len(distValues))
	for i, dist := range distValues {
		distances[i] = clusterDist{
			idx:  i,
			dist: dist,
		}
	}

	// Sort by distance
	sort.Slice(distances, func(i, j int) bool {
		return distances[i].dist < distances[j].dist
	})

	// Return top nprobe cluster indices
	result := make([]int, nprobe)
	for i := 0; i < nprobe && i < len(distances); i++ {
		result[i] = distances[i].idx
	}

	return result
}

// computeCentroidDistancesCPU computes distances to all centroids on CPU
func (ivf *IVFIndex) computeCentroidDistancesCPU(query []float32) []float32 {
	distances := make([]float32, len(ivf.centroids))
	for i, centroid := range ivf.centroids {
		distances[i] = ivf.computeDistance(query, centroid)
	}
	return distances
}

// computeCandidateDistancesCPU computes distances to candidate vectors on CPU
func (ivf *IVFIndex) computeCandidateDistancesCPU(query []float32, candidates [][]float32) []float32 {
	distances := make([]float32, len(candidates))
	for i, vec := range candidates {
		distances[i] = ivf.computeDistance(query, vec)
	}
	return distances
}

// computeDistance computes distance between two vectors based on metric
func (ivf *IVFIndex) computeDistance(a, b []float32) float32 {
	if ivf.metric == "euclidean" {
		return euclideanDistance(a, b)
	}
	// Default: cosine distance
	return cosineDistance(a, b)
}

// bruteForceSearch performs exhaustive search when centroids not trained
func (ivf *IVFIndex) bruteForceSearch(query []float32, k int) ([]Result, error) {
	results := make([]Result, 0, len(ivf.vectors))

	for id, vec := range ivf.vectors {
		if ivf.deleted[id] {
			continue
		}

		distance := ivf.computeDistance(query, vec)
		results = append(results, Result{
			ID:       id,
			Distance: distance,
			Score:    1.0 / (1.0 + distance),
		})
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	// Return top-k
	if len(results) > k {
		results = results[:k]
	}

	return results, nil
}

// estimateMemory estimates memory usage in bytes
func (ivf *IVFIndex) estimateMemory() int {
	// Vectors (quantized or unquantized)
	var vectorMem int
	if ivf.quantizer != nil {
		// Quantized data: ID + compressed vector
		for _, data := range ivf.quantizedData {
			vectorMem += 8 + len(data)
		}
	} else {
		// Unquantized vectors: ID + float32 array
		vectorMem = len(ivf.vectors) * (8 + ivf.dim*4)
	}

	// Centroids
	centroidMem := 0
	if ivf.centroids != nil {
		centroidMem = len(ivf.centroids) * ivf.dim * 4
	}

	// Postings
	postingMem := 0
	for _, ids := range ivf.postings {
		postingMem += len(ids) * 8
	}

	// Maps overhead
	clusterIDMem := len(ivf.clusterID) * 12
	deletedMem := len(ivf.deleted) * 9

	return vectorMem + centroidMem + postingMem + clusterIDMem + deletedMem
}

// ExportJSON exports index metadata as JSON (for debugging)
func (ivf *IVFIndex) ExportJSON() ([]byte, error) {
	ivf.mu.RLock()
	defer ivf.mu.RUnlock()

	export := map[string]interface{}{
		"type":            "ivf",
		"dim":             ivf.dim,
		"nlist":           ivf.nlist,
		"nprobe":          ivf.nprobe,
		"count":           ivf.count,
		"metric":          ivf.metric,
		"centroids_ready": ivf.centroids != nil,
		"stats":           ivf.Stats(),
	}

	return json.Marshal(export)
}

// K-means clustering implementation

// kMeans performs k-means clustering on vectors using mini-batch k-means for scalability
func kMeans(vectors [][]float32, k int, dim int, maxIterations int) ([][]float32, error) {
	if len(vectors) < k {
		return nil, fmt.Errorf("not enough vectors for k-means: need %d, have %d", k, len(vectors))
	}

	// Use mini-batch k-means for large datasets (>10K vectors)
	if len(vectors) > 10000 {
		return miniBatchKMeans(vectors, k, dim, maxIterations)
	}

	// Standard k-means for small datasets
	return standardKMeans(vectors, k, dim, maxIterations)
}

// standardKMeans performs standard k-means clustering (for small datasets)
func standardKMeans(vectors [][]float32, k int, dim int, maxIterations int) ([][]float32, error) {
	// Initialize centroids using k-means++ algorithm
	centroids := kMeansPlusPlus(vectors, k, dim)

	// Iterate to convergence
	for iter := 0; iter < maxIterations; iter++ {
		// Assignment step: assign each vector to nearest centroid
		assignments := make([]int, len(vectors))
		for i, vec := range vectors {
			minDist := float32(math.Inf(1))
			minIdx := 0

			for j, centroid := range centroids {
				dist := euclideanDistance(vec, centroid)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}

			assignments[i] = minIdx
		}

		// Update step: recompute centroids
		newCentroids := make([][]float32, k)
		counts := make([]int, k)

		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float32, dim)
		}

		for i, vec := range vectors {
			clusterIdx := assignments[i]
			for d := 0; d < dim; d++ {
				newCentroids[clusterIdx][d] += vec[d]
			}
			counts[clusterIdx]++
		}

		// Average
		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				for d := 0; d < dim; d++ {
					newCentroids[i][d] /= float32(counts[i])
				}
			} else {
				// Empty cluster: reinitialize with random vector
				newCentroids[i] = vectors[rand.Intn(len(vectors))]
			}
		}

		// Check convergence (if centroids didn't change much)
		converged := true
		threshold := float32(1e-4)
		for i := 0; i < k; i++ {
			dist := euclideanDistance(centroids[i], newCentroids[i])
			if dist > threshold {
				converged = false
				break
			}
		}

		centroids = newCentroids

		if converged {
			break
		}
	}

	return centroids, nil
}

// miniBatchKMeans performs mini-batch k-means for large datasets
// Processes random batches instead of all vectors, providing O(batch_size) complexity per iteration
func miniBatchKMeans(vectors [][]float32, k int, dim int, maxIterations int) ([][]float32, error) {
	// Initialize centroids using k-means++ with sampling for speed
	centroids := kMeansPlusPlusSampled(vectors, k, dim, 5000)

	// Batch size: max(sqrt(n), 1000), capped at 5000
	batchSize := int(math.Sqrt(float64(len(vectors))))
	if batchSize < 1000 {
		batchSize = 1000
	}
	if batchSize > 5000 {
		batchSize = 5000
	}

	// Per-cluster counts for weighted averaging
	clusterCounts := make([]int, k)

	// Iterate with mini-batches
	for iter := 0; iter < maxIterations; iter++ {
		// Sample random batch
		batch := sampleVectors(vectors, batchSize)

		// Assignment step: assign batch vectors to nearest centroid
		assignments := make([]int, len(batch))
		for i, vec := range batch {
			minDist := float32(math.Inf(1))
			minIdx := 0

			for j, centroid := range centroids {
				dist := euclideanDistance(vec, centroid)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}

			assignments[i] = minIdx
		}

		// Update step: incrementally update centroids using per-cluster learning rate
		for i, vec := range batch {
			clusterIdx := assignments[i]
			clusterCounts[clusterIdx]++

			// Learning rate: 1 / count (classic mini-batch k-means update)
			learningRate := float32(1.0 / float64(clusterCounts[clusterIdx]))

			// Update centroid: c = c + lr * (v - c)
			for d := 0; d < dim; d++ {
				centroids[clusterIdx][d] += learningRate * (vec[d] - centroids[clusterIdx][d])
			}
		}

		// Check convergence every 5 iterations (less frequent for mini-batch)
		if iter%5 == 4 {
			// Sample validation batch
			validationBatch := sampleVectors(vectors, min(2000, len(vectors)))

			// Compute assignment stability
			stable := 0
			for _, vec := range validationBatch {
				minDist := float32(math.Inf(1))

				for _, centroid := range centroids {
					dist := euclideanDistance(vec, centroid)
					if dist < minDist {
						minDist = dist
					}
				}

				// Check if this is a stable assignment (close to centroid)
				threshold := float32(0.1)
				if minDist < threshold {
					stable++
				}
			}

			// Converged if >90% of vectors have stable assignments
			if float64(stable)/float64(len(validationBatch)) > 0.9 {
				break
			}
		}
	}

	return centroids, nil
}

// sampleVectors randomly samples n vectors from the input
func sampleVectors(vectors [][]float32, n int) [][]float32 {
	if n >= len(vectors) {
		return vectors
	}

	// Create random permutation of first n indices
	indices := rand.Perm(len(vectors))[:n]

	sample := make([][]float32, n)
	for i, idx := range indices {
		sample[i] = vectors[idx]
	}

	return sample
}

// kMeansPlusPlusSampled performs k-means++ initialization with sampling for large datasets
func kMeansPlusPlusSampled(vectors [][]float32, k int, dim int, sampleSize int) [][]float32 {
	// For large datasets, sample for k-means++ initialization
	var workingSet [][]float32
	if len(vectors) > sampleSize {
		workingSet = sampleVectors(vectors, sampleSize)
	} else {
		workingSet = vectors
	}

	return kMeansPlusPlus(workingSet, k, dim)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// kMeansPlusPlus initializes centroids using k-means++ algorithm
func kMeansPlusPlus(vectors [][]float32, k int, dim int) [][]float32 {
	centroids := make([][]float32, 0, k)

	// Choose first centroid randomly
	firstIdx := rand.Intn(len(vectors))
	centroid := make([]float32, dim)
	copy(centroid, vectors[firstIdx])
	centroids = append(centroids, centroid)

	// Choose remaining centroids
	for len(centroids) < k {
		// Compute distances to nearest centroid for each vector
		distances := make([]float32, len(vectors))
		sumDist := float32(0)

		for i, vec := range vectors {
			minDist := float32(math.Inf(1))
			for _, c := range centroids {
				dist := euclideanDistance(vec, c)
				if dist < minDist {
					minDist = dist
				}
			}
			distances[i] = minDist * minDist // Squared distance
			sumDist += distances[i]
		}

		// Choose next centroid with probability proportional to squared distance
		target := rand.Float32() * sumDist
		cumSum := float32(0)
		nextIdx := 0

		for i, dist := range distances {
			cumSum += dist
			if cumSum >= target {
				nextIdx = i
				break
			}
		}

		centroid := make([]float32, dim)
		copy(centroid, vectors[nextIdx])
		centroids = append(centroids, centroid)
	}

	return centroids
}

// Distance functions

func euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func cosineDistance(a, b []float32) float32 {
	var dot, normA, normB float32

	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - similarity
}
