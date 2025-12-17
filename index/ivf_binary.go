package index

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
)

// ======================================================================================
// IVF-Binary Index: Combines IVF clustering with Binary Quantization
// ======================================================================================
// Architecture:
// 1. Coarse quantizer: K-means clusters the vector space into nlist partitions
// 2. Fine quantizer: Binary quantization within each cluster
// 3. Search: Probe nprobe clusters, binary search within each, rescore top candidates
//
// Performance characteristics:
// - 30x memory reduction (binary quantization)
// - 10-100x search speedup (cluster pruning)
// - 95%+ recall with proper reranking
// ======================================================================================

// IVFBinaryIndex combines IVF clustering with binary quantization
type IVFBinaryIndex struct {
	mu sync.RWMutex

	// Configuration
	dim    int // Vector dimension
	nlist  int // Number of clusters
	nprobe int // Number of clusters to search

	// Coarse quantizer (cluster centroids)
	centroids    [][]float32 // [nlist][dim]
	centroidNorm []float32   // Precomputed L2 norms for fast distance

	// Per-cluster binary indexes
	clusters []*clusterData

	// ID mapping
	idMap map[uint64]clusterLoc // ID -> (cluster, position)

	// Training state
	trained bool

	// Stats
	totalVectors int
}

// clusterData holds data for a single IVF cluster
type clusterData struct {
	mu          sync.RWMutex
	binaryData  []byte    // Binary quantized vectors
	ids         []uint64  // Vector IDs
	bytesPerVec int       // Bytes per binary vector
	thresholds  []float32 // Binary quantization thresholds
}

// clusterLoc identifies a vector's location
type clusterLoc struct {
	cluster int
	pos     int
}

// IVFBinaryConfig holds configuration for IVF-Binary index
type IVFBinaryConfig struct {
	Dim         int     // Vector dimension (required)
	Nlist       int     // Number of clusters (default: sqrt(n))
	Nprobe      int     // Clusters to search (default: nlist/10)
	TrainRatio  float64 // Fraction of data for training (default: 0.1)
}

// NewIVFBinaryIndex creates a new IVF-Binary index
func NewIVFBinaryIndex(config IVFBinaryConfig) (*IVFBinaryIndex, error) {
	if config.Dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if config.Nlist <= 0 {
		config.Nlist = 100 // Will be adjusted during training
	}
	if config.Nprobe <= 0 {
		config.Nprobe = 10
	}

	bytesPerVec := (config.Dim + 7) / 8

	clusters := make([]*clusterData, config.Nlist)
	for i := range clusters {
		clusters[i] = &clusterData{
			bytesPerVec: bytesPerVec,
			thresholds:  make([]float32, config.Dim),
		}
	}

	return &IVFBinaryIndex{
		dim:      config.Dim,
		nlist:    config.Nlist,
		nprobe:   config.Nprobe,
		clusters: clusters,
		idMap:    make(map[uint64]clusterLoc),
	}, nil
}

// Train trains the index on a sample of vectors
func (idx *IVFBinaryIndex) Train(vectors []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors)%idx.dim != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension")
	}

	numVectors := len(vectors) / idx.dim
	if numVectors < idx.nlist {
		// Adjust nlist if not enough vectors
		idx.nlist = numVectors / 10
		if idx.nlist < 1 {
			idx.nlist = 1
		}
	}

	// Train coarse quantizer (k-means)
	centroids, err := kMeansPlusPlusFlat(vectors, idx.nlist, idx.dim, 25)
	if err != nil {
		return fmt.Errorf("k-means training failed: %w", err)
	}
	idx.centroids = centroids

	// Precompute centroid norms for fast distance
	idx.centroidNorm = make([]float32, idx.nlist)
	for i, c := range centroids {
		var norm float32
		for _, v := range c {
			norm += v * v
		}
		idx.centroidNorm[i] = norm
	}

	// Reinitialize clusters with correct count
	idx.clusters = make([]*clusterData, idx.nlist)
	bytesPerVec := (idx.dim + 7) / 8
	for i := range idx.clusters {
		idx.clusters[i] = &clusterData{
			bytesPerVec: bytesPerVec,
			thresholds:  make([]float32, idx.dim),
		}
	}

	// Train binary thresholds per cluster using assigned vectors
	assignments := idx.assignToClusters(vectors)
	for c := 0; c < idx.nlist; c++ {
		// Collect vectors assigned to this cluster
		var clusterVecs []float32
		for i, a := range assignments {
			if a == c {
				clusterVecs = append(clusterVecs, vectors[i*idx.dim:(i+1)*idx.dim]...)
			}
		}

		if len(clusterVecs) > 0 {
			// Compute per-dimension mean as threshold
			numInCluster := len(clusterVecs) / idx.dim
			for d := 0; d < idx.dim; d++ {
				var sum float32
				for i := 0; i < numInCluster; i++ {
					sum += clusterVecs[i*idx.dim+d]
				}
				idx.clusters[c].thresholds[d] = sum / float32(numInCluster)
			}
		}
	}

	idx.trained = true
	return nil
}

// Add adds a single vector to the index
func (idx *IVFBinaryIndex) Add(id uint64, vector []float32) error {
	if len(vector) != idx.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dim, len(vector))
	}

	idx.mu.Lock()
	if !idx.trained {
		idx.mu.Unlock()
		return fmt.Errorf("index must be trained before adding vectors")
	}

	// Find nearest cluster
	clusterIdx := idx.findNearestCluster(vector)
	cluster := idx.clusters[clusterIdx]
	idx.mu.Unlock()

	// Quantize and add to cluster
	cluster.mu.Lock()
	binary := quantizeBinary(vector, cluster.thresholds)
	pos := len(cluster.ids)
	cluster.binaryData = append(cluster.binaryData, binary...)
	cluster.ids = append(cluster.ids, id)
	cluster.mu.Unlock()

	idx.mu.Lock()
	idx.idMap[id] = clusterLoc{cluster: clusterIdx, pos: pos}
	idx.totalVectors++
	idx.mu.Unlock()

	return nil
}

// AddBatch adds multiple vectors efficiently
func (idx *IVFBinaryIndex) AddBatch(ids []uint64, vectors []float32) error {
	if len(vectors)%idx.dim != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension")
	}
	numVectors := len(vectors) / idx.dim
	if len(ids) != numVectors {
		return fmt.Errorf("ids count mismatch: expected %d, got %d", numVectors, len(ids))
	}

	idx.mu.Lock()
	if !idx.trained {
		idx.mu.Unlock()
		return fmt.Errorf("index must be trained before adding vectors")
	}
	idx.mu.Unlock()

	// Assign all vectors to clusters
	assignments := idx.assignToClusters(vectors)

	// Group by cluster for batch insertion
	clusterBatches := make([][]int, idx.nlist) // cluster -> vector indices
	for i, c := range assignments {
		clusterBatches[c] = append(clusterBatches[c], i)
	}

	// Add to each cluster
	for c := 0; c < idx.nlist; c++ {
		batch := clusterBatches[c]
		if len(batch) == 0 {
			continue
		}

		cluster := idx.clusters[c]
		cluster.mu.Lock()

		for _, vecIdx := range batch {
			vec := vectors[vecIdx*idx.dim : (vecIdx+1)*idx.dim]
			binary := quantizeBinary(vec, cluster.thresholds)

			pos := len(cluster.ids)
			cluster.binaryData = append(cluster.binaryData, binary...)
			cluster.ids = append(cluster.ids, ids[vecIdx])

			idx.mu.Lock()
			idx.idMap[ids[vecIdx]] = clusterLoc{cluster: c, pos: pos}
			idx.totalVectors++
			idx.mu.Unlock()
		}

		cluster.mu.Unlock()
	}

	return nil
}

// Search finds top-k nearest neighbors
func (idx *IVFBinaryIndex) Search(query []float32, k int) ([]SearchResult, error) {
	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.dim, len(query))
	}

	idx.mu.RLock()
	if !idx.trained {
		idx.mu.RUnlock()
		return nil, fmt.Errorf("index must be trained before searching")
	}
	nprobe := idx.nprobe
	if nprobe > idx.nlist {
		nprobe = idx.nlist
	}
	idx.mu.RUnlock()

	// Find nprobe nearest clusters
	clusterDists := idx.computeClusterDistances(query)
	topClusters := selectTopKIndices(clusterDists, nprobe)

	// Search within each cluster in parallel
	type candidate struct {
		id       uint64
		hamming  int
		cluster  int
	}

	var allCandidates []candidate
	var candidatesMu sync.Mutex
	var wg sync.WaitGroup

	for _, c := range topClusters {
		wg.Add(1)
		go func(clusterIdx int) {
			defer wg.Done()

			cluster := idx.clusters[clusterIdx]
			cluster.mu.RLock()
			defer cluster.mu.RUnlock()

			if len(cluster.ids) == 0 {
				return
			}

			// Quantize query with cluster's thresholds
			binaryQuery := quantizeBinary(query, cluster.thresholds)

			// Compute Hamming distances
			distances := HammingDistanceBatch(binaryQuery, cluster.binaryData, cluster.bytesPerVec)

			// Collect candidates
			candidatesMu.Lock()
			for i, dist := range distances {
				allCandidates = append(allCandidates, candidate{
					id:      cluster.ids[i],
					hamming: dist,
					cluster: clusterIdx,
				})
			}
			candidatesMu.Unlock()
		}(c)
	}

	wg.Wait()

	// Sort by Hamming distance and return top-k
	sort.Slice(allCandidates, func(i, j int) bool {
		return allCandidates[i].hamming < allCandidates[j].hamming
	})

	if k > len(allCandidates) {
		k = len(allCandidates)
	}

	results := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		results[i] = SearchResult{
			ID:    allCandidates[i].id,
			Score: HammingToCosineSimilarity(allCandidates[i].hamming, idx.dim),
		}
	}

	return results, nil
}

// SearchWithRescore searches and rescores with original vectors
func (idx *IVFBinaryIndex) SearchWithRescore(query []float32, k int, candidates int, getVector func(uint64) []float32) ([]SearchResult, error) {
	// Get more candidates than needed
	if candidates < k*10 {
		candidates = k * 10
	}

	binaryCandidates, err := idx.Search(query, candidates)
	if err != nil {
		return nil, err
	}

	// Rescore with exact cosine similarity
	type scored struct {
		id    uint64
		score float32
	}
	rescored := make([]scored, 0, len(binaryCandidates))

	for _, c := range binaryCandidates {
		vec := getVector(c.ID)
		if vec == nil {
			continue
		}
		score := cosineSimilarity(query, vec)
		rescored = append(rescored, scored{id: c.ID, score: score})
	}

	// Sort by true score
	sort.Slice(rescored, func(i, j int) bool {
		return rescored[i].score > rescored[j].score
	})

	if k > len(rescored) {
		k = len(rescored)
	}

	results := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		results[i] = SearchResult{
			ID:    rescored[i].id,
			Score: rescored[i].score,
		}
	}

	return results, nil
}

// Size returns number of vectors in index
func (idx *IVFBinaryIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.totalVectors
}

// MemoryUsage returns approximate memory usage in bytes
func (idx *IVFBinaryIndex) MemoryUsage() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	var total int64

	// Centroids
	total += int64(idx.nlist * idx.dim * 4)

	// Cluster data
	for _, c := range idx.clusters {
		c.mu.RLock()
		total += int64(len(c.binaryData))
		total += int64(len(c.ids) * 8)
		total += int64(len(c.thresholds) * 4)
		c.mu.RUnlock()
	}

	// ID map
	total += int64(len(idx.idMap) * 24) // approximate

	return total
}

// Stats returns index statistics
func (idx *IVFBinaryIndex) Stats() map[string]interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	clusterSizes := make([]int, idx.nlist)
	for i, c := range idx.clusters {
		c.mu.RLock()
		clusterSizes[i] = len(c.ids)
		c.mu.RUnlock()
	}

	return map[string]interface{}{
		"type":          "ivf_binary",
		"dimension":     idx.dim,
		"nlist":         idx.nlist,
		"nprobe":        idx.nprobe,
		"total_vectors": idx.totalVectors,
		"trained":       idx.trained,
		"memory_bytes":  idx.MemoryUsage(),
		"cluster_sizes": clusterSizes,
	}
}

// ======================================================================================
// Helper Functions
// ======================================================================================

func (idx *IVFBinaryIndex) assignToClusters(vectors []float32) []int {
	numVectors := len(vectors) / idx.dim
	assignments := make([]int, numVectors)

	// Parallel assignment
	numWorkers := runtime.NumCPU()
	chunkSize := (numVectors + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * chunkSize
			end := start + chunkSize
			if end > numVectors {
				end = numVectors
			}

			for i := start; i < end; i++ {
				vec := vectors[i*idx.dim : (i+1)*idx.dim]
				assignments[i] = idx.findNearestCluster(vec)
			}
		}(w)
	}
	wg.Wait()

	return assignments
}

func (idx *IVFBinaryIndex) findNearestCluster(vec []float32) int {
	minDist := float32(math.Inf(1))
	minIdx := 0

	for i, c := range idx.centroids {
		dist := euclideanDistSquared(vec, c)
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}

	return minIdx
}

func (idx *IVFBinaryIndex) computeClusterDistances(query []float32) []float32 {
	distances := make([]float32, idx.nlist)
	for i, c := range idx.centroids {
		distances[i] = euclideanDistSquared(query, c)
	}
	return distances
}

func selectTopKIndices(values []float32, k int) []int {
	type idxVal struct {
		idx int
		val float32
	}
	pairs := make([]idxVal, len(values))
	for i, v := range values {
		pairs[i] = idxVal{idx: i, val: v}
	}

	// Partial sort for top-k smallest
	for i := 0; i < k && i < len(pairs); i++ {
		minIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].val < pairs[minIdx].val {
				minIdx = j
			}
		}
		pairs[i], pairs[minIdx] = pairs[minIdx], pairs[i]
	}

	result := make([]int, k)
	for i := 0; i < k && i < len(pairs); i++ {
		result[i] = pairs[i].idx
	}
	return result
}

func quantizeBinary(vec []float32, thresholds []float32) []byte {
	bytesPerVec := (len(vec) + 7) / 8
	binary := make([]byte, bytesPerVec)

	for d := 0; d < len(vec); d++ {
		if vec[d] > thresholds[d] {
			byteIdx := d / 8
			bitIdx := uint(d % 8)
			binary[byteIdx] |= 1 << bitIdx
		}
	}

	return binary
}

func euclideanDistSquared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// kMeansPlusPlusFlat implements k-means++ initialization for flat vector arrays
// This version takes flat []float32 arrays (more memory efficient for large datasets)
func kMeansPlusPlusFlat(vectors []float32, k int, dim int, maxIter int) ([][]float32, error) {
	numVectors := len(vectors) / dim
	if numVectors < k {
		return nil, fmt.Errorf("not enough vectors: need %d, have %d", k, numVectors)
	}

	// K-means++ initialization
	centroids := make([][]float32, k)

	// First centroid: random
	firstIdx := rand.Intn(numVectors)
	centroids[0] = make([]float32, dim)
	copy(centroids[0], vectors[firstIdx*dim:(firstIdx+1)*dim])

	// Remaining centroids: proportional to squared distance
	minDists := make([]float32, numVectors)
	for i := range minDists {
		minDists[i] = math.MaxFloat32
	}

	for c := 1; c < k; c++ {
		// Update min distances
		var totalDist float32
		for i := 0; i < numVectors; i++ {
			vec := vectors[i*dim : (i+1)*dim]
			dist := euclideanDistSquared(vec, centroids[c-1])
			if dist < minDists[i] {
				minDists[i] = dist
			}
			totalDist += minDists[i]
		}

		// Sample proportional to distance
		threshold := rand.Float32() * totalDist
		var cumDist float32
		selectedIdx := 0
		for i := 0; i < numVectors; i++ {
			cumDist += minDists[i]
			if cumDist >= threshold {
				selectedIdx = i
				break
			}
		}

		centroids[c] = make([]float32, dim)
		copy(centroids[c], vectors[selectedIdx*dim:(selectedIdx+1)*dim])
	}

	// Lloyd's algorithm iterations
	assignments := make([]int, numVectors)

	for iter := 0; iter < maxIter; iter++ {
		// Assignment step
		for i := 0; i < numVectors; i++ {
			vec := vectors[i*dim : (i+1)*dim]
			minDist := float32(math.Inf(1))
			minIdx := 0
			for j, c := range centroids {
				dist := euclideanDistSquared(vec, c)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}
			assignments[i] = minIdx
		}

		// Update step
		newCentroids := make([][]float32, k)
		counts := make([]int, k)
		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float32, dim)
		}

		for i := 0; i < numVectors; i++ {
			c := assignments[i]
			vec := vectors[i*dim : (i+1)*dim]
			for d := 0; d < dim; d++ {
				newCentroids[c][d] += vec[d]
			}
			counts[c]++
		}

		for i := 0; i < k; i++ {
			if counts[i] > 0 {
				for d := 0; d < dim; d++ {
					newCentroids[i][d] /= float32(counts[i])
				}
			} else {
				// Empty cluster: reinitialize randomly
				idx := rand.Intn(numVectors)
				copy(newCentroids[i], vectors[idx*dim:(idx+1)*dim])
			}
		}

		centroids = newCentroids
	}

	return centroids, nil
}

// SearchResult holds a search result
type SearchResult struct {
	ID    uint64
	Score float32
}
