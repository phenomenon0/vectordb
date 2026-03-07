package index

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
)

// ======================================================================================
// ADC (Asymmetric Distance Computation) for Product Quantization
// ======================================================================================
// ADC precomputes distances from query to all centroids, enabling O(M) lookups
// instead of O(M × K × dsub) per vector comparison.
//
// Performance comparison (768d vectors, M=96, K=256):
// - Naive PQ:  ~50 QPS    (recompute distances for each comparison)
// - ADC:       ~2,500 QPS (precomputed distance tables)
// - FastScan:  ~25,000 QPS (SIMD + 4-bit codes)
// ======================================================================================

// DistanceTable holds precomputed distances from query subvectors to centroids
// table[m][k] = squared L2 distance from query subvector m to centroid k
type DistanceTable struct {
	tables [][]float32 // [M][K] distance values
	m      int         // Number of subvectors
	k      int         // Number of centroids per subvector
}

// ComputeDistanceTable precomputes distances from query to all centroids
// This is the key to ADC - computed once per query, used for all vector comparisons
func (pq *ProductQuantizer) ComputeDistanceTable(query []float32) (*DistanceTable, error) {
	if len(query) != pq.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", pq.dim, len(query))
	}
	if !pq.trained {
		return nil, fmt.Errorf("quantizer must be trained")
	}

	dt := &DistanceTable{
		tables: make([][]float32, pq.m),
		m:      pq.m,
		k:      pq.ksub,
	}

	// Compute distance from each query subvector to each centroid
	for m := 0; m < pq.m; m++ {
		dt.tables[m] = make([]float32, pq.ksub)
		querySubvec := query[m*pq.dsub : (m+1)*pq.dsub]

		for k := 0; k < pq.ksub; k++ {
			// Squared L2 distance
			var dist float32
			centroid := pq.codebooks[m][k]
			for d := 0; d < pq.dsub; d++ {
				diff := querySubvec[d] - centroid[d]
				dist += diff * diff
			}
			dt.tables[m][k] = dist
		}
	}

	return dt, nil
}

// LookupDistance computes distance to a single PQ-encoded vector using table lookup
// O(M) operations instead of O(M × K × dsub)
func (dt *DistanceTable) LookupDistance(codes []byte) float32 {
	var dist float32
	for m := 0; m < dt.m && m < len(codes); m++ {
		dist += dt.tables[m][codes[m]]
	}
	return dist
}

// LookupDistanceBatch computes distances to multiple PQ-encoded vectors
// Uses table lookup for O(N × M) total operations
func (dt *DistanceTable) LookupDistanceBatch(codes []byte, numVectors int) []float32 {
	distances := make([]float32, numVectors)
	bytesPerVec := dt.m

	for i := 0; i < numVectors; i++ {
		offset := i * bytesPerVec
		var dist float32
		for m := 0; m < dt.m; m++ {
			dist += dt.tables[m][codes[offset+m]]
		}
		distances[i] = dist
	}

	return distances
}

// LookupDistanceBatchParallel computes distances in parallel for large batches
func (dt *DistanceTable) LookupDistanceBatchParallel(codes []byte, numVectors int) []float32 {
	distances := make([]float32, numVectors)
	bytesPerVec := dt.m

	// Use parallel processing for large batches
	numWorkers := runtime.NumCPU()
	if numVectors < numWorkers*100 {
		// Small batch - use single-threaded
		return dt.LookupDistanceBatch(codes, numVectors)
	}

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
				offset := i * bytesPerVec
				var dist float32
				for m := 0; m < dt.m; m++ {
					dist += dt.tables[m][codes[offset+m]]
				}
				distances[i] = dist
			}
		}(w)
	}

	wg.Wait()
	return distances
}

// ======================================================================================
// PQIndex - Full ADC-accelerated Product Quantization Index
// ======================================================================================

// PQIndex is a complete PQ index with ADC-accelerated search
type PQIndex struct {
	mu sync.RWMutex

	// Quantizer
	pq *ProductQuantizer

	// Stored data
	codes []byte   // PQ codes for all vectors
	ids   []uint64 // Vector IDs

	// Configuration
	dim int
}

// PQIndexConfig holds configuration for PQ index
type PQIndexConfig struct {
	Dim       int // Vector dimension (required)
	M         int // Number of subvectors (default: dim/8, max 256)
	Ksub      int // Centroids per subvector (default: 256)
	TrainSize int // Number of vectors to use for training
}

// NewPQIndex creates a new PQ index with ADC search
func NewPQIndex(config PQIndexConfig) (*PQIndex, error) {
	if config.Dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	// Default M to dim/8, capped at reasonable values
	m := config.M
	if m <= 0 {
		m = config.Dim / 8
		if m < 4 {
			m = 4
		}
		if m > 128 {
			m = 128
		}
	}

	// Ensure dimension is divisible by M
	if config.Dim%m != 0 {
		// Find nearest valid M
		for m > 1 && config.Dim%m != 0 {
			m--
		}
	}

	ksub := config.Ksub
	if ksub <= 0 {
		ksub = 256 // 8-bit codes
	}

	pq, err := NewProductQuantizer(config.Dim, m, ksub)
	if err != nil {
		return nil, err
	}

	return &PQIndex{
		pq:  pq,
		dim: config.Dim,
	}, nil
}

// Train trains the PQ codebooks on sample vectors
func (idx *PQIndex) Train(vectors []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	return idx.pq.Train(vectors, 25) // 25 k-means iterations
}

// Add adds a single vector to the index
func (idx *PQIndex) Add(id uint64, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vector) != idx.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dim, len(vector))
	}

	if !idx.pq.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	codes, err := idx.pq.Quantize(vector)
	if err != nil {
		return err
	}

	idx.codes = append(idx.codes, codes...)
	idx.ids = append(idx.ids, id)
	return nil
}

// AddBatch adds multiple vectors efficiently
func (idx *PQIndex) AddBatch(ids []uint64, vectors []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors)%idx.dim != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension")
	}
	numVectors := len(vectors) / idx.dim
	if len(ids) != numVectors {
		return fmt.Errorf("ids count mismatch")
	}

	if !idx.pq.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	codes, err := idx.pq.Quantize(vectors)
	if err != nil {
		return err
	}

	idx.codes = append(idx.codes, codes...)
	idx.ids = append(idx.ids, ids...)
	return nil
}

// PQSearchResult holds a search result
type PQSearchResult struct {
	ID       uint64
	Distance float32
}

// Search finds top-k nearest neighbors using ADC
func (idx *PQIndex) Search(query []float32, k int) ([]PQSearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	numVectors := len(idx.ids)
	if numVectors == 0 {
		return nil, nil
	}

	// Precompute distance table (the key ADC optimization)
	dt, err := idx.pq.ComputeDistanceTable(query)
	if err != nil {
		return nil, err
	}

	// Compute all distances using table lookup
	distances := dt.LookupDistanceBatchParallel(idx.codes, numVectors)

	// Find top-k using partial sort
	type idxDist struct {
		idx  int
		dist float32
	}

	// For small k, use simple selection
	if k > numVectors {
		k = numVectors
	}

	pairs := make([]idxDist, numVectors)
	for i, d := range distances {
		pairs[i] = idxDist{idx: i, dist: d}
	}

	// Partial sort for top-k smallest distances
	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].dist < pairs[minIdx].dist {
				minIdx = j
			}
		}
		pairs[i], pairs[minIdx] = pairs[minIdx], pairs[i]
	}

	results := make([]PQSearchResult, k)
	for i := 0; i < k; i++ {
		results[i] = PQSearchResult{
			ID:       idx.ids[pairs[i].idx],
			Distance: pairs[i].dist,
		}
	}

	return results, nil
}

// SearchWithHeap uses a max-heap for more efficient top-k selection
func (idx *PQIndex) SearchWithHeap(query []float32, k int) ([]PQSearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	numVectors := len(idx.ids)
	if numVectors == 0 {
		return nil, nil
	}

	// Precompute distance table
	dt, err := idx.pq.ComputeDistanceTable(query)
	if err != nil {
		return nil, err
	}

	if k > numVectors {
		k = numVectors
	}

	// Use a sorted slice as heap (simple but effective for small k)
	type candidate struct {
		id   uint64
		dist float32
	}
	heap := make([]candidate, 0, k+1)

	bytesPerVec := idx.pq.m
	for i := 0; i < numVectors; i++ {
		offset := i * bytesPerVec
		dist := dt.LookupDistance(idx.codes[offset : offset+bytesPerVec])

		if len(heap) < k {
			heap = append(heap, candidate{id: idx.ids[i], dist: dist})
			// Keep sorted (insertion sort for small k)
			for j := len(heap) - 1; j > 0 && heap[j].dist < heap[j-1].dist; j-- {
				heap[j], heap[j-1] = heap[j-1], heap[j]
			}
		} else if dist < heap[k-1].dist {
			// Replace worst and re-sort
			heap[k-1] = candidate{id: idx.ids[i], dist: dist}
			for j := k - 1; j > 0 && heap[j].dist < heap[j-1].dist; j-- {
				heap[j], heap[j-1] = heap[j-1], heap[j]
			}
		}
	}

	results := make([]PQSearchResult, len(heap))
	for i, c := range heap {
		results[i] = PQSearchResult{ID: c.id, Distance: c.dist}
	}
	return results, nil
}

// Size returns number of vectors in index
func (idx *PQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.ids)
}

// MemoryUsage returns approximate memory usage in bytes
func (idx *PQIndex) MemoryUsage() int64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Codes: numVectors * M bytes
	// IDs: numVectors * 8 bytes
	// Codebooks: M * K * dsub * 4 bytes
	codesSize := int64(len(idx.codes))
	idsSize := int64(len(idx.ids) * 8)
	codebooksSize := int64(idx.pq.m * idx.pq.ksub * idx.pq.dsub * 4)

	return codesSize + idsSize + codebooksSize
}

// CompressionRatio returns the compression ratio vs full float32 storage
func (idx *PQIndex) CompressionRatio() float64 {
	if len(idx.ids) == 0 {
		return 0
	}
	originalSize := float64(len(idx.ids) * idx.dim * 4)
	compressedSize := float64(len(idx.codes))
	return originalSize / compressedSize
}

// Stats returns index statistics
func (idx *PQIndex) Stats() map[string]interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return map[string]interface{}{
		"type":              "pq_adc",
		"dimension":         idx.dim,
		"num_subvectors":    idx.pq.m,
		"centroids_per_sub": idx.pq.ksub,
		"sub_dimension":     idx.pq.dsub,
		"total_vectors":     len(idx.ids),
		"memory_bytes":      idx.MemoryUsage(),
		"compression_ratio": idx.CompressionRatio(),
		"trained":           idx.pq.trained,
	}
}

// ======================================================================================
// IVF-PQ Index with ADC (combines IVF clustering with PQ compression)
// ======================================================================================

// IVFPQIndex combines IVF clustering with PQ compression and ADC search
type IVFPQIndex struct {
	mu sync.RWMutex

	// Configuration
	dim    int
	nlist  int // Number of clusters
	nprobe int // Number of clusters to search

	// Coarse quantizer (cluster centroids)
	centroids [][]float32 // [nlist][dim]

	// Per-cluster PQ indexes
	clusters []*clusterPQData

	// Shared PQ quantizer (trained on all data)
	pq *ProductQuantizer

	// ID mapping
	idMap map[uint64]ivfpqLoc

	// State
	trained      bool
	totalVectors int
}

type clusterPQData struct {
	mu    sync.RWMutex
	codes []byte   // PQ codes for vectors in this cluster
	ids   []uint64 // Vector IDs
}

type ivfpqLoc struct {
	cluster int
	pos     int
}

// IVFPQConfig holds configuration for IVF-PQ index
type IVFPQConfig struct {
	Dim    int // Vector dimension
	Nlist  int // Number of clusters (default: sqrt(n) or 100)
	Nprobe int // Clusters to search (default: 10)
	M      int // PQ subvectors (default: dim/8)
	Ksub   int // PQ centroids per subvector (default: 256)
}

// NewIVFPQIndex creates a new IVF-PQ index
func NewIVFPQIndex(config IVFPQConfig) (*IVFPQIndex, error) {
	if config.Dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}

	nlist := config.Nlist
	if nlist <= 0 {
		nlist = 100
	}

	nprobe := config.Nprobe
	if nprobe <= 0 {
		nprobe = 10
	}
	if nprobe > nlist {
		nprobe = nlist
	}

	// Setup PQ
	m := config.M
	if m <= 0 {
		m = config.Dim / 8
		if m < 4 {
			m = 4
		}
	}
	// Ensure divisibility
	for m > 1 && config.Dim%m != 0 {
		m--
	}

	ksub := config.Ksub
	if ksub <= 0 {
		ksub = 256
	}

	pq, err := NewProductQuantizer(config.Dim, m, ksub)
	if err != nil {
		return nil, err
	}

	clusters := make([]*clusterPQData, nlist)
	for i := range clusters {
		clusters[i] = &clusterPQData{}
	}

	return &IVFPQIndex{
		dim:      config.Dim,
		nlist:    nlist,
		nprobe:   nprobe,
		clusters: clusters,
		pq:       pq,
		idMap:    make(map[uint64]ivfpqLoc),
	}, nil
}

// Train trains both the coarse quantizer and PQ codebooks
func (idx *IVFPQIndex) Train(vectors []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors)%idx.dim != 0 {
		return fmt.Errorf("vectors length must be multiple of dimension")
	}

	numVectors := len(vectors) / idx.dim
	if numVectors < idx.nlist {
		idx.nlist = numVectors / 10
		if idx.nlist < 1 {
			idx.nlist = 1
		}
		// Reinitialize clusters
		idx.clusters = make([]*clusterPQData, idx.nlist)
		for i := range idx.clusters {
			idx.clusters[i] = &clusterPQData{}
		}
	}

	// Train coarse quantizer (k-means on full vectors)
	centroids, err := kMeansPlusPlusFlat(vectors, idx.nlist, idx.dim, 25)
	if err != nil {
		return fmt.Errorf("coarse quantizer training failed: %w", err)
	}
	idx.centroids = centroids

	// Train PQ on residuals (vectors - cluster centroids)
	// This is important: PQ on residuals has better recall than PQ on raw vectors
	assignments := idx.assignToClusters(vectors)
	residuals := make([]float32, len(vectors))

	for i := 0; i < numVectors; i++ {
		c := assignments[i]
		for d := 0; d < idx.dim; d++ {
			residuals[i*idx.dim+d] = vectors[i*idx.dim+d] - idx.centroids[c][d]
		}
	}

	if err := idx.pq.Train(residuals, 25); err != nil {
		return fmt.Errorf("PQ training failed: %w", err)
	}

	idx.trained = true
	return nil
}

func (idx *IVFPQIndex) assignToClusters(vectors []float32) []int {
	numVectors := len(vectors) / idx.dim
	assignments := make([]int, numVectors)

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

func (idx *IVFPQIndex) findNearestCluster(vec []float32) int {
	minDist := float32(math.Inf(1))
	minIdx := 0

	for i, c := range idx.centroids {
		var dist float32
		for d := range vec {
			diff := vec[d] - c[d]
			dist += diff * diff
		}
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}

	return minIdx
}

// Add adds a single vector to the index
func (idx *IVFPQIndex) Add(id uint64, vector []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	if len(vector) != idx.dim {
		return fmt.Errorf("vector dimension mismatch")
	}

	// Find nearest cluster
	clusterIdx := idx.findNearestCluster(vector)
	cluster := idx.clusters[clusterIdx]

	// Compute residual
	residual := make([]float32, idx.dim)
	for d := 0; d < idx.dim; d++ {
		residual[d] = vector[d] - idx.centroids[clusterIdx][d]
	}

	// Quantize residual with PQ
	codes, err := idx.pq.Quantize(residual)
	if err != nil {
		return err
	}

	// Add to cluster
	cluster.mu.Lock()
	pos := len(cluster.ids)
	cluster.codes = append(cluster.codes, codes...)
	cluster.ids = append(cluster.ids, id)
	cluster.mu.Unlock()

	idx.idMap[id] = ivfpqLoc{cluster: clusterIdx, pos: pos}
	idx.totalVectors++

	return nil
}

// AddBatch adds multiple vectors efficiently
func (idx *IVFPQIndex) AddBatch(ids []uint64, vectors []float32) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	numVectors := len(vectors) / idx.dim
	if len(ids) != numVectors {
		return fmt.Errorf("ids count mismatch")
	}

	// Assign all vectors to clusters
	assignments := idx.assignToClusters(vectors)

	// Group by cluster
	clusterBatches := make([][]int, idx.nlist)
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

		// Compute residuals and quantize
		residuals := make([]float32, len(batch)*idx.dim)
		for j, vecIdx := range batch {
			for d := 0; d < idx.dim; d++ {
				residuals[j*idx.dim+d] = vectors[vecIdx*idx.dim+d] - idx.centroids[c][d]
			}
		}

		codes, err := idx.pq.Quantize(residuals)
		if err != nil {
			return err
		}

		cluster.mu.Lock()
		for j, vecIdx := range batch {
			pos := len(cluster.ids)
			codeStart := j * idx.pq.m
			cluster.codes = append(cluster.codes, codes[codeStart:codeStart+idx.pq.m]...)
			cluster.ids = append(cluster.ids, ids[vecIdx])
			idx.idMap[ids[vecIdx]] = ivfpqLoc{cluster: c, pos: pos}
		}
		cluster.mu.Unlock()
	}

	idx.totalVectors += numVectors
	return nil
}

// Search finds top-k nearest neighbors using IVF + PQ + ADC
func (idx *IVFPQIndex) Search(query []float32, k int) ([]PQSearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.trained {
		return nil, fmt.Errorf("index must be trained before searching")
	}

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	// Find nprobe nearest clusters
	clusterDists := make([]float32, idx.nlist)
	for i, c := range idx.centroids {
		var dist float32
		for d := range query {
			diff := query[d] - c[d]
			dist += diff * diff
		}
		clusterDists[i] = dist
	}

	// Select top nprobe clusters
	type clusterIdx struct {
		idx  int
		dist float32
	}
	clusters := make([]clusterIdx, idx.nlist)
	for i, d := range clusterDists {
		clusters[i] = clusterIdx{idx: i, dist: d}
	}
	sort.Slice(clusters, func(i, j int) bool {
		return clusters[i].dist < clusters[j].dist
	})

	nprobe := idx.nprobe
	if nprobe > idx.nlist {
		nprobe = idx.nlist
	}

	// Search each cluster in parallel
	type candidate struct {
		id   uint64
		dist float32
	}
	var allCandidates []candidate
	var candidatesMu sync.Mutex
	var wg sync.WaitGroup

	for i := 0; i < nprobe; i++ {
		wg.Add(1)
		go func(clusterIdx int) {
			defer wg.Done()

			cluster := idx.clusters[clusterIdx]
			cluster.mu.RLock()
			defer cluster.mu.RUnlock()

			if len(cluster.ids) == 0 {
				return
			}

			// Compute residual query for this cluster
			residualQuery := make([]float32, idx.dim)
			for d := 0; d < idx.dim; d++ {
				residualQuery[d] = query[d] - idx.centroids[clusterIdx][d]
			}

			// Compute distance table for residual query
			dt, err := idx.pq.ComputeDistanceTable(residualQuery)
			if err != nil {
				return
			}

			// Lookup distances to all vectors in cluster
			numInCluster := len(cluster.ids)
			distances := dt.LookupDistanceBatch(cluster.codes, numInCluster)

			// Collect candidates
			candidates := make([]candidate, numInCluster)
			for j := 0; j < numInCluster; j++ {
				candidates[j] = candidate{
					id:   cluster.ids[j],
					dist: distances[j],
				}
			}

			candidatesMu.Lock()
			allCandidates = append(allCandidates, candidates...)
			candidatesMu.Unlock()
		}(clusters[i].idx)
	}

	wg.Wait()

	// Sort and return top-k
	sort.Slice(allCandidates, func(i, j int) bool {
		return allCandidates[i].dist < allCandidates[j].dist
	})

	if k > len(allCandidates) {
		k = len(allCandidates)
	}

	results := make([]PQSearchResult, k)
	for i := 0; i < k; i++ {
		results[i] = PQSearchResult{
			ID:       allCandidates[i].id,
			Distance: allCandidates[i].dist,
		}
	}

	return results, nil
}

// Size returns number of vectors in index
func (idx *IVFPQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.totalVectors
}

// Stats returns index statistics
func (idx *IVFPQIndex) Stats() map[string]interface{} {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	clusterSizes := make([]int, idx.nlist)
	for i, c := range idx.clusters {
		c.mu.RLock()
		clusterSizes[i] = len(c.ids)
		c.mu.RUnlock()
	}

	return map[string]interface{}{
		"type":              "ivf_pq_adc",
		"dimension":         idx.dim,
		"nlist":             idx.nlist,
		"nprobe":            idx.nprobe,
		"pq_subvectors":     idx.pq.m,
		"pq_centroids":      idx.pq.ksub,
		"total_vectors":     idx.totalVectors,
		"trained":           idx.trained,
		"cluster_sizes":     clusterSizes,
	}
}

// Register with factory
func init() {
	Register("pq", func(dim int, config map[string]interface{}) (Index, error) {
		m := GetConfigInt(config, "m", dim/8)
		ksub := GetConfigInt(config, "ksub", 256)
		
		idx, err := NewPQIndex(PQIndexConfig{
			Dim:  dim,
			M:    m,
			Ksub: ksub,
		})
		if err != nil {
			return nil, err
		}
		
		return &pqIndexWrapper{idx: idx, dim: dim}, nil
	})

	Register("ivf_pq", func(dim int, config map[string]interface{}) (Index, error) {
		nlist := GetConfigInt(config, "nlist", 100)
		nprobe := GetConfigInt(config, "nprobe", 10)
		m := GetConfigInt(config, "m", dim/8)
		ksub := GetConfigInt(config, "ksub", 256)
		
		idx, err := NewIVFPQIndex(IVFPQConfig{
			Dim:    dim,
			Nlist:  nlist,
			Nprobe: nprobe,
			M:      m,
			Ksub:   ksub,
		})
		if err != nil {
			return nil, err
		}
		
		return &ivfpqIndexWrapper{idx: idx, dim: dim}, nil
	})
}

// Index interface wrappers
type pqIndexWrapper struct {
	idx *PQIndex
	dim int
}

func (w *pqIndexWrapper) Name() string { return "PQ-ADC" }

func (w *pqIndexWrapper) Add(ctx context.Context, id uint64, vector []float32) error {
	return w.idx.Add(id, vector)
}

func (w *pqIndexWrapper) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	results, err := w.idx.SearchWithHeap(query, k)
	if err != nil {
		return nil, err
	}
	
	out := make([]Result, len(results))
	for i, r := range results {
		out[i] = Result{ID: r.ID, Distance: r.Distance, Score: 1 / (1 + r.Distance)}
	}
	return out, nil
}

func (w *pqIndexWrapper) Delete(ctx context.Context, id uint64) error {
	return fmt.Errorf("delete not supported for PQ index")
}

func (w *pqIndexWrapper) Stats() IndexStats {
	stats := w.idx.Stats()
	return IndexStats{
		Name:       "PQ-ADC",
		Dim:        w.dim,
		Count:      w.idx.Size(),
		MemoryUsed: w.idx.MemoryUsage(),
		Extra:      stats,
	}
}

func (w *pqIndexWrapper) Export() ([]byte, error) {
	w.idx.mu.RLock()
	defer w.idx.mu.RUnlock()

	return exportPQData(w.idx.dim, w.idx.pq.m, w.idx.pq.ksub, w.idx.pq.dsub, w.idx.pq.trained,
		w.idx.pq.codebooks, w.idx.codes, w.idx.ids)
}

func (w *pqIndexWrapper) Import(data []byte) error {
	w.idx.mu.Lock()
	defer w.idx.mu.Unlock()

	dim, m, ksub, dsub, trained, codebooks, codes, ids, err := importPQData(data)
	if err != nil {
		return fmt.Errorf("pq import: %w", err)
	}

	w.idx.dim = dim
	w.idx.pq.dim = dim
	w.idx.pq.m = m
	w.idx.pq.ksub = ksub
	w.idx.pq.dsub = dsub
	w.idx.pq.trained = trained
	w.idx.pq.codebooks = codebooks
	w.idx.codes = codes
	w.idx.ids = ids
	w.dim = dim
	return nil
}

type ivfpqIndexWrapper struct {
	idx *IVFPQIndex
	dim int
}

func (w *ivfpqIndexWrapper) Name() string { return "IVF-PQ-ADC" }

func (w *ivfpqIndexWrapper) Add(ctx context.Context, id uint64, vector []float32) error {
	return w.idx.Add(id, vector)
}

func (w *ivfpqIndexWrapper) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	results, err := w.idx.Search(query, k)
	if err != nil {
		return nil, err
	}
	
	out := make([]Result, len(results))
	for i, r := range results {
		out[i] = Result{ID: r.ID, Distance: r.Distance, Score: 1 / (1 + r.Distance)}
	}
	return out, nil
}

func (w *ivfpqIndexWrapper) Delete(ctx context.Context, id uint64) error {
	return fmt.Errorf("delete not supported for IVF-PQ index")
}

func (w *ivfpqIndexWrapper) Stats() IndexStats {
	stats := w.idx.Stats()
	return IndexStats{
		Name:       "IVF-PQ-ADC",
		Dim:        w.dim,
		Count:      w.idx.Size(),
		Extra:      stats,
	}
}

func (w *ivfpqIndexWrapper) Export() ([]byte, error) {
	return nil, fmt.Errorf("export not implemented for IVF-PQ: use PQ or PQ4 index for persistence")
}

func (w *ivfpqIndexWrapper) Import(data []byte) error {
	return fmt.Errorf("import not implemented for IVF-PQ: use PQ or PQ4 index for persistence")
}
