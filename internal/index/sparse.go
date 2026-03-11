package index

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
)

// SparseCoO represents a sparse vector in Coordinate format
// This is duplicated here to avoid circular imports
type SparseCoO struct {
	Indices []uint32
	Values  []float32
	Dim     int
}

// SparseIndex implements inverted index for sparse vectors
// Optimized for high-dimensional sparse data (e.g., text embeddings, SPLADE)
type SparseIndex struct {
	mu sync.RWMutex

	dim      int                      // Vector dimension
	vectors  map[uint64]*SparseCoO    // ID -> sparse vector
	inverted map[uint32][]uint64      // Dimension index -> list of vector IDs
	norms    map[uint64]float32       // Pre-computed norms for cosine similarity
	count    int                      // Total vectors added
	deleted  map[uint64]bool          // Tombstone deletions

	distanceMetric string // "cosine", "dot", "jaccard"
}

// NewSparseIndex creates a new sparse vector index
func NewSparseIndex(dim int, config map[string]interface{}) (Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", dim)
	}

	metric := "cosine" // default
	if m, ok := config["metric"].(string); ok {
		metric = m
	}

	return &SparseIndex{
		dim:            dim,
		vectors:        make(map[uint64]*SparseCoO),
		inverted:       make(map[uint32][]uint64),
		norms:          make(map[uint64]float32),
		deleted:        make(map[uint64]bool),
		distanceMetric: metric,
	}, nil
}

func init() {
	// Register sparse index type
	Register("sparse", NewSparseIndex)
}

// Name returns the index type name
func (s *SparseIndex) Name() string {
	return "Sparse"
}

// Add adds a vector to the sparse index
// Note: This expects dense float32 input and converts to sparse internally
func (s *SparseIndex) Add(ctx context.Context, id uint64, vector []float32) error {
	if len(vector) != s.dim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dim, len(vector))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Check for re-add: clean up old inverted index entries first
	isReAdd := false
	if oldSparse, exists := s.vectors[id]; exists {
		isReAdd = true
		for _, idx := range oldSparse.Indices {
			if ids, ok := s.inverted[idx]; ok {
				for j, pid := range ids {
					if pid == id {
						s.inverted[idx] = append(ids[:j], ids[j+1:]...)
						break
					}
				}
			}
		}
	}

	// Convert dense to sparse (only store non-zero elements)
	sparse := denseToSparse(vector)

	// Store vector
	s.vectors[id] = sparse

	// Update inverted index
	for i, idx := range sparse.Indices {
		if s.inverted[idx] == nil {
			s.inverted[idx] = make([]uint64, 0, 1)
		}
		s.inverted[idx] = append(s.inverted[idx], id)

		// Maintain sorted order for efficient search
		sort.Slice(s.inverted[idx], func(a, b int) bool {
			return s.inverted[idx][a] < s.inverted[idx][b]
		})

		// Pre-compute norm for cosine similarity
		if i == 0 {
			s.norms[id] = 0
		}
		s.norms[id] += sparse.Values[i] * sparse.Values[i]
	}

	if len(sparse.Indices) > 0 {
		s.norms[id] = float32(math.Sqrt(float64(s.norms[id])))
	}

	delete(s.deleted, id)
	if !isReAdd {
		s.count++
	}

	return nil
}

// Search performs sparse vector search using inverted index
func (s *SparseIndex) Search(ctx context.Context, query []float32, k int, params SearchParams) ([]Result, error) {
	if len(query) != s.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", s.dim, len(query))
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Convert query to sparse
	queryVec := denseToSparse(query)

	if len(queryVec.Indices) == 0 {
		return []Result{}, nil
	}

	// Find candidate vectors using inverted index
	// Only consider vectors that share at least one dimension with query
	candidates := make(map[uint64]bool)
	for _, idx := range queryVec.Indices {
		if ids, ok := s.inverted[idx]; ok {
			for _, id := range ids {
				if !s.deleted[id] {
					candidates[id] = true
				}
			}
		}
	}

	if len(candidates) == 0 {
		return []Result{}, nil
	}

	// Compute distances for candidates
	results := make([]Result, 0, len(candidates))

	for id := range candidates {
		vec := s.vectors[id]
		var distance float32

		switch s.distanceMetric {
		case "cosine":
			distance = sparseCosineDistance(queryVec, vec, s.norms[id])
		case "dot":
			// Dot product (negative for similarity -> distance)
			distance = -sparseDotProduct(queryVec, vec)
		case "jaccard":
			distance = 1.0 - sparseJaccardSimilarity(queryVec, vec)
		default:
			distance = sparseCosineDistance(queryVec, vec, s.norms[id])
		}

		results = append(results, Result{
			ID:       id,
			Distance: distance,
			Score:    1.0 / (1.0 + distance),
		})
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
func (s *SparseIndex) Delete(ctx context.Context, id uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.vectors[id]; !exists {
		return fmt.Errorf("vector %d not found", id)
	}

	s.deleted[id] = true
	return nil
}

// Stats returns index statistics
func (s *SparseIndex) Stats() IndexStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	deletedCount := len(s.deleted)
	avgSparsity := s.calculateAvgSparsity()

	return IndexStats{
		Name:       "Sparse",
		Dim:        s.dim,
		Count:      s.count,
		Deleted:    deletedCount,
		Active:     s.count - deletedCount,
		MemoryUsed: int64(s.estimateMemory()),
		DiskUsed:   0,
		Extra: map[string]interface{}{
			"metric":         s.distanceMetric,
			"avg_sparsity":   avgSparsity,
			"inverted_terms": len(s.inverted),
		},
	}
}

// Export serializes the sparse index
func (s *SparseIndex) Export() ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Export format:
	// [4 bytes: dim]
	// [4 bytes: count]
	// [4 bytes: metric length] [metric string]
	// [4 bytes: num vectors]
	// For each vector:
	//   [8 bytes: id] [4 bytes: nnz] [indices...] [values...]

	buf := make([]byte, 0, s.estimateMemory())

	// Dimension
	dimBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(dimBytes, uint32(s.dim))
	buf = append(buf, dimBytes...)

	// Count
	countBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(countBytes, uint32(s.count))
	buf = append(buf, countBytes...)

	// Metric
	metricBytes := []byte(s.distanceMetric)
	metricLen := make([]byte, 4)
	binary.LittleEndian.PutUint32(metricLen, uint32(len(metricBytes)))
	buf = append(buf, metricLen...)
	buf = append(buf, metricBytes...)

	// Number of vectors
	numVecs := make([]byte, 4)
	binary.LittleEndian.PutUint32(numVecs, uint32(len(s.vectors)))
	buf = append(buf, numVecs...)

	// Vectors
	for id, vec := range s.vectors {
		// ID
		idBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(idBytes, id)
		buf = append(buf, idBytes...)

		// NNZ
		nnz := len(vec.Indices)
		nnzBytes := make([]byte, 4)
		binary.LittleEndian.PutUint32(nnzBytes, uint32(nnz))
		buf = append(buf, nnzBytes...)

		// Indices
		for _, idx := range vec.Indices {
			idxBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(idxBytes, idx)
			buf = append(buf, idxBytes...)
		}

		// Values
		for _, val := range vec.Values {
			valBytes := make([]byte, 4)
			binary.LittleEndian.PutUint32(valBytes, math.Float32bits(val))
			buf = append(buf, valBytes...)
		}

		// Deleted flag
		if s.deleted[id] {
			buf = append(buf, 1)
		} else {
			buf = append(buf, 0)
		}
	}

	return buf, nil
}

// Import deserializes the sparse index
func (s *SparseIndex) Import(data []byte) error {
	if len(data) < 16 {
		return fmt.Errorf("invalid sparse index data: too short")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	offset := 0

	// Dimension
	s.dim = int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	// Count
	s.count = int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	// Metric
	metricLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4
	s.distanceMetric = string(data[offset : offset+metricLen])
	offset += metricLen

	// Number of vectors
	numVecs := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	// Initialize maps
	s.vectors = make(map[uint64]*SparseCoO, numVecs)
	s.inverted = make(map[uint32][]uint64)
	s.norms = make(map[uint64]float32, numVecs)
	s.deleted = make(map[uint64]bool)

	// Read vectors
	for i := 0; i < numVecs; i++ {
		if offset+12 > len(data) {
			return fmt.Errorf("incomplete sparse vector data")
		}

		// ID
		id := binary.LittleEndian.Uint64(data[offset : offset+8])
		offset += 8

		// NNZ
		nnz := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		// Indices
		indices := make([]uint32, nnz)
		for j := 0; j < nnz; j++ {
			indices[j] = binary.LittleEndian.Uint32(data[offset : offset+4])
			offset += 4
		}

		// Values
		values := make([]float32, nnz)
		var norm float32
		for j := 0; j < nnz; j++ {
			bits := binary.LittleEndian.Uint32(data[offset : offset+4])
			values[j] = math.Float32frombits(bits)
			norm += values[j] * values[j]
			offset += 4
		}

		// Store vector
		vec := &SparseCoO{
			Indices: indices,
			Values:  values,
			Dim:     s.dim,
		}
		s.vectors[id] = vec
		s.norms[id] = float32(math.Sqrt(float64(norm)))

		// Rebuild inverted index
		for _, idx := range indices {
			if s.inverted[idx] == nil {
				s.inverted[idx] = make([]uint64, 0)
			}
			s.inverted[idx] = append(s.inverted[idx], id)
		}

		// Deleted flag
		if offset < len(data) {
			if data[offset] == 1 {
				s.deleted[id] = true
			}
			offset++
		}
	}

	return nil
}

// Helper functions

func denseToSparse(dense []float32) *SparseCoO {
	const threshold = 1e-6

	indices := make([]uint32, 0)
	values := make([]float32, 0)

	for i, v := range dense {
		if math.Abs(float64(v)) > threshold {
			indices = append(indices, uint32(i))
			values = append(values, v)
		}
	}

	return &SparseCoO{
		Indices: indices,
		Values:  values,
		Dim:     len(dense),
	}
}

func sparseDotProduct(a, b *SparseCoO) float32 {
	var result float32
	i, j := 0, 0

	for i < len(a.Indices) && j < len(b.Indices) {
		if a.Indices[i] == b.Indices[j] {
			result += a.Values[i] * b.Values[j]
			i++
			j++
		} else if a.Indices[i] < b.Indices[j] {
			i++
		} else {
			j++
		}
	}

	return result
}

func sparseCosineDistance(a, b *SparseCoO, normB float32) float32 {
	dot := sparseDotProduct(a, b)
	if dot == 0 {
		return 1.0
	}

	// Compute norm of a
	var normA float32
	for _, v := range a.Values {
		normA += v * v
	}
	normA = float32(math.Sqrt(float64(normA)))

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (normA * normB)
	return 1.0 - similarity
}

func sparseJaccardSimilarity(a, b *SparseCoO) float32 {
	if len(a.Indices) == 0 && len(b.Indices) == 0 {
		return 1.0
	}
	if len(a.Indices) == 0 || len(b.Indices) == 0 {
		return 0
	}

	intersection := 0
	i, j := 0, 0

	for i < len(a.Indices) && j < len(b.Indices) {
		if a.Indices[i] == b.Indices[j] {
			intersection++
			i++
			j++
		} else if a.Indices[i] < b.Indices[j] {
			i++
		} else {
			j++
		}
	}

	union := len(a.Indices) + len(b.Indices) - intersection
	if union == 0 {
		return 1.0
	}

	return float32(intersection) / float32(union)
}

func (s *SparseIndex) calculateAvgSparsity() float64 {
	if len(s.vectors) == 0 {
		return 0
	}

	totalNNZ := 0
	for _, vec := range s.vectors {
		totalNNZ += len(vec.Indices)
	}

	avgNNZ := float64(totalNNZ) / float64(len(s.vectors))
	return 1.0 - (avgNNZ / float64(s.dim))
}

func (s *SparseIndex) estimateMemory() int {
	// Vectors: ID (8) + pointer overhead + sparse data
	vectorMem := 0
	for _, vec := range s.vectors {
		vectorMem += 8 + len(vec.Indices)*4 + len(vec.Values)*4
	}

	// Inverted index: key (4) + slice of IDs
	invertedMem := 0
	for _, ids := range s.inverted {
		invertedMem += 4 + len(ids)*8
	}

	// Norms: ID (8) + float32 (4)
	normsMem := len(s.norms) * 12

	// Deleted: ID (8) + bool (1)
	deletedMem := len(s.deleted) * 9

	return vectorMem + invertedMem + normsMem + deletedMem
}

// ExportJSON exports index metadata as JSON (for debugging)
func (s *SparseIndex) ExportJSON() ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	export := map[string]interface{}{
		"type":   "sparse",
		"dim":    s.dim,
		"count":  s.count,
		"metric": s.distanceMetric,
		"stats":  s.Stats(),
	}

	return json.Marshal(export)
}
