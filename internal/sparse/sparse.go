package sparse

import (
	"fmt"
	"math"
	"sort"
)

// SparseVector represents a sparse vector in Coordinate (CoO) format.
// This is the standard format for BM25-style sparse vectors used in RAG systems.
//
// Sparse vectors are highly compressed: a typical 10K dimension vector with
// 300 non-zero values uses only 2.4KB vs 40KB for dense representation.
type SparseVector struct {
	Indices []uint32  // Non-zero indices (sorted ascending)
	Values  []float32 // Non-zero values (corresponding to Indices)
	Dim     int       // Total dimension (max possible index + 1)
}

// NewSparseVector creates a new sparse vector from indices and values.
// Automatically sorts by indices and validates input.
func NewSparseVector(indices []uint32, values []float32, dim int) (*SparseVector, error) {
	if len(indices) != len(values) {
		return nil, fmt.Errorf("indices and values length mismatch: %d vs %d", len(indices), len(values))
	}

	if len(indices) == 0 {
		return &SparseVector{
			Indices: []uint32{},
			Values:  []float32{},
			Dim:     dim,
		}, nil
	}

	// Validate and find max index
	maxIdx := uint32(0)
	for i, idx := range indices {
		if values[i] == 0 {
			return nil, fmt.Errorf("zero value at index %d (sparse vectors should only contain non-zeros)", idx)
		}
		if idx >= uint32(dim) {
			return nil, fmt.Errorf("index %d exceeds dimension %d", idx, dim)
		}
		if idx > maxIdx {
			maxIdx = idx
		}
	}

	// Copy and sort by indices
	sortedIndices := make([]uint32, len(indices))
	sortedValues := make([]float32, len(values))
	copy(sortedIndices, indices)
	copy(sortedValues, values)

	// Create index-value pairs for sorting
	type pair struct {
		idx uint32
		val float32
	}
	pairs := make([]pair, len(sortedIndices))
	for i := range sortedIndices {
		pairs[i] = pair{sortedIndices[i], sortedValues[i]}
	}

	// Sort by index
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].idx < pairs[j].idx
	})

	// Extract sorted arrays
	for i, p := range pairs {
		sortedIndices[i] = p.idx
		sortedValues[i] = p.val
	}

	// Check for duplicate indices
	for i := 1; i < len(sortedIndices); i++ {
		if sortedIndices[i] == sortedIndices[i-1] {
			return nil, fmt.Errorf("duplicate index %d", sortedIndices[i])
		}
	}

	return &SparseVector{
		Indices: sortedIndices,
		Values:  sortedValues,
		Dim:     dim,
	}, nil
}

// FromDense converts a dense vector to sparse by keeping non-zero values.
func FromDense(dense []float32, threshold float32) *SparseVector {
	if threshold == 0 {
		threshold = 1e-9 // Avoid exact zero comparisons
	}

	// Count non-zeros
	nnz := 0
	for _, v := range dense {
		if math.Abs(float64(v)) > float64(threshold) {
			nnz++
		}
	}

	indices := make([]uint32, 0, nnz)
	values := make([]float32, 0, nnz)

	for i, v := range dense {
		if math.Abs(float64(v)) > float64(threshold) {
			indices = append(indices, uint32(i))
			values = append(values, v)
		}
	}

	return &SparseVector{
		Indices: indices,
		Values:  values,
		Dim:     len(dense),
	}
}

// ToDense converts sparse vector to dense representation.
func (sv *SparseVector) ToDense() []float32 {
	dense := make([]float32, sv.Dim)
	for i, idx := range sv.Indices {
		dense[idx] = sv.Values[i]
	}
	return dense
}

// Nnz returns the number of non-zero values.
func (sv *SparseVector) Nnz() int {
	return len(sv.Indices)
}

// Sparsity returns the sparsity ratio (0.0 = dense, 1.0 = all zeros).
func (sv *SparseVector) Sparsity() float64 {
	if sv.Dim == 0 {
		return 0
	}
	return 1.0 - float64(sv.Nnz())/float64(sv.Dim)
}

// Norm computes the L2 norm of the sparse vector.
func (sv *SparseVector) Norm() float32 {
	sum := float32(0)
	for _, v := range sv.Values {
		sum += v * v
	}
	return float32(math.Sqrt(float64(sum)))
}

// Normalize normalizes the sparse vector in-place to unit L2 norm.
func (sv *SparseVector) Normalize() {
	norm := sv.Norm()
	if norm > 0 {
		for i := range sv.Values {
			sv.Values[i] /= norm
		}
	}
}

// DotProduct computes the dot product between two sparse vectors.
// Efficiently handles sparse-sparse multiplication using merge algorithm.
func DotProduct(a, b *SparseVector) float32 {
	if a.Dim != b.Dim {
		return 0 // Dimension mismatch
	}

	result := float32(0)
	i, j := 0, 0

	// Merge algorithm - O(nnz_a + nnz_b)
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

// CosineSimilarity computes cosine similarity between two sparse vectors.
func CosineSimilarity(a, b *SparseVector) float32 {
	dot := DotProduct(a, b)
	normA := a.Norm()
	normB := b.Norm()

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (normA * normB)
}

// Add performs element-wise addition of two sparse vectors.
// Returns a new sparse vector (does not modify inputs).
func Add(a, b *SparseVector) (*SparseVector, error) {
	if a.Dim != b.Dim {
		return nil, fmt.Errorf("dimension mismatch: %d vs %d", a.Dim, b.Dim)
	}

	// Merge with addition
	indices := make([]uint32, 0, len(a.Indices)+len(b.Indices))
	values := make([]float32, 0, len(a.Values)+len(b.Values))

	i, j := 0, 0
	for i < len(a.Indices) || j < len(b.Indices) {
		if i >= len(a.Indices) {
			// Only b left
			indices = append(indices, b.Indices[j])
			values = append(values, b.Values[j])
			j++
		} else if j >= len(b.Indices) {
			// Only a left
			indices = append(indices, a.Indices[i])
			values = append(values, a.Values[i])
			i++
		} else if a.Indices[i] == b.Indices[j] {
			// Both have this index
			sum := a.Values[i] + b.Values[j]
			if math.Abs(float64(sum)) > 1e-9 {
				indices = append(indices, a.Indices[i])
				values = append(values, sum)
			}
			i++
			j++
		} else if a.Indices[i] < b.Indices[j] {
			indices = append(indices, a.Indices[i])
			values = append(values, a.Values[i])
			i++
		} else {
			indices = append(indices, b.Indices[j])
			values = append(values, b.Values[j])
			j++
		}
	}

	return &SparseVector{
		Indices: indices,
		Values:  values,
		Dim:     a.Dim,
	}, nil
}

// Scale multiplies the sparse vector by a scalar.
func (sv *SparseVector) Scale(scalar float32) {
	for i := range sv.Values {
		sv.Values[i] *= scalar
	}
}

// Clone creates a deep copy of the sparse vector.
func (sv *SparseVector) Clone() *SparseVector {
	indices := make([]uint32, len(sv.Indices))
	values := make([]float32, len(sv.Values))
	copy(indices, sv.Indices)
	copy(values, sv.Values)

	return &SparseVector{
		Indices: indices,
		Values:  values,
		Dim:     sv.Dim,
	}
}

// String returns a human-readable representation.
func (sv *SparseVector) String() string {
	return fmt.Sprintf("SparseVector{dim=%d, nnz=%d, sparsity=%.2f%%}",
		sv.Dim, sv.Nnz(), sv.Sparsity()*100)
}
