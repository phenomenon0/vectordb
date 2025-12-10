package main

import (
	"math"
)

// SparseDistanceFunc represents a distance function for sparse vectors
type SparseDistanceFunc func(a, b *SparseCoO) float32

// SparseDotProduct computes dot product between two sparse vectors
// Used for cosine similarity: sim = dot(a, b) / (norm(a) * norm(b))
func SparseDotProduct(a, b *SparseCoO) float32 {
	if len(a.Indices) == 0 || len(b.Indices) == 0 {
		return 0
	}

	var result float32
	i, j := 0, 0

	// Two-pointer algorithm for sorted indices
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

// SparseCosineSimilarity computes cosine similarity between two sparse vectors
// Returns value in [0, 1] where 1 is identical, 0 is orthogonal
func SparseCosineSimilarity(a, b *SparseCoO) float32 {
	dot := SparseDotProduct(a, b)
	if dot == 0 {
		return 0
	}

	normA := SparseNorm(a)
	normB := SparseNorm(b)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (normA * normB)
}

// SparseCosineDistance computes cosine distance (1 - similarity)
// Returns value in [0, 1] where 0 is identical, 1 is orthogonal
func SparseCosineDistance(a, b *SparseCoO) float32 {
	return 1.0 - SparseCosineSimilarity(a, b)
}

// SparseEuclideanDistance computes L2 (Euclidean) distance between sparse vectors
// Efficient implementation that only considers non-zero elements
func SparseEuclideanDistance(a, b *SparseCoO) float32 {
	if a.Dim != b.Dim {
		return float32(math.Inf(1))
	}

	var sumSq float32
	i, j := 0, 0

	// Handle elements present in both vectors
	for i < len(a.Indices) && j < len(b.Indices) {
		if a.Indices[i] == b.Indices[j] {
			diff := a.Values[i] - b.Values[j]
			sumSq += diff * diff
			i++
			j++
		} else if a.Indices[i] < b.Indices[j] {
			// Element in a but not b (b[idx] = 0)
			sumSq += a.Values[i] * a.Values[i]
			i++
		} else {
			// Element in b but not a (a[idx] = 0)
			sumSq += b.Values[j] * b.Values[j]
			j++
		}
	}

	// Handle remaining elements in a
	for i < len(a.Indices) {
		sumSq += a.Values[i] * a.Values[i]
		i++
	}

	// Handle remaining elements in b
	for j < len(b.Indices) {
		sumSq += b.Values[j] * b.Values[j]
		j++
	}

	return float32(math.Sqrt(float64(sumSq)))
}

// SparseNorm computes the L2 norm of a sparse vector
func SparseNorm(a *SparseCoO) float32 {
	var sumSq float32
	for _, v := range a.Values {
		sumSq += v * v
	}
	return float32(math.Sqrt(float64(sumSq)))
}

// SparseJaccardSimilarity computes Jaccard similarity between two sparse vectors
// Treats vectors as sets of indices (ignoring values)
// Returns value in [0, 1] where 1 is identical sets
func SparseJaccardSimilarity(a, b *SparseCoO) float32 {
	if len(a.Indices) == 0 && len(b.Indices) == 0 {
		return 1.0
	}
	if len(a.Indices) == 0 || len(b.Indices) == 0 {
		return 0
	}

	intersection := 0
	i, j := 0, 0

	// Count intersection
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

// SparseBM25Similarity computes BM25-like similarity for sparse vectors
// Commonly used for text retrieval with term frequency vectors
// Parameters: k1 (term saturation, typical: 1.2-2.0), b (length normalization, typical: 0.75)
func SparseBM25Similarity(query, doc *SparseCoO, avgDocLength float32, k1, b float32) float32 {
	if len(query.Indices) == 0 {
		return 0
	}

	docLength := float32(doc.NonZeroCount())
	if docLength == 0 {
		return 0
	}

	var score float32
	i, j := 0, 0

	// For each query term
	for i < len(query.Indices) {
		// Find matching term in document
		for j < len(doc.Indices) && doc.Indices[j] < query.Indices[i] {
			j++
		}

		if j < len(doc.Indices) && doc.Indices[j] == query.Indices[i] {
			// Term found in document
			queryTF := query.Values[i]
			docTF := doc.Values[j]

			// BM25 term score
			numerator := docTF * (k1 + 1)
			denominator := docTF + k1*(1-b+b*(docLength/avgDocLength))
			termScore := queryTF * (numerator / denominator)

			score += termScore
			j++
		}

		i++
	}

	return score
}

// DenseToSparseCoO converts a dense float32 vector to sparse CoO format
// Only stores values above the given threshold (default: 1e-6)
func DenseToSparseCoO(dense []float32, threshold float32) *SparseCoO {
	if threshold == 0 {
		threshold = 1e-6
	}

	indices := make([]uint32, 0)
	values := make([]float32, 0)

	for i, v := range dense {
		if math.Abs(float64(v)) > float64(threshold) {
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

// SparseAdd performs element-wise addition of two sparse vectors
// Returns a new sparse vector (does not modify inputs)
func SparseAdd(a, b *SparseCoO) *SparseCoO {
	if a.Dim != b.Dim {
		return nil
	}

	indices := make([]uint32, 0)
	values := make([]float32, 0)

	i, j := 0, 0

	for i < len(a.Indices) && j < len(b.Indices) {
		if a.Indices[i] == b.Indices[j] {
			sum := a.Values[i] + b.Values[j]
			if math.Abs(float64(sum)) > 1e-6 {
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

	// Remaining elements from a
	for i < len(a.Indices) {
		indices = append(indices, a.Indices[i])
		values = append(values, a.Values[i])
		i++
	}

	// Remaining elements from b
	for j < len(b.Indices) {
		indices = append(indices, b.Indices[j])
		values = append(values, b.Values[j])
		j++
	}

	return &SparseCoO{
		Indices: indices,
		Values:  values,
		Dim:     a.Dim,
	}
}

// SparseScale multiplies a sparse vector by a scalar
// Returns a new sparse vector (does not modify input)
func SparseScale(a *SparseCoO, scalar float32) *SparseCoO {
	values := make([]float32, len(a.Values))
	for i, v := range a.Values {
		values[i] = v * scalar
	}

	return &SparseCoO{
		Indices: a.Indices, // Can reuse indices (immutable)
		Values:  values,
		Dim:     a.Dim,
	}
}

// SparseNormalize normalizes a sparse vector to unit L2 norm
// Returns a new sparse vector (does not modify input)
func SparseNormalize(a *SparseCoO) *SparseCoO {
	norm := SparseNorm(a)
	if norm == 0 {
		return &SparseCoO{
			Indices: a.Indices,
			Values:  make([]float32, len(a.Values)),
			Dim:     a.Dim,
		}
	}
	return SparseScale(a, 1.0/norm)
}
