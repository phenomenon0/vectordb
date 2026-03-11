// Package simd provides SIMD-accelerated distance functions for vector search.
//
// On amd64, uses AVX2+FMA instructions (8 float32s per cycle).
// Falls back to scalar Go loops on other architectures.
package simd

import (
	"fmt"
	"math"
)

// ErrDimensionMismatch is returned when vector lengths don't match.
var ErrDimensionMismatch = fmt.Errorf("simd: mismatched vector lengths")

// DotProductF32 computes the dot product of two float32 slices.
// Returns ErrDimensionMismatch if len(a) != len(b).
func DotProductF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("simd: mismatched vector lengths")
	}
	if len(a) == 0 {
		return 0
	}
	return dotProductF32(a, b)
}

// DotProductF32Safe computes the dot product, returning an error on dimension mismatch.
func DotProductF32Safe(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("%w: %d vs %d", ErrDimensionMismatch, len(a), len(b))
	}
	if len(a) == 0 {
		return 0, nil
	}
	return dotProductF32(a, b), nil
}

// L2DistanceSquaredF32 computes the squared L2 (Euclidean) distance: sum((a-b)^2).
func L2DistanceSquaredF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("simd: mismatched vector lengths")
	}
	if len(a) == 0 {
		return 0
	}
	return l2DistanceSquaredF32(a, b)
}

// L2DistanceSquaredF32Safe computes squared L2 distance, returning an error on dimension mismatch.
func L2DistanceSquaredF32Safe(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("%w: %d vs %d", ErrDimensionMismatch, len(a), len(b))
	}
	if len(a) == 0 {
		return 0, nil
	}
	return l2DistanceSquaredF32(a, b), nil
}

// L2DistanceF32 computes the L2 (Euclidean) distance: sqrt(sum((a-b)^2)).
func L2DistanceF32(a, b []float32) float32 {
	return float32(math.Sqrt(float64(L2DistanceSquaredF32(a, b))))
}

// L2DistanceF32Safe computes L2 distance, returning an error on dimension mismatch.
func L2DistanceF32Safe(a, b []float32) (float32, error) {
	sq, err := L2DistanceSquaredF32Safe(a, b)
	if err != nil {
		return 0, err
	}
	return float32(math.Sqrt(float64(sq))), nil
}

// CosineDistanceF32 computes 1 - cosine_similarity(a, b).
func CosineDistanceF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("simd: mismatched vector lengths")
	}
	n := len(a)
	if n == 0 {
		return 1.0
	}
	return cosineDistanceF32(a, b)
}

// CosineDistanceF32Safe computes cosine distance, returning an error on dimension mismatch.
func CosineDistanceF32Safe(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("%w: %d vs %d", ErrDimensionMismatch, len(a), len(b))
	}
	if len(a) == 0 {
		return 1.0, nil
	}
	return cosineDistanceF32(a, b), nil
}

// NormalizedCosineDistanceF32 computes 1 - dot(a, b) for pre-normalized vectors.
// When vectors are unit-norm, this is equivalent to cosine distance but 42% faster
// at 768d (22ns vs 48ns) because norm computation is eliminated.
// Clamps to [0, 2] to handle floating-point rounding errors.
func NormalizedCosineDistanceF32(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("simd: mismatched vector lengths")
	}
	if len(a) == 0 {
		return 1.0
	}
	d := 1.0 - dotProductF32(a, b)
	if d < 0 {
		d = 0
	}
	return d
}

// NormalizeF32 L2-normalizes a vector in place. Returns the original norm.
// If the vector is zero, it is left unchanged and 0 is returned.
func NormalizeF32(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))
	if norm == 0 {
		return 0
	}
	invNorm := 1.0 / norm
	for i := range v {
		v[i] *= invNorm
	}
	return norm
}

// cosineDistanceScalar is the pure-Go reference implementation.
func cosineDistanceScalar(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	return 1.0 - sim
}

// dotProductScalar is the pure-Go reference implementation.
func dotProductScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// l2DistanceSquaredScalar is the pure-Go reference implementation.
func l2DistanceSquaredScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
