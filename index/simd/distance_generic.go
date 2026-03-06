//go:build !amd64

package simd

// Scalar fallback implementations for non-amd64 architectures.

func dotProductF32(a, b []float32) float32 {
	return dotProductScalar(a, b)
}

func l2DistanceSquaredF32(a, b []float32) float32 {
	return l2DistanceSquaredScalar(a, b)
}

func cosineDistanceF32(a, b []float32) float32 {
	return cosineDistanceScalar(a, b)
}
